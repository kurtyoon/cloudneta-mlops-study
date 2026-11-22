"""신분증 탐지 YOLO 모델을 KServe V1 프로토콜로 서빙하는 작은 HTTP 서버.

InferenceService idcard-detector의 kserve-container가 ultralytics 이미지에서 이 파일을
실행한다. 모델은 KServe가 storageUri pvc://models/idcard/<version>을 /mnt/models에
읽기 전용으로 마운트한 것이다. 복사 없이 버킷 PVC의 파일을 바로 읽는다.

엔드포인트:
- GET  /v1/models/<name>          {"name", "ready", "model_version"}. readiness probe도 이것을 쓴다.
- POST /v1/models/<name>:predict  {"instances": [{"image": {"b64": "<PNG 또는 JPEG>"}}]}
  → {"predictions": [[{"name", "class", "confidence", "box": {x1, y1, x2, y2}}], ...]}

표준 라이브러리 http.server만 쓴다. 이미지를 새로 빌드하지 않기 위해서다.
"""
import base64
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_NAME = os.environ.get('MODEL_NAME', 'idcard-detector')
MODEL_DIR = os.environ.get('MODEL_DIR', '/mnt/models')
PORT = int(os.environ.get('PORT', '8080'))

STATE = {'ready': False, 'error': 'loading'}
LOCK = threading.Lock()


def load():
    t0 = time.monotonic()
    from ultralytics import YOLO

    model = YOLO(os.path.join(MODEL_DIR, 'best.pt'))
    meta = {}
    meta_path = os.path.join(MODEL_DIR, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as file:
            meta = json.load(file)
    loaded = time.monotonic() - t0
    # The first predict builds the inference graph and takes seconds; do it before reporting ready.
    from PIL import Image
    model.predict(Image.new('RGB', (640, 640)), verbose=False)
    STATE.update(model=model, meta=meta, ready=True, error=None)
    print(f'loaded {MODEL_DIR}/best.pt version {meta.get("model_version")} in {loaded:.1f}s, '
          f'warm-up done at {time.monotonic() - t0:.1f}s', flush=True)


def predict(instances):
    from PIL import Image

    images = [Image.open(io.BytesIO(base64.b64decode(item['image']['b64']))).convert('RGB') for item in instances]
    with LOCK:
        results = STATE['model'].predict(images, verbose=False)
    return [json.loads(result.to_json()) for result in results]


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == f'/v1/models/{MODEL_NAME}':
            self.send(200 if STATE['ready'] else 503, {
                'name': MODEL_NAME, 'ready': STATE['ready'],
                'model_version': STATE.get('meta', {}).get('model_version'), 'error': STATE['error']})
        else:
            self.send(404, {'error': 'not found'})

    def do_POST(self):
        if self.path != f'/v1/models/{MODEL_NAME}:predict':
            return self.send(404, {'error': 'not found'})
        if not STATE['ready']:
            return self.send(503, {'error': STATE['error']})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            t0 = time.monotonic()
            predictions = predict(body['instances'])
            print(f'predict {len(predictions)} image(s) in {time.monotonic() - t0:.3f}s', flush=True)
            self.send(200, {'predictions': predictions})
        except (KeyError, TypeError, ValueError) as error:
            self.send(400, {'error': f'{type(error).__name__}: {error}'})
        except Exception as error:  # noqa: BLE001
            self.send(500, {'error': f'{type(error).__name__}: {error}'})

    def log_message(self, fmt, *args):
        print(f'{self.address_string()} {fmt % args}', flush=True)


def boot():
    try:
        load()
    except Exception as error:  # noqa: BLE001
        STATE.update(ready=False, error=f'{type(error).__name__}: {error}'[:300])
        print('load failed:', STATE['error'], flush=True)


if __name__ == '__main__':
    threading.Thread(target=boot, daemon=True).start()
    ThreadingHTTPServer(('0.0.0.0', PORT), Handler).serve_forever()

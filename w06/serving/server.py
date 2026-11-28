"""신분증 탐지 YOLO 모델을 KServe V1 프로토콜로 서빙하는 작은 HTTP 서버.

InferenceService idcard-detector의 kserve-container가 ultralytics 이미지에서 이 파일을
실행한다. 모델은 KServe가 storageUri pvc://models/idcard/<version>을 /mnt/models에
읽기 전용으로 마운트한 것이다. 복사 없이 버킷 PVC의 파일을 바로 읽는다.

엔드포인트:
- GET  /v1/models/<name>          {"name", "ready", "model_version"}. readiness probe도 이것을 쓴다.
- POST /v1/models/<name>:predict  {"instances": [{"image": {"b64": "<PNG 또는 JPEG>"}}]}
  → {"predictions": [[{"name", "class", "confidence", "box": {x1, y1, x2, y2}}], ...]}
- GET  /metrics                   Prometheus 텍스트 형식 지표. PodMonitor predictors가 읽는다.
  요청 수(응답 코드별), predict 지연, 모델 준비 여부와 버전, 이미지당 검출 수, 검출 confidence.

표준 라이브러리 http.server만 쓴다. 이미지를 새로 빌드하지 않기 위해서다. 지표도 prometheus_client 없이
같은 ConfigMap의 promtext.py로 만든다.
"""
import base64
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import promtext

MODEL_NAME = os.environ.get('MODEL_NAME', 'idcard-detector')
MODEL_DIR = os.environ.get('MODEL_DIR', '/mnt/models')
PORT = int(os.environ.get('PORT', '8080'))

STATE = {'ready': False, 'error': 'loading'}
LOCK = threading.Lock()

METRICS = promtext.Registry()
REQUESTS = METRICS.counter('predictor_requests', 'Predict requests by HTTP response code', ['model', 'code'])
LATENCY = METRICS.histogram('predictor_request_duration_seconds', 'Time to answer one predict request',
                            ['model'], promtext.LATENCY_BUCKETS)
READY = METRICS.gauge('predictor_model_ready', '1 once the model is loaded and warmed up', ['model'])
INFO = METRICS.gauge('predictor_model_info', 'Always 1; the served model version is in the labels',
                     ['model', 'version'])
DETECTIONS = METRICS.histogram('idcard_detections_per_image', 'ID cards found in one image', ['model'],
                               (0, 1, 2, 3, 5, 10))
# ultralytics drops boxes under its default conf 0.25, so the lowest bucket starts above it.
CONFIDENCE = METRICS.histogram('idcard_detection_confidence', 'Confidence of each detected ID card', ['model'],
                               (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95))
for code in ('200', '400', '500', '503'):
    REQUESTS.labels(MODEL_NAME, code)
for metric in (LATENCY, DETECTIONS, CONFIDENCE):
    metric.labels(MODEL_NAME)
READY.set(0, MODEL_NAME)


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
    READY.set(1, MODEL_NAME)
    INFO.set(1, MODEL_NAME, meta.get('model_version', 'unknown'))
    print(f'loaded {MODEL_DIR}/best.pt version {meta.get("model_version")} in {loaded:.1f}s, '
          f'warm-up done at {time.monotonic() - t0:.1f}s', flush=True)


def predict(instances):
    from PIL import Image

    images = [Image.open(io.BytesIO(base64.b64decode(item['image']['b64']))).convert('RGB') for item in instances]
    with LOCK:
        results = STATE['model'].predict(images, verbose=False)
    predictions = [json.loads(result.to_json()) for result in results]
    for boxes in predictions:
        DETECTIONS.observe(len(boxes), MODEL_NAME)
        for box in boxes:
            CONFIDENCE.observe(box['confidence'], MODEL_NAME)
    return predictions


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, content_type='application/json'):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def answer(self, t0, code, body):
        """A predict response, counted by code and timed from the start of the request."""
        REQUESTS.inc(MODEL_NAME, str(code))
        LATENCY.observe(time.monotonic() - t0, MODEL_NAME)
        self.send(code, body)

    def do_GET(self):
        if self.path == '/metrics':
            self.send(200, METRICS.render(), promtext.CONTENT_TYPE)
        elif self.path == f'/v1/models/{MODEL_NAME}':
            self.send(200 if STATE['ready'] else 503, {
                'name': MODEL_NAME, 'ready': STATE['ready'],
                'model_version': STATE.get('meta', {}).get('model_version'), 'error': STATE['error']})
        else:
            self.send(404, {'error': 'not found'})

    def do_POST(self):
        if self.path != f'/v1/models/{MODEL_NAME}:predict':
            return self.send(404, {'error': 'not found'})
        t0 = time.monotonic()
        if not STATE['ready']:
            return self.answer(t0, 503, {'error': STATE['error']})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            predictions = predict(body['instances'])
            print(f'predict {len(predictions)} image(s) in {time.monotonic() - t0:.3f}s', flush=True)
            self.answer(t0, 200, {'predictions': predictions})
        except (KeyError, TypeError, ValueError) as error:
            self.answer(t0, 400, {'error': f'{type(error).__name__}: {error}'})
        except Exception as error:  # noqa: BLE001
            self.answer(t0, 500, {'error': f'{type(error).__name__}: {error}'})

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

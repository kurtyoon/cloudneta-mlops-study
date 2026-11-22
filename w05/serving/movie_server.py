"""영화 추천 모델(TorchScript)을 KServe V1 프로토콜로 서빙하는 작은 HTTP 서버.

InferenceService movie-recommender의 kserve-container가 ultralytics 이미지의 torch로 이
파일을 실행한다. 모델 디렉터리는 movie-publish가 PVC models의 movie/v<버전>/에 쓴 것이고,
KServe가 /mnt/models에 읽기 전용으로 마운트한다.
- model.pt: TorchScript. forward(user_index, item_index) → 예측 평점. 인덱스는 ID - 1.
- movies.json: 영화 ID → 제목. 추천 후보는 이 목록의 영화다.
- seen.json: 사용자 ID → 평점을 남긴 영화 ID. exclude_seen이면 후보에서 뺀다.
- meta.json: 레지스트리 이름과 버전, MLflow run ID, 지표.

엔드포인트:
- GET  /v1/models/<name>          {"name", "ready", "model_version", "run_id"}. readiness probe도 이것을 쓴다.
- POST /v1/models/<name>:predict  {"instances": [{"user_id": 50, "top_k": 10, "exclude_seen": true}]}
  → {"predictions": [{"user_id", "candidates", "items": [{"movie_id", "title", "score"}, ...]}]}

표준 라이브러리 http.server와 torch만 쓴다.
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_NAME = os.environ.get('MODEL_NAME', 'movie-recommender')
MODEL_DIR = os.environ.get('MODEL_DIR', '/mnt/models')
PORT = int(os.environ.get('PORT', '8080'))
MAX_TOP_K = 100

STATE = {'ready': False, 'error': 'loading'}


def load():
    t0 = time.monotonic()
    import torch

    model = torch.jit.load(os.path.join(MODEL_DIR, 'model.pt'), map_location='cpu')
    model.eval()
    with open(os.path.join(MODEL_DIR, 'meta.json')) as file:
        meta = json.load(file)
    with open(os.path.join(MODEL_DIR, 'movies.json')) as file:
        movies = {int(k): v for k, v in json.load(file).items()}
    with open(os.path.join(MODEL_DIR, 'seen.json')) as file:
        seen = {int(k): set(v) for k, v in json.load(file).items()}
    # Movie IDs above n_items have no embedding row (n_items is the largest ID in the training split).
    candidates = torch.tensor(sorted(m for m in movies if m <= meta['n_items']))
    STATE.update(torch=torch, model=model, meta=meta, movies=movies, seen=seen, candidates=candidates)
    recommend(1, 1, True)  # first call builds the graph; do it before reporting ready
    STATE.update(ready=True, error=None)
    print(f'loaded {MODEL_DIR} {meta["registered_model_name"]} {meta["model_version"]} run {meta["run_id"]}: '
          f'{len(candidates)} candidate movies, {len(seen)} users, in {time.monotonic() - t0:.1f}s', flush=True)


def recommend(user_id, top_k, exclude_seen):
    torch = STATE['torch']
    if not 1 <= user_id <= STATE['meta']['n_users']:
        raise ValueError(f'user_id {user_id} is outside 1..{STATE["meta"]["n_users"]}')
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f'top_k {top_k} is outside 1..{MAX_TOP_K}')
    items = STATE['candidates']
    if exclude_seen:
        seen = STATE['seen'].get(user_id, set())
        items = items[~torch.isin(items, torch.tensor(sorted(seen), dtype=items.dtype))] if seen else items
    with torch.no_grad():
        scores = STATE['model'](torch.full_like(items, user_id - 1), items - 1).squeeze(-1)
    best = torch.topk(scores, min(top_k, len(items)))
    return {
        'user_id': user_id,
        'candidates': len(items),
        'items': [{'movie_id': int(items[i]), 'title': STATE['movies'][int(items[i])], 'score': round(float(s), 4)}
                  for s, i in zip(best.values, best.indices)],
    }


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == f'/v1/models/{MODEL_NAME}':
            meta = STATE.get('meta', {})
            self.send(200 if STATE['ready'] else 503, {
                'name': MODEL_NAME, 'ready': STATE['ready'], 'model_version': meta.get('model_version'),
                'run_id': meta.get('run_id'), 'error': STATE['error']})
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
            predictions = [recommend(int(item['user_id']), int(item.get('top_k', 10)), bool(item.get('exclude_seen', True)))
                           for item in body['instances']]
            print(f'predict {len(predictions)} user(s) in {time.monotonic() - t0:.3f}s', flush=True)
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

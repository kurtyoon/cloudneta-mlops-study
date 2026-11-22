"""KFP 단계 하나가 RunPod GPU Pod에 신분증 탐지 학습을 맡기고 결과를 아티팩트로 회수한다.

GPU Pod에서는 yolo_train.py가 클러스터 CPU 실습과 같은 데이터, 같은 분할, 같은 ultralytics 버전으로 학습한다.

- runpod-train: RunPod REST API로 GPU Pod를 생성하고, 학습이 끝나면 결과를 다운로드한 뒤 Pod를 삭제한다.
- report: 시간과 mAP를 Metrics와 Markdown으로 남긴다.

GPU Pod는 결과 폴더를 HTTP(포트 8000, RunPod 프록시)로 제공한다. 폴더 이름은 run마다 만드는 임의 토큰이다.
네트워크 볼륨과 S3 키는 쓰지 않는다. 클러스터 밖으로 나가는 것은 학습 스크립트와 그 토큰뿐이다.

필요한 Secret: 파이프라인 네임스페이스의 runpod-api (키 RUNPOD_API_KEY).

    pip install kfp==2.9.0 kfp-kubernetes==1.3.0
    python pipeline.py   # pipeline.yaml을 만든다
"""
from pathlib import Path
from typing import NamedTuple

from kfp import compiler, dsl, kubernetes
from kfp.dsl import Input, Markdown, Metrics, Model, Output

# 작은 이미지에 CUDA torch와 ultralytics를 PyPI에서 설치한다. Docker Hub에서 큰 이미지를 받지 않는다.
# ultralytics/ultralytics:8.4.207(amd64)은 3.14GB 레이어 하나를 15분 넘게 받다가 지웠고,
# runpod/pytorch는 압축 10.56GB라 서버에 캐시가 없으면 더 오래 걸린다.
GPU_IMAGE = 'python:3.11-slim'
# torch는 CUDA 12.6 빌드로 고정한다. PyPI 기본은 CUDA 13.2 빌드(2.15.0+cu132)라 서버 드라이버가 못 받칠 수 있다.
# Pod 생성 요청의 gpu.minCudaVersion도 12.6이다. ultralytics는 클러스터 CPU 실습과 같은 8.4.207이다.
# ultralytics 8.4.207은 PyYAML에 의존하지 않아 따로 설치한다. opencv-python은 libGL이 필요해 headless로 바꾼다.
GPU_SETUP = ('pip install -q --root-user-action=ignore torch==2.8.0 torchvision==0.23.0 '
             '--index-url https://download.pytorch.org/whl/cu126 '
             '&& pip install -q --root-user-action=ignore pyyaml scikit-learn ultralytics==8.4.207 '
             '&& pip uninstall -y -q opencv-python && pip install -q --root-user-action=ignore opencv-python-headless')
MIN_CUDA = '12.6'
TRAIN_SCRIPT = (Path(__file__).parent / 'yolo_train.py').read_text()
RUNPOD_SECRET = 'runpod-api'


@dsl.component(base_image='python:3.12-slim')
def runpod_train(train_script: str, train_args: str, image: str, setup: str, min_cuda: str, gpu_types: str,
                 timeout_minutes: int, ready_timeout_minutes: int, model: Output[Model]) -> NamedTuple(
        'Outputs', [('pod_id', str), ('ready_seconds', float), ('total_seconds', float), ('cost_per_hr', float)]):
    import base64
    import json
    import os
    import secrets
    import time
    import urllib.error
    import urllib.request

    # Pod API v2. 범위를 지정한 새 API 키(REST API의 Pods Read/Write)는 예전 rest.runpod.io/v1에서 401이다
    api = 'https://api.runpod.io/v2'
    key = os.environ['RUNPOD_API_KEY']
    # RunPod API와 프록시는 Cloudflare 뒤에 있고, urllib 기본 User-Agent(Python-urllib)는 403 1010으로 막힌다
    agent = {'User-Agent': 'kfp-runpod-runner/1.0'}

    def call(method, path, body=None):
        req = urllib.request.Request(api + path, method=method,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={**agent, 'Authorization': f'Bearer {key}',
                                              'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                text = r.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f'{method} {path}: {e.code} {e.read().decode(errors="replace")[:500]}') from None
        return json.loads(text) if text else {}

    def fetch(url):
        # 200이면 본문, 아직 없으면 None. 프록시가 Pod에 닿기 전의 502, 404도 None이다.
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=agent), timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError:
            return None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            return None

    token = secrets.token_hex(16)
    srv = f'/srv/{token}'
    # 결과 서버를 먼저 띄우고 학습한다. 끝나면 done 또는 failed 파일을 만든다.
    # 실행기가 Pod를 지우지 못했을 때를 대비해 30분 뒤 RunPod가 Pod에 넣어 주는 Pod 전용 키로 스스로 지운다.
    # 이미지에 runpodctl이 없어 python으로 부르고, 그마저 실패하면 컨테이너를 끝낸다.
    self_delete = ('import os,urllib.request as u; u.urlopen(u.Request("https://api.runpod.io/v2/pods/"'
                   '+os.environ["RUNPOD_POD_ID"],method="DELETE",headers={"User-Agent":"kfp-runpod-runner/1.0",'
                   '"Authorization":"Bearer "+os.environ["RUNPOD_API_KEY"]}),timeout=30)')
    start = (
        f'mkdir -p {srv} && : > /srv/index.html && : > {srv}/index.html; '
        '(cd /srv && python3 -m http.server 8000 >/dev/null 2>&1 &); '
        f'echo "$TRAIN_PY" | base64 -d > /train.py; '
        f'if ( {setup} ) > {srv}/log.txt 2>&1 && '
        f'YOLO_CONFIG_DIR=/tmp/ultralytics python3 /train.py --out {srv}/out {train_args} >> {srv}/log.txt 2>&1; '
        f'then touch {srv}/done; else touch {srv}/failed; fi; '
        f"sleep 1800; python3 -c '{self_delete}'; exit 0"
    )
    t0 = time.time()
    # v2의 gpu.id는 하나만 받는다. 앞의 GPU가 없으면 다음 GPU로 다시 요청한다.
    pod, errors = None, []
    for gpu_id in [g.strip() for g in gpu_types.split(',') if g.strip()]:
        try:
            pod = call('POST', '/pods', {
                'name': f'kfp-idcard-{token[:8]}',
                'image': image,
                'gpu': {'id': gpu_id, 'count': 1, 'minCudaVersion': min_cuda},
                'cloud': 'SECURE',
                'disk': 40,
                'ports': ['8000/http'],
                'env': {'TRAIN_PY': base64.b64encode(train_script.encode()).decode()},
                'entrypoint': ['bash', '-c'],
                'cmd': [start],
            })
            break
        except RuntimeError as e:
            errors.append(str(e))
            print(f'create with {gpu_id} failed: {e}', flush=True)
            if ': 401 ' in str(e) or ': 403 ' in str(e):
                break  # 키 문제는 다른 GPU로 바꿔도 같다
    if pod is None:
        raise RuntimeError('no pod created:\n' + '\n'.join(errors))
    pod_id = pod['id']
    print(f'pod {pod_id} created in {time.time() - t0:.1f}s with {gpu_id}, status {pod.get("status")}', flush=True)
    base = f'https://{pod_id}-8000.proxy.runpod.net/{token}'
    ready = None
    cost = float(pod.get('cost') or 0)
    status, checked = pod.get('status'), 0.0
    try:
        while True:
            if time.time() - t0 > timeout_minutes * 60:
                raise TimeoutError(f'no result within {timeout_minutes} min')
            # 이미지 받기가 늦어지면 GPU 요금만 나가므로 일찍 포기한다
            if ready is None and time.time() - t0 > ready_timeout_minutes * 60:
                raise TimeoutError(f'container not up within {ready_timeout_minutes} min (image pull?)')
            if time.time() - checked > 30:
                checked = time.time()
                info = call('GET', f'/pods/{pod_id}')
                cost = float(info.get('cost') or cost)
                if info.get('status') != status:
                    status = info.get('status')
                    print(f'status {status} at {time.time() - t0:.0f}s, cost/hr {cost}', flush=True)
                if status in ('ERROR', 'EXITED', 'TERMINATED'):
                    raise RuntimeError(f'pod {status} before a result')
            if ready is None and fetch(f'{base}/index.html') is not None:
                ready = time.time() - t0
                print(f'result server up after {ready:.0f}s (image pulled, container started)', flush=True)
            if fetch(f'{base}/done') is not None:
                break
            if fetch(f'{base}/failed') is not None:
                log = (fetch(f'{base}/log.txt') or b'').decode(errors='replace')
                raise RuntimeError('training failed on RunPod:\n' + log[-5000:])
            time.sleep(10)
        log = (fetch(f'{base}/log.txt') or b'').decode(errors='replace')
        print(log[-5000:], flush=True)
        result_body = fetch(f'{base}/out/result.json')
        if result_body is None:
            raise RuntimeError('missing result.json')
        result = json.loads(result_body)
        os.makedirs(model.path, exist_ok=True)
        with open(os.path.join(model.path, 'result.json'), 'wb') as f:
            f.write(result_body)
        for run in result['runs']:
            os.makedirs(os.path.join(model.path, f'e{run["epochs"]}'), exist_ok=True)
            for name in ('best.pt', 'results.csv'):
                body = fetch(f'{base}/out/e{run["epochs"]}/{name}')
                if body is None:
                    raise RuntimeError(f'missing e{run["epochs"]}/{name}')
                with open(os.path.join(model.path, f'e{run["epochs"]}', name), 'wb') as f:
                    f.write(body)
    finally:
        try:
            call('DELETE', f'/pods/{pod_id}')
            print(f'pod {pod_id} deleted', flush=True)
        except Exception as e:  # 지우지 못하면 로그에 남기고, 사용자가 콘솔에서 지운다
            print(f'DELETE FAILED for pod {pod_id}: {e}', flush=True)
    total = time.time() - t0
    return (pod_id, round(ready or 0.0, 1), round(total, 1), cost)


@dsl.component(base_image='python:3.12-slim')
def report(gpu_model: Input[Model], ready_seconds: float, total_seconds: float, cost_per_hr: float,
           metrics: Output[Metrics], summary: Output[Markdown]):
    import json
    import os

    r = json.load(open(os.path.join(gpu_model.path, 'result.json')))
    metrics.log_metric('ready_seconds', ready_seconds)
    metrics.log_metric('download_seconds', r['download_seconds'])
    metrics.log_metric('step_seconds', total_seconds)
    metrics.log_metric('cost_usd_estimate', round(cost_per_hr * total_seconds / 3600, 4))
    lines = [f'GPU: {r["gpu"]}, torch {r["torch"]}', '',
             f'split: ' + ', '.join(f'{k} {v["count"]}' for k, v in r['splits'].items()), '',
             '| epoch | 학습 시간(초) | mAP50 | mAP50-95 |', '|---|---|---|---|']
    for run in r['runs']:
        metrics.log_metric(f'e{run["epochs"]}_train_seconds', run['train_seconds'])
        metrics.log_metric(f'e{run["epochs"]}_map50', run['map50'])
        lines.append(f'| {run["epochs"]} | {run["train_seconds"]} | {run["map50"]} | {run["map50_95"]} |')
    lines += ['', f'Pod 준비 {ready_seconds}초, 단계 전체 {total_seconds}초, 시간당 {cost_per_hr}달러']
    with open(summary.path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


@dsl.pipeline(name='idcard-runpod-gpu', description='신분증 탐지 학습을 RunPod GPU에서 돌려 클러스터 CPU run과 비교한다')
def pipeline(epochs: str = '1,5,10', gpu_types: str = 'NVIDIA GeForce RTX 4090,NVIDIA L4', image: str = GPU_IMAGE,
             setup: str = GPU_SETUP, timeout_minutes: int = 40, ready_timeout_minutes: int = 10):
    gpu = runpod_train(train_script=TRAIN_SCRIPT, train_args=f'--epochs {epochs}', image=image, setup=setup, min_cuda=MIN_CUDA,
                       gpu_types=gpu_types, timeout_minutes=timeout_minutes,
                       ready_timeout_minutes=ready_timeout_minutes).set_display_name('runpod-train')
    gpu.set_cpu_request('100m').set_cpu_limit('500m').set_memory_request('256Mi').set_memory_limit('512Mi')
    gpu.set_caching_options(False)
    kubernetes.use_secret_as_env(gpu, secret_name=RUNPOD_SECRET, secret_key_to_env={'RUNPOD_API_KEY': 'RUNPOD_API_KEY'})

    report(gpu_model=gpu.outputs['model'], ready_seconds=gpu.outputs['ready_seconds'],
           total_seconds=gpu.outputs['total_seconds'], cost_per_hr=gpu.outputs['cost_per_hr']).set_display_name('report')


if __name__ == '__main__':
    compiler.Compiler().compile(pipeline_func=pipeline, package_path=str(Path(__file__).parent / 'pipeline.yaml'))

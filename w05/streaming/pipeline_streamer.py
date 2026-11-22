"""safetensors 모델 하나를 세 경로에서 읽는 시간을 잰다.

경로:
- local: Pod의 로컬 디스크(/tmp). 버킷 PVC에서 복사한 뒤 page cache에서 비운다.
- pvc: 버킷 PVC(GeeseFS)의 파일. 경로는 /models/llm/...
- s3: 같은 파일을 S3 객체로 직접 읽는다. s3://csi-volumes/models/llm/...

로더:
- baseline: 파일은 safetensors.torch.load_file, S3는 boto3로 객체 전체를 받은 뒤
  safetensors.torch.load. 한 스트림으로 읽는다.
- streamer: Run:ai Model Streamer(SafetensorsStreamer). 텐서를 여러 스레드로
  동시에 읽는다.

모델은 Qwen2.5-0.5B-Instruct의 model.safetensors(988 MB, Apache-2.0)이고, 처음
한 번 PVC models에 받아 둔다.

경로와 로더의 조합마다 Pod를 따로 띄운다. 각 Pod의 첫 읽기가 cold, 이후가 warm이다.
- local: 읽을 때마다 page cache를 비우므로 모든 읽기가 디스크에서 온다.
- pvc: Pod마다 GeeseFS 프로세스가 새로 뜬다(다른 Pod가 이 prefix를 마운트하지 않을
  때). 첫 읽기는 S3에서 받고, 이후 읽기는 GeeseFS 메모리와 page cache에서 온다.
- s3: 클라이언트 캐시가 없어 매번 S3에서 받는다.
한 Pod에서 두 로더를 번갈아 돌리면 pvc는 첫 로더의 첫 읽기만 cold가 되므로 조합마다
Pod를 나눈다.

노드 한 대라 SeaweedFS의 volume 파일이 노드 page cache에 남아 있을 수 있다. pvc와
s3의 cold 수치에도 이 효과가 섞여 있다.

컴파일: python pipeline_streamer.py 를 실행하면 pipeline_streamer.yaml이 생긴다.
"""
import os
import sys

from kfp import compiler, dsl, kubernetes

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'idcard'))
from pipeline import YOLO_IMAGE  # noqa: E402

MODEL_REPO = 'Qwen/Qwen2.5-0.5B-Instruct'
MODEL_REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
MODEL_DIR = 'llm/qwen2.5-0.5b-instruct'
S3_ENDPOINT = 'http://seaweedfs.kubeflow.svc:9000'


@dsl.component(packages_to_install=['requests'], base_image='python:3.11')
def fetch_model(repo: str, revision: str, model_dir: str):
    import os
    import time

    import requests

    target = os.path.join('/models', model_dir)
    os.makedirs(target, exist_ok=True)
    for name in ['config.json', 'model.safetensors']:
        url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
        path = os.path.join(target, name)
        with requests.head(url, allow_redirects=True, timeout=60) as head:
            head.raise_for_status()
            size = int(head.headers['Content-Length'])
        if os.path.exists(path) and os.path.getsize(path) == size:
            print(f'{name}: present, {size} bytes')
            continue
        t0 = time.monotonic()
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(path + '.part', 'wb') as file:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    file.write(chunk)
        os.replace(path + '.part', path)
        print(f'{name}: fetched {size} bytes in {time.monotonic() - t0:.1f}s')


@dsl.component(base_image=YOLO_IMAGE, packages_to_install=['kfp==2.17.0', '--no-deps'])
def measure_load(source: str, loader: str, model_dir: str, repeats: int, concurrency: int) -> str:
    import gc
    import os
    import shutil
    import subprocess
    import sys
    import time

    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '--quiet', '--no-warn-script-location',
        'runai-model-streamer[s3]==0.17.1', 'safetensors==0.8.0',
    ])
    if concurrency > 0:
        os.environ['RUNAI_STREAMER_CONCURRENCY'] = str(concurrency)

    import boto3
    import safetensors.torch
    from runai_model_streamer import SafetensorsStreamer

    key = f'models/{model_dir}/model.safetensors'
    pvc_path = f'/models/{model_dir}/model.safetensors'
    if source == 'local':
        path = '/tmp/model.safetensors'
        shutil.copyfile(pvc_path, path)
    elif source == 'pvc':
        path = pvc_path
    elif source == 's3':
        path = f's3://csi-volumes/{key}'
    else:
        raise ValueError(f'unknown source {source}')

    def drop_local_cache():
        # The copy above (and each read) leaves the file in the node's page
        # cache; without this, local would measure memory, not the disk.
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

    def baseline():
        if source == 's3':
            body = boto3.client('s3').get_object(Bucket='csi-volumes', Key=key)['Body'].read()
            tensors = safetensors.torch.load(body)
            del body
        else:
            tensors = safetensors.torch.load_file(path)
        return sum(t.numel() * t.element_size() for t in tensors.values())

    def streamer():
        total = 0
        with SafetensorsStreamer() as s:
            s.stream_file(path)
            for _, tensor in s.get_tensors():
                total += tensor.numel() * tensor.element_size()
        return total

    load = {'baseline': baseline, 'streamer': streamer}[loader]
    lines = []
    for run in range(1, repeats + 1):
        if source == 'local':
            drop_local_cache()
        gc.collect()
        t0 = time.monotonic()
        size = load()
        seconds = time.monotonic() - t0
        state = 'cold' if run == 1 or source == 'local' else 'warm'
        line = f'TIMING {source} {loader} {state} run {run} {seconds:.2f}s {size / 1e9:.3f} GB {size / 1e6 / seconds:.0f} MB/s'
        print(line, flush=True)
        lines.append(line)
    return '\n'.join(lines)


@dsl.pipeline(name='model-streamer', description='safetensors 모델을 로컬 디스크, 버킷 PVC, S3에서 기본 로더와 Run:ai Model Streamer로 읽는 시간')
def streamer_pipeline(repeats: int = 3, concurrency: int = 0):
    fetch_op = fetch_model(repo=MODEL_REPO, revision=MODEL_REVISION, model_dir=MODEL_DIR).set_display_name('fetch-model')
    kubernetes.mount_pvc(fetch_op, pvc_name='models', mount_path='/models')
    fetch_op.set_caching_options(False)

    previous = fetch_op
    for source, loader in [(s, l) for s in ['local', 'pvc', 's3'] for l in ['baseline', 'streamer']]:
        op = measure_load(source=source, loader=loader, model_dir=MODEL_DIR, repeats=repeats, concurrency=concurrency)
        op.set_display_name(f'load-{source}-{loader}').after(previous)
        op.set_cpu_request('1').set_cpu_limit('2').set_memory_request('1Gi').set_memory_limit('4Gi')
        op.set_env_variable('OMP_NUM_THREADS', '2')
        op.set_caching_options(False)
        if source in ('local', 'pvc'):
            kubernetes.mount_pvc(op, pvc_name='models', mount_path='/models')
        if source == 's3':
            kubernetes.use_secret_as_env(
                op,
                secret_name='kfp-pipeline-s3',
                secret_key_to_env={'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY'},
            )
            op.set_env_variable('AWS_ENDPOINT_URL', S3_ENDPOINT)
            op.set_env_variable('AWS_REGION', 'us-east-1')
            op.set_env_variable('AWS_EC2_METADATA_DISABLED', 'true')
            op.set_env_variable('RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING', '0')
        previous = op


if __name__ == '__main__':
    compiler.Compiler().compile(pipeline_func=streamer_pipeline, package_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline_streamer.yaml'))

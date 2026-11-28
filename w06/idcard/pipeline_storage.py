"""신분증 탐지 학습, 검증 파이프라인(pipeline.py)을 PVC의 StorageClass만 바꿔 돌릴 수 있게 만든 버전.

비교 대상:
- local-path: RWO PVC. 노드 디스크의 디렉터리다.
- csi-s3: GeeseFS로 마운트한 버킷 PVC. RWX이고 데이터가 csi-volumes 버킷의 객체로 남는다.

- StorageClass와 access mode를 컴파일 시점에 고른다. kfp-kubernetes의 CreatePVC는 이 값을 파이프라인 파라미터로 받지 않는다.
- download와 split 안에서 단계별 시간을 출력한다. Pod 시간에는 이미지 pull과 pip 설치가 섞이기 때문이다.
- validate에서 du 대신 파일 수와 바이트 합계를 센다. GeeseFS의 du는 디렉터리 크기를 0으로 보고한다.

idcard-csi-s3-fsync: csi-s3 파이프라인의 split에서 이동 직후 /data 디렉터리를 fsync하는 버전.
GeeseFS는 rename을 비동기로 S3에 반영하고, 디렉터리 fsync는 그 안의 변경을 모두 S3에 반영할 때까지 기다린다.
split Pod가 타이머 밖에서 더 걸린 시간이 이 반영인지 보려고 fsync 시간을 TIMING으로 잰다.
fsync를 split 다음의 별도 단계로 두면 Pod가 바뀌는 사이 반영이 끝나므로, split과 같은 컨테이너, 같은 GeeseFS 프로세스 안에서 잰다.
split_dataset은 그대로 두어 기존 두 파이프라인의 spec과 PipelineVersion 이름이 바뀌지 않게 했다.

컴파일: python pipeline_storage.py 를 실행하면 pipeline_local-path.yaml, pipeline_csi-s3.yaml, pipeline_csi-s3-fsync.yaml이 생긴다.
"""
import os
import sys

from kfp import compiler, dsl, kubernetes
from kfp.dsl import Artifact, Input, Metrics, Model, Output

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import TRAINING_SCRIPT, YOLO_IMAGE  # noqa: E402

STORAGE = {
    'local-path': ['ReadWriteOnce'],
    'csi-s3': ['ReadWriteMany'],
}


@dsl.component(packages_to_install=['requests'], base_image='python:3.11')
def download_dataset():
    import os
    import tarfile
    import time

    import requests

    url = 'https://manning.box.com/shared/static/coiv3n2t5t0v42xgfhlsfi8bhvd7b441.gz'
    downloaded_file = '/tmp/DATASET.gz'

    t0 = time.monotonic()
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(downloaded_file, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                file.write(chunk)
    t1 = time.monotonic()
    print(f'downloaded {os.path.getsize(downloaded_file)} bytes in {t1 - t0:.1f}s')

    os.makedirs('/data/DATASET', exist_ok=True)
    with tarfile.open(downloaded_file, 'r:gz') as tar:
        members = tar.getmembers()
        tar.extractall('/data/DATASET', filter='data')
    t2 = time.monotonic()
    print(f'TIMING extract {t2 - t1:.1f}s members {len(members)}')
    os.remove(downloaded_file)


@dsl.component(packages_to_install=['scikit-learn'], base_image='python:3.11')
def split_dataset(random_state: int):
    import glob
    import os
    import shutil
    import time

    from sklearn.model_selection import train_test_split

    t0 = time.monotonic()
    base = '/data/DATASET/MINIDATA'
    images = sorted(glob.glob(os.path.join(base, 'images', '*')))
    labels = sorted(glob.glob(os.path.join(base, 'labels', '*')))
    t1 = time.monotonic()
    print(f'TIMING list {t1 - t0:.1f}s')

    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    if not images or [stem(p) for p in images] != [stem(p) for p in labels]:
        raise ValueError(f'images {len(images)} and labels {len(labels)} do not pair up')
    print(f'pairs {len(images)}')

    train_ratio, validation_ratio, test_ratio = 0.75, 0.15, 0.10
    rest_ratio = test_ratio / (test_ratio + validation_ratio)
    x_train, x_rest, y_train, y_rest = train_test_split(images, labels, test_size=1 - train_ratio, random_state=random_state)
    x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size=rest_ratio, random_state=random_state)

    for split, x_files, y_files in [('train', x_train, y_train), ('val', x_val, y_val), ('test', x_test, y_test)]:
        for sub, files in [('images', x_files), ('labels', y_files)]:
            os.makedirs(os.path.join('/data', split, sub), exist_ok=True)
            for src in files:
                shutil.move(src, os.path.join('/data', split, sub, os.path.basename(src)))
        print(f'{split}: {len(x_files)}')
    print(f'TIMING move {time.monotonic() - t1:.1f}s')


@dsl.component(packages_to_install=['scikit-learn'], base_image='python:3.11')
def split_dataset_fsync(random_state: int):
    """split_dataset과 같고, 이동이 끝나면 /data를 fsync해 S3 반영까지 기다린다."""
    import glob
    import os
    import shutil
    import time

    from sklearn.model_selection import train_test_split

    t0 = time.monotonic()
    base = '/data/DATASET/MINIDATA'
    images = sorted(glob.glob(os.path.join(base, 'images', '*')))
    labels = sorted(glob.glob(os.path.join(base, 'labels', '*')))
    t1 = time.monotonic()
    print(f'TIMING list {t1 - t0:.1f}s')

    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    if not images or [stem(p) for p in images] != [stem(p) for p in labels]:
        raise ValueError(f'images {len(images)} and labels {len(labels)} do not pair up')
    print(f'pairs {len(images)}')

    train_ratio, validation_ratio, test_ratio = 0.75, 0.15, 0.10
    rest_ratio = test_ratio / (test_ratio + validation_ratio)
    x_train, x_rest, y_train, y_rest = train_test_split(images, labels, test_size=1 - train_ratio, random_state=random_state)
    x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size=rest_ratio, random_state=random_state)

    for split, x_files, y_files in [('train', x_train, y_train), ('val', x_val, y_val), ('test', x_test, y_test)]:
        for sub, files in [('images', x_files), ('labels', y_files)]:
            os.makedirs(os.path.join('/data', split, sub), exist_ok=True)
            for src in files:
                shutil.move(src, os.path.join('/data', split, sub, os.path.basename(src)))
        print(f'{split}: {len(x_files)}')
    t2 = time.monotonic()
    print(f'TIMING move {t2 - t1:.1f}s')

    fd = os.open('/data', os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print(f'TIMING fsync {time.monotonic() - t2:.1f}s')


@dsl.container_component
def train_model(
    epochs: int,
    batch: int,
    yolo_model_name: str,
    model_output: Output[Model],
    data_yaml: Output[Artifact],
):
    return dsl.ContainerSpec(
        image=YOLO_IMAGE,
        command=['python3', '-c', TRAINING_SCRIPT],
        args=[
            '--train-path', '/data/train',
            '--val-path', '/data/val',
            '--test-path', '/data/test',
            '--epochs', epochs,
            '--batch', batch,
            '--model-name', yolo_model_name,
            '--model-output', model_output.path,
            '--data-yaml', data_yaml.path,
        ],
    )


@dsl.component(base_image=YOLO_IMAGE, packages_to_install=['kfp==2.17.0', '--no-deps'])
def validate_model(
    data_yaml: Input[Artifact],
    model: Input[Model],
    metrics: Output[Metrics],
):
    import os

    from ultralytics import YOLO

    trained = YOLO(os.path.join(model.path, 'best.pt'))
    results = trained.val(
        data=os.path.join(data_yaml.path, 'data.yaml'),
        imgsz=640,
        batch=1,
        device='cpu',
        workers=0,
        project='/tmp/yolo-val',
        verbose=True,
    )
    metrics.log_metric('map50-95', float(results.box.map))
    metrics.log_metric('map50', float(results.box.map50))
    metrics.log_metric('map75', float(results.box.map75))

    for top in sorted(os.listdir('/data')):
        count = size = 0
        for root, _, files in os.walk(os.path.join('/data', top)):
            for name in files:
                count += 1
                size += os.path.getsize(os.path.join(root, name))
        print(f'PVC usage {top}: {count} files {size / 1e6:.1f} MB')


def build(storage_class: str, fsync: bool = False):
    name = f'idcard-{storage_class}' + ('-fsync' if fsync else '')
    description = f'신분증 탐지 YOLO 학습과 검증, 데이터 전달은 {storage_class} PVC' + (', split 안에서 이동 직후 디렉터리 fsync' if fsync else '')

    @dsl.pipeline(name=name, description=description)
    def storage_pipeline(
        epochs: int = 1,
        batch: int = 8,
        random_state: int = 42,
        yolo_model_name: str = 'yolov8n_custom',
    ):
        pvc = kubernetes.CreatePVC(
            pvc_name_suffix='-idcard',
            access_modes=STORAGE[storage_class],
            size='3Gi',
            storage_class_name=storage_class,
        ).set_display_name('create-pvc')
        pvc.set_caching_options(False)

        download_op = download_dataset().set_display_name('download').set_retry(num_retries=2).after(pvc)
        download_op.set_memory_request('512Mi').set_memory_limit('2Gi')

        split_component = split_dataset_fsync if fsync else split_dataset
        split_op = split_component(random_state=random_state).set_display_name('split').after(download_op)

        train_op = train_model(epochs=epochs, batch=batch, yolo_model_name=yolo_model_name).set_display_name('train').after(split_op)
        train_op.set_cpu_request('1').set_cpu_limit('2').set_memory_request('2Gi').set_memory_limit('4Gi')
        train_op.set_env_variable('OMP_NUM_THREADS', '2')
        train_op.set_env_variable('YOLO_CONFIG_DIR', '/tmp/ultralytics')

        validate_op = validate_model(
            data_yaml=train_op.outputs['data_yaml'],
            model=train_op.outputs['model_output'],
        ).set_display_name('validate').after(train_op)
        validate_op.set_cpu_request('500m').set_cpu_limit('1').set_memory_request('1Gi').set_memory_limit('3Gi')
        validate_op.set_env_variable('OMP_NUM_THREADS', '1')
        validate_op.set_env_variable('YOLO_CONFIG_DIR', '/tmp/ultralytics')

        for op in [download_op, split_op, train_op, validate_op]:
            op.set_caching_options(False)
            kubernetes.mount_pvc(op, pvc_name=pvc.outputs['name'], mount_path='/data')

        delete_pvc = kubernetes.DeletePVC(pvc_name=pvc.outputs['name']).set_display_name('delete-pvc').after(validate_op)
        delete_pvc.set_caching_options(False)

    return storage_pipeline


if __name__ == '__main__':
    for storage_class in STORAGE:
        compiler.Compiler().compile(pipeline_func=build(storage_class), package_path=f'pipeline_{storage_class}.yaml')
    compiler.Compiler().compile(pipeline_func=build('csi-s3', fsync=True), package_path='pipeline_csi-s3-fsync.yaml')

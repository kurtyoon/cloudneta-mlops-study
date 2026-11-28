"""저자의 신분증 탐지 학습, 검증 파이프라인을 CPU 전용 arm64 단일 노드에 맞춘 버전.

원본: practical-mlops/object-detection-project의 training_and_validation_pipeline.py

원본과 달라진 점:
- 학습과 검증에 사전 빌드한 arm64 이미지를 쓴다. 학습 단계에서 apt와 pip로 torch를 설치하지 않는다.
- 이미지와 라벨 목록을 정렬해 쌍을 맞추고, 쌍이 어긋나면 실패시킨다. 원본은 glob 순서에 의존한다.
- split에서 파일을 복사하지 않고 옮긴다. 컨테이너 ephemeral 기본 limit 4Gi 안에 들어가게 하기 위해서다.
- 다운로드 후 압축 파일을 지우고, 상태 코드를 확인한다.
- DataLoader worker를 쓰지 않는다. 컨테이너의 /dev/shm은 기본 64MB다.
- 학습 단계에 CPU와 메모리를 지정하고 스레드 수를 CPU limit에 맞춘다.
- split_unit으로 프레임 단위와 영상 단위 분할을 고른다. 원본은 프레임 단위뿐이다.
- 검증 뒤에 mAP50 게이트를 둔다.
- evaluate 단계에서 학습 모델과 사전 학습 가중치를 mAP와 이미지별 추론 결과로 비교한다.
"""
from kfp import compiler, dsl
from kfp.dsl import Artifact, Dataset, Input, Markdown, Metrics, Model, Output

YOLO_IMAGE = 'ultralytics/ultralytics:8.4.207-arm64'


@dsl.component(packages_to_install=['requests'], base_image='python:3.11')
def download_dataset(output_dataset: Output[Dataset], output_dir: str = 'DATASET'):
    import os
    import tarfile

    import requests

    url = 'https://manning.box.com/shared/static/coiv3n2t5t0v42xgfhlsfi8bhvd7b441.gz'
    downloaded_file = 'DATASET.gz'

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(downloaded_file, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                file.write(chunk)
    print(f'downloaded {os.path.getsize(downloaded_file)} bytes')

    extraction_path = os.path.join(output_dataset.path, output_dir)
    os.makedirs(extraction_path, exist_ok=True)
    with tarfile.open(downloaded_file, 'r:gz') as tar:
        tar.extractall(extraction_path, filter='data')
    os.remove(downloaded_file)


@dsl.component(packages_to_install=['scikit-learn'], base_image='python:3.11')
def split_dataset(
    random_state: int,
    split_unit: str,
    input_dataset: Input[Dataset],
    train_dataset: Output[Dataset],
    validation_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
):
    import glob
    import os
    import shutil

    from sklearn.model_selection import train_test_split

    base = os.path.join(input_dataset.path, 'DATASET', 'MINIDATA')
    images = sorted(glob.glob(os.path.join(base, 'images', '*')))
    labels = sorted(glob.glob(os.path.join(base, 'labels', '*')))

    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    if not images or [stem(p) for p in images] != [stem(p) for p in labels]:
        raise ValueError(f'images {len(images)} and labels {len(labels)} do not pair up')
    print(f'pairs {len(images)}')

    train_ratio, validation_ratio, test_ratio = 0.75, 0.15, 0.10
    rest_ratio = test_ratio / (test_ratio + validation_ratio)
    if split_unit == 'frame':
        x_train, x_rest, y_train, y_rest = train_test_split(
            images, labels, test_size=1 - train_ratio, random_state=random_state
        )
        x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size=rest_ratio, random_state=random_state)
    elif split_unit == 'clip':
        # 파일 이름은 <영상>_<프레임>이다. 같은 영상의 연속 프레임이 서로 다른 split에 들어가지 않게 영상 단위로 나눈다.
        clip = lambda p: stem(p).rsplit('_', 1)[0]
        train_clips, rest_clips = train_test_split(
            sorted({clip(p) for p in images}), test_size=1 - train_ratio, random_state=random_state
        )
        val_clips, test_clips = train_test_split(rest_clips, test_size=rest_ratio, random_state=random_state)
        print(f'clips train {sorted(train_clips)} val {sorted(val_clips)} test {sorted(test_clips)}')
        pick = lambda files, wanted: [p for p in files if clip(p) in wanted]
        x_train, y_train = pick(images, train_clips), pick(labels, train_clips)
        x_val, y_val = pick(images, val_clips), pick(labels, val_clips)
        x_test, y_test = pick(images, test_clips), pick(labels, test_clips)
    else:
        raise ValueError(f'unknown split_unit {split_unit}')

    for output, x_files, y_files in [
        (train_dataset, x_train, y_train),
        (validation_dataset, x_val, y_val),
        (test_dataset, x_test, y_test),
    ]:
        for sub, files in [('images', x_files), ('labels', y_files)]:
            os.makedirs(os.path.join(output.path, sub), exist_ok=True)
            for src in files:
                shutil.move(src, os.path.join(output.path, sub, os.path.basename(src)))
        print(f'{output.name}: {len(x_files)}')


TRAINING_SCRIPT = '''
import argparse
import os
import shutil
import time

import yaml
from ultralytics import YOLO

parser = argparse.ArgumentParser()
for name in ['train-path', 'val-path', 'test-path', 'model-name', 'model-output', 'data-yaml']:
    parser.add_argument('--' + name, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--batch', type=int, required=True)
args = parser.parse_args()

data = {
    'train': os.path.join(args.train_path, 'images'),
    'val': os.path.join(args.val_path, 'images'),
    'test': os.path.join(args.test_path, 'images'),
    'nc': 1,
    'names': {0: 'id_card'},
}
os.makedirs(args.data_yaml, exist_ok=True)
data_yaml_path = os.path.join(args.data_yaml, 'data.yaml')
with open(data_yaml_path, 'w') as file:
    yaml.dump(data, file)

for attempt in range(5):
    try:
        model = YOLO('yolov8n.pt')
        break
    except Exception as error:
        if attempt == 4:
            raise
        print(f'weights download failed: {error}')
        time.sleep(30)

start = time.time()
model.train(
    data=data_yaml_path,
    imgsz=640,
    epochs=args.epochs,
    batch=args.batch,
    device='cpu',
    workers=0,
    project='/tmp/yolo',
    name=args.model_name,
)
print(f'train seconds {time.time() - start:.0f}')

os.makedirs(args.model_output, exist_ok=True)
shutil.copy2(model.trainer.best, os.path.join(args.model_output, 'best.pt'))
shutil.copy2(os.path.join(model.trainer.save_dir, 'results.csv'), os.path.join(args.model_output, 'results.csv'))
'''


@dsl.container_component
def train_model(
    epochs: int,
    batch: int,
    yolo_model_name: str,
    train_dataset: Input[Dataset],
    validation_dataset: Input[Dataset],
    test_dataset: Input[Dataset],
    model_output: Output[Model],
    data_yaml: Output[Artifact],
):
    return dsl.ContainerSpec(
        image=YOLO_IMAGE,
        command=['python3', '-c', TRAINING_SCRIPT],
        args=[
            '--train-path', train_dataset.path,
            '--val-path', validation_dataset.path,
            '--test-path', test_dataset.path,
            '--epochs', epochs,
            '--batch', batch,
            '--model-name', yolo_model_name,
            '--model-output', model_output.path,
            '--data-yaml', data_yaml.path,
        ],
    )


# YOLO_IMAGE는 Python 3.13이다. 컴파일에 쓰는 kfp 2.9.0은 Python 3.13 미만만 지원해서 설치되지 않는다.
# 실행 시점의 kfp만 3.13을 지원하는 버전으로 지정한다. 의존성을 함께 받지 않는 것은 SDK 기본 동작과 같다.
@dsl.component(base_image=YOLO_IMAGE, packages_to_install=['kfp==2.17.0', '--no-deps'])
def validate_model(
    data_yaml: Input[Artifact],
    model: Input[Model],
    validation_dataset: Input[Dataset],
    metrics: Output[Metrics],
):
    import os

    from ultralytics import YOLO

    # data.yaml의 val 경로는 학습 단계의 입력 경로다. 같은 아티팩트를 입력으로 받아야 이 경로에 파일이 생긴다.
    print(f'validation dataset {validation_dataset.path}')
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


# 학습한 모델을 val과 test에서, 사전 학습 가중치를 test에서 평가한다. 게이트와 무관한 비교용이다.
# yolov8n.pt는 COCO 80클래스 모델이다. 신분증 클래스를 학습한 적이 없어 낮은 값이 나오는 것이 정상이다.
@dsl.component(base_image=YOLO_IMAGE, packages_to_install=['kfp==2.17.0', '--no-deps'])
def evaluate_model(
    data_yaml: Input[Artifact],
    model: Input[Model],
    validation_dataset: Input[Dataset],
    test_dataset: Input[Dataset],
    metrics: Output[Metrics],
):
    import os
    import time

    from ultralytics import YOLO

    print(f'datasets {validation_dataset.path} {test_dataset.path}')
    data = os.path.join(data_yaml.path, 'data.yaml')

    def evaluate(name, target, split):
        results = target.val(
            data=data, split=split, imgsz=640, batch=1, device='cpu', workers=0, project='/tmp/yolo-eval', name=f'{name}-{split}'
        )
        print(f'EVAL {name} {split} map50 {results.box.map50:.6f} map50-95 {results.box.map:.6f} map75 {results.box.map75:.6f}')
        metrics.log_metric(f'{name}-{split}-map50', float(results.box.map50))
        metrics.log_metric(f'{name}-{split}-map50-95', float(results.box.map))

    def box_iou(truth, box):
        cx, cy, w, h = truth
        left, top, right, bottom = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        overlap_w = max(0.0, min(right, box[2]) - max(left, box[0]))
        overlap_h = max(0.0, min(bottom, box[3]) - max(top, box[1]))
        overlap = overlap_w * overlap_h
        union = w * h + (box[2] - box[0]) * (box[3] - box[1]) - overlap
        return overlap / union if union > 0 else 0.0

    # 책 8.1.7의 추론 비교. mAP와 달리 conf 0.25 이상인 박스만 남겨 이미지마다 최상위 신뢰도와 정답과의 최고 IoU를 본다.
    # 기본 가중치의 클래스는 COCO 이름이라서 IoU는 클래스와 무관하게 계산하고, 최상위 클래스는 따로 센다.
    def inspect(name, target, split, dataset):
        rows = []
        results = target.predict(
            source=os.path.join(dataset.path, 'images'), conf=0.25, imgsz=640, device='cpu', stream=True, verbose=False
        )
        for result in results:
            stem = os.path.splitext(os.path.basename(result.path))[0]
            with open(os.path.join(dataset.path, 'labels', f'{stem}.txt')) as file:
                truths = [[float(value) for value in line.split()[1:5]] for line in file if line.strip()]
            boxes, confs, classes = result.boxes.xyxyn.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()
            top = max(range(len(confs)), key=confs.__getitem__) if confs else None
            rows.append({
                'file': stem,
                'conf': confs[top] if confs else 0.0,
                'iou': max((box_iou(truth, box) for truth in truths for box in boxes), default=0.0),
                'cls': target.names[int(classes[top])] if confs else None,
            })

        count = len(rows)
        counts = {}
        for row in rows:
            if row['cls']:
                counts[row['cls']] = counts.get(row['cls'], 0) + 1
        summary = {
            'no-detection': sum(row['cls'] is None for row in rows),
            'mean-top-conf': sum(row['conf'] for row in rows) / count,
            'iou50-rate': sum(row['iou'] >= 0.5 for row in rows) / count,
            'iou95-rate': sum(row['iou'] >= 0.95 for row in rows) / count,
            'mean-best-iou': sum(row['iou'] for row in rows) / count,
            'min-best-iou': min(row['iou'] for row in rows),
        }
        print(f'INFER {name} {split} images {count} ' + ' '.join(f'{k} {v:.4f}' for k, v in summary.items()) + f' classes {counts}')
        for key, value in summary.items():
            metrics.log_metric(f'{name}-{split}-{key}', float(value))
        for row in sorted(rows, key=lambda row: row['iou'])[:5]:
            print(f'WORST {name} {split} {row["file"]} iou {row["iou"]:.4f} conf {row["conf"]:.4f} cls {row["cls"]}')

    trained = YOLO(os.path.join(model.path, 'best.pt'))
    evaluate('trained', trained, 'val')
    evaluate('trained', trained, 'test')
    inspect('trained', trained, 'val', validation_dataset)
    inspect('trained', trained, 'test', test_dataset)

    os.chdir('/tmp')
    try:
        for attempt in range(5):
            try:
                base = YOLO('yolov8n.pt')
                break
            except Exception as error:
                if attempt == 4:
                    raise
                print(f'weights download failed: {error}')
                time.sleep(30)
        evaluate('base', base, 'test')
        inspect('base', base, 'test', test_dataset)
    except Exception as error:
        print(f'EVAL base test failed: {error}')


# 보고서 작성과 실패 처리를 나눈다. 실패한 단계의 출력 아티팩트는 올라가지 않을 수 있어서,
# 판정 결과를 먼저 성공한 단계에서 남기고 그 다음 단계에서 실패시킨다.
@dsl.component(base_image='python:3.11')
def gate_report(metrics: Input[Metrics], min_map50: float, report: Output[Markdown]) -> bool:
    map50 = float(metrics.metadata['map50'])
    passed = map50 >= min_map50
    lines = [
        '# 검증 게이트',
        '',
        '| 지표 | 값 |',
        '|---|---|',
        f'| mAP50 | {map50:.6f} |',
        f'| mAP50-95 | {float(metrics.metadata["map50-95"]):.6f} |',
        f'| mAP75 | {float(metrics.metadata["map75"]):.6f} |',
        f'| 기준 mAP50 | {min_map50} |',
        f'| 판정 | {"통과" if passed else "미달"} |',
    ]
    with open(report.path, 'w') as file:
        file.write('\n'.join(lines) + '\n')
    print(f'map50 {map50} min {min_map50} passed {passed}')
    return passed


@dsl.component(base_image='python:3.11')
def enforce_gate(passed: bool):
    if not passed:
        raise RuntimeError('mAP50이 기준에 못 미쳤다. gate-report의 report 아티팩트를 본다.')


@dsl.pipeline(name='idcard-train-val', description='신분증 탐지 YOLO 학습과 검증, arm64 CPU')
def pipeline(
    epochs: int = 1,
    batch: int = 8,
    random_state: int = 42,
    yolo_model_name: str = 'yolov8n_custom',
    min_map50: float = 0.5,
    split_unit: str = 'frame',
):
    download_op = download_dataset().set_display_name('download').set_retry(num_retries=2)
    download_op.set_memory_request('512Mi').set_memory_limit('2Gi')

    split_op = split_dataset(random_state=random_state, split_unit=split_unit, input_dataset=download_op.outputs['output_dataset'])
    split_op.set_display_name('split')

    train_op = train_model(
        epochs=epochs,
        batch=batch,
        yolo_model_name=yolo_model_name,
        train_dataset=split_op.outputs['train_dataset'],
        validation_dataset=split_op.outputs['validation_dataset'],
        test_dataset=split_op.outputs['test_dataset'],
    ).set_display_name('train')
    train_op.set_cpu_request('1').set_cpu_limit('2').set_memory_request('2Gi').set_memory_limit('4Gi')
    train_op.set_env_variable('OMP_NUM_THREADS', '2')
    train_op.set_env_variable('YOLO_CONFIG_DIR', '/tmp/ultralytics')

    validate_op = validate_model(
        data_yaml=train_op.outputs['data_yaml'],
        model=train_op.outputs['model_output'],
        validation_dataset=split_op.outputs['validation_dataset'],
    ).set_display_name('validate')
    validate_op.set_cpu_request('500m').set_cpu_limit('1').set_memory_request('1Gi').set_memory_limit('3Gi')
    validate_op.set_env_variable('OMP_NUM_THREADS', '1')
    validate_op.set_env_variable('YOLO_CONFIG_DIR', '/tmp/ultralytics')

    report_op = gate_report(metrics=validate_op.outputs['metrics'], min_map50=min_map50).set_display_name('gate-report')
    enforce_gate(passed=report_op.outputs['Output']).set_display_name('gate')

    evaluate_op = evaluate_model(
        data_yaml=train_op.outputs['data_yaml'],
        model=train_op.outputs['model_output'],
        validation_dataset=split_op.outputs['validation_dataset'],
        test_dataset=split_op.outputs['test_dataset'],
    ).set_display_name('evaluate')
    evaluate_op.set_cpu_request('500m').set_cpu_limit('1').set_memory_request('1Gi').set_memory_limit('3Gi')
    evaluate_op.set_env_variable('OMP_NUM_THREADS', '1')
    evaluate_op.set_env_variable('YOLO_CONFIG_DIR', '/tmp/ultralytics')
    evaluate_op.set_caching_options(False)


if __name__ == '__main__':
    compiler.Compiler().compile(pipeline_func=pipeline, package_path='pipeline.yaml')

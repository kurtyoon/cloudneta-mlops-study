"""RunPod GPU Pod 안에서 신분증 탐지 학습을 클러스터 CPU 실습과 같은 조건으로 돌린다.

클러스터 CPU 실습 파이프라인의 download, split, train을 한 프로세스로 옮겼다.
- 데이터: 같은 주소의 MINI 세트를 받는다.
- 분할: 같은 코드로 나눈다. 정렬한 파일 목록, train 0.75, val 0.15, test 0.10, frame 단위, random_state 42.
- 학습: 같은 yolov8n.pt, imgsz 640, batch 8. device만 GPU 0이다.

--epochs 1,5,10이면 run 셋을 차례로 학습한다. 클러스터에서도 epoch마다 run을 따로 돌렸다.
결과는 --out 아래 result.json과 run별 best.pt, results.csv다.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import tarfile
import time
import urllib.request

import torch
import yaml
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

URL = 'https://manning.box.com/shared/static/coiv3n2t5t0v42xgfhlsfi8bhvd7b441.gz'

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True)
parser.add_argument('--work', default='/work')
parser.add_argument('--epochs', default='1,5,10')
parser.add_argument('--batch', type=int, default=8)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--random-state', type=int, default=42)
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
os.makedirs(args.work, exist_ok=True)
result = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'runs': []}
t = time.time()


def mark(name):
    global t
    now = time.time()
    result[f'{name}_seconds'] = round(now - t, 1)
    print(f'{name} {now - t:.1f}s', flush=True)
    t = now


archive = os.path.join(args.work, 'DATASET.gz')
urllib.request.urlretrieve(URL, archive)
result['archive_bytes'] = os.path.getsize(archive)
extract = os.path.join(args.work, 'DATASET')
with tarfile.open(archive, 'r:gz') as tar:
    tar.extractall(extract, filter='data')
os.remove(archive)
mark('download')

base = os.path.join(extract, 'MINIDATA')
images = sorted(glob.glob(os.path.join(base, 'images', '*')))
labels = sorted(glob.glob(os.path.join(base, 'labels', '*')))
stem = lambda p: os.path.splitext(os.path.basename(p))[0]
if not images or [stem(p) for p in images] != [stem(p) for p in labels]:
    raise ValueError(f'images {len(images)} and labels {len(labels)} do not pair up')
# 비율 식은 클러스터 실습의 split_dataset과 글자 그대로 같다. 부동소수 결과까지 같게 하기 위해서다.
train_ratio, validation_ratio, test_ratio = 0.75, 0.15, 0.10
rest_ratio = test_ratio / (test_ratio + validation_ratio)
x_train, x_rest, y_train, y_rest = train_test_split(images, labels, test_size=1 - train_ratio, random_state=args.random_state)
x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size=rest_ratio, random_state=args.random_state)
splits = {}
for name, xs, ys in [('train', x_train, y_train), ('val', x_val, y_val), ('test', x_test, y_test)]:
    for sub, files in [('images', xs), ('labels', ys)]:
        os.makedirs(os.path.join(args.work, name, sub), exist_ok=True)
        for src in files:
            shutil.move(src, os.path.join(args.work, name, sub, os.path.basename(src)))
    # 클러스터 split과 같은지 대조할 수 있게 파일 이름 목록의 해시를 남긴다
    names = sorted(os.path.basename(p) for p in xs)
    splits[name] = {'count': len(xs), 'sha256': hashlib.sha256('\n'.join(names).encode()).hexdigest()}
result['splits'] = splits
data_yaml = os.path.join(args.work, 'data.yaml')
with open(data_yaml, 'w') as f:
    yaml.dump({'train': os.path.join(args.work, 'train', 'images'), 'val': os.path.join(args.work, 'val', 'images'),
               'test': os.path.join(args.work, 'test', 'images'), 'nc': 1, 'names': {0: 'id_card'}}, f)
mark('split')

for epochs in [int(e) for e in args.epochs.split(',')]:
    model = YOLO('yolov8n.pt')
    start = time.time()
    model.train(data=data_yaml, imgsz=640, epochs=epochs, batch=args.batch, device=0, workers=args.workers,
                project=os.path.join(args.work, 'runs'), name=f'e{epochs}', exist_ok=True)
    train_seconds = time.time() - start
    run_out = os.path.join(args.out, f'e{epochs}')
    os.makedirs(run_out, exist_ok=True)
    shutil.copy2(model.trainer.best, os.path.join(run_out, 'best.pt'))
    shutil.copy2(os.path.join(model.trainer.save_dir, 'results.csv'), os.path.join(run_out, 'results.csv'))
    metrics = model.trainer.metrics or {}
    result['runs'].append({
        'epochs': epochs,
        'train_seconds': round(train_seconds, 1),
        'map50': round(float(metrics.get('metrics/mAP50(B)', 0)), 6),
        'map50_95': round(float(metrics.get('metrics/mAP50-95(B)', 0)), 6),
    })
    print(json.dumps(result['runs'][-1]), flush=True)
    with open(os.path.join(args.out, 'result.json'), 'w') as f:
        json.dump(result, f)

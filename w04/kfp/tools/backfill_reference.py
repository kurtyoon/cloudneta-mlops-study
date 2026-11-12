"""이미 있는 champion run에 드리프트 reference를 넣는다. 3주차 이전에 학습한 run에 한 번만 쓴다.

학습 URL을 run 파라미터로 features.py와 같은 코드로 다시 만들고, 라벨 parquet의 sha256으로 같은 표본인지
확인한다. parquet에 들어간 피처 단계 시각은 run에 없어서, 학습 run 시작 시각부터 1초씩 거슬러 찾는다.
같은 바이트가 나와야 하므로 학습 때와 같은 capstone-kfp 이미지에서 돌린다.

    docker run --rm -v "$PWD:/src" -w /src -e PYTHONPATH=/src -v <원본 CSV 디렉터리>:/data:ro \
        -e MLFLOW_TRACKING_URI=<MLflow 주소> -e MLFLOW_TRACKING_USERNAME=<사용자> \
        -e MLFLOW_TRACKING_PASSWORD=<앱 비밀번호> \
        capstone-kfp:<태그> python tools/backfill_reference.py --raw-csv /data/malicious_phish.csv [--dry-run]

--raw-csv는 로컬 경로나 s3:// URI다(기본값은 파이프라인의 원본 CSV, S3 키가 있어야 읽힌다).
"""
import argparse
import io
import sys

import pandas as pd
from mlflow import MlflowClient

from capstone_kfp import common, config, drift, features, url_drift

PARAMS = ('labels_sha256', 'sample_size', 'seed', 'n_online', 'train_size')


class NoTokens:
    """build_frames가 부르는 토크나이저 메서드 둘만 둔다. 토큰 id는 라벨 parquet에 들어가지 않는다."""

    def tokenize(self, text):
        return []

    def convert_tokens_to_ids(self, tokens):
        return []


def read_raw(path):
    if path.startswith('s3://'):
        return pd.read_csv(io.BytesIO(common.read_bytes(path)))
    return pd.read_csv(path)


def rebuild_labels(raw, params, started, search_seconds):
    """sha256이 labels_sha256과 같아지는 피처 단계 시각을 started부터 search_seconds초 전까지 찾는다."""
    deduped, _ = features.dedupe(raw)
    sample = features.stratified_sample(deduped, int(params['sample_size']), int(params['seed']))
    start = started.floor('s')
    _, labels = features.build_frames(sample, NoTokens(), 0, int(params['seed']), int(params['n_online']), start)
    for back in range(search_seconds + 1):
        # online 행은 그 시각, 나머지는 하루 전이다(build_frames). 둘 다 같이 옮긴다.
        candidate = labels.assign(event_timestamp=labels['event_timestamp'] - pd.Timedelta(seconds=back))
        if common.sha256(common.parquet_bytes(candidate)) == params['labels_sha256']:
            return candidate, start - pd.Timedelta(seconds=back)
    return None, None


def main(argv=None, client=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--model-name', default=config.REGISTERED_MODEL)
    parser.add_argument('--alias', default=config.CHAMPION)
    parser.add_argument('--raw-csv', default=config.RAW_CSV)
    parser.add_argument('--search-minutes', type=int, default=60,
                        help='학습 run 시작 전 몇 분까지 피처 단계 시각을 찾을지')
    parser.add_argument('--dry-run', action='store_true', help='찾기만 하고 run에 쓰지 않는다')
    args = parser.parse_args(argv)

    client = client or MlflowClient()
    version = client.get_model_version_by_alias(args.model_name, args.alias)
    run = client.get_run(version.run_id)
    run_id = run.info.run_id
    print(f'{args.model_name}@{args.alias}: v{version.version}, run {run_id} ({run.info.run_name})')
    if drift.reference_input(client, run_id) is not None:
        print(f'run에 데이터셋 입력 {config.REFERENCE_DATASET}이 이미 있다. 아무것도 하지 않는다')
        return 0

    params = run.data.params
    missing = [k for k in PARAMS if k not in params]
    if missing:
        sys.exit(f'run에 파라미터 {", ".join(missing)}가 없다. KFP 학습 단계(capstone_kfp/train.py)가 만든 run이 '
                 f'아니다. 2주차 노트북 run은 표본을 만드는 방법(중복 제거 없음)과 남긴 값(url_ids.csv의 sha256)이 '
                 f'달라 이 스크립트로 다시 만들지 않는다')

    started = pd.to_datetime(run.info.start_time, unit='ms', utc=True)
    labels, at = rebuild_labels(read_raw(args.raw_csv), params, started, args.search_minutes * 60)
    if labels is None:
        sys.exit(f'학습 run 시작({started}) 전 {args.search_minutes}분 안에서 labels_sha256 '
                 f'{params["labels_sha256"][:12]}…과 같은 라벨 parquet을 만들지 못했다. 원본 CSV, 피처 코드, '
                 f'pandas와 pyarrow 버전(capstone-kfp 이미지)이 학습 때와 같은지 확인한다. '
                 f'run에는 아무것도 쓰지 않았다')
    train_urls = labels.loc[labels['split'] == 'train', 'url']
    if len(train_urls) != int(params['train_size']):
        sys.exit(f'학습 URL {len(train_urls)}개가 run의 train_size {params["train_size"]}와 다르다')
    print(f'labels_sha256 일치: 피처 단계 시각 {at}, 학습 URL {len(train_urls)}개')

    reference = url_drift.url_stats(train_urls)
    if args.dry_run:
        print(reference.describe().round(2).to_string())
        return 0
    url = drift.log_reference(client, run_id, reference)
    print(f'run {run_id}에 {config.REFERENCE_ARTIFACT}와 데이터셋 입력 {config.REFERENCE_DATASET}을 남겼다: {url}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

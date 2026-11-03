"""모든 단계가 같이 쓰는 이름과 주소.

kfp만 설치한 컴파일 환경에서도 pipeline.py가 import하므로 외부 패키지를 import하지 않는다.
"""

LABELS = ['benign', 'phishing', 'defacement', 'malware']  # analysis/preprocessing/util.py 순서
TOKENIZER_REVISION = 'cd5ef92a9fb2f889e972770a36d4ed042daf221e'
BERT_DIR = '/opt/capstone/bert'  # Dockerfile이 위 revision으로 넣은 토크나이저와 config.json

# 클러스터 안 주소. 파이프라인 Pod의 egress가 열린 곳은 여기뿐이다.
S3_ENDPOINT = 'http://seaweedfs-all-in-one.storage.svc.cluster.local:9000'
MLFLOW_TRACKING_URI = 'http://mlflow.mlflow.svc.cluster.local:5000'

# kfp-pipeline 키는 feature-data-sets/capstone/, feature-registry/capstone/, inference-datasets/에만 쓸 수 있다.
RAW_CSV = 's3://feature-data-sets/capstone/raw/malicious_phish.csv'
FEATURES_URI = 's3://feature-data-sets/capstone/url_tokens.parquet'  # Feast 오프라인 스토어
LABELS_URI = 's3://feature-data-sets/capstone/url_labels.parquet'  # 엔티티와 라벨
INFERENCE_PREFIX = 's3://inference-datasets/capstone'  # 아래에 <KFP run id>/

FEATURE_VIEW = 'url_tokens'  # feature_repo/definitions.py
EXPERIMENT = 'capstone-kfp'
REGISTERED_MODEL = 'capstone-url-classifier'
CHAMPION = 'champion'

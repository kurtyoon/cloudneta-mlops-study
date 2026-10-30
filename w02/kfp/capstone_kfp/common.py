"""여러 단계가 같이 쓰는 S3 입출력과 Feast 스토어.

S3 키는 환경 변수 AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY로만 받는다.
"""
import functools
import hashlib
import io
import os
from pathlib import Path

import boto3
import pandas as pd

from capstone_kfp import config

REPO_DIR = Path(__file__).resolve().parent / 'feature_repo'


def _s3_env():
    # boto3, Feast 레지스트리, s3fs, pyarrow가 모두 같은 엔드포인트와 리전을 쓰게 한다.
    # SeaweedFS는 리전을 보지 않는다.
    os.environ.setdefault('AWS_DEFAULT_REGION', 'us-east-1')
    os.environ.setdefault('FEAST_S3_ENDPOINT_URL', config.S3_ENDPOINT)


def split_uri(uri):
    bucket, key = uri.removeprefix('s3://').split('/', 1)
    return bucket, key


def s3():
    _s3_env()
    return boto3.client('s3', endpoint_url=config.S3_ENDPOINT)


def read_bytes(uri):
    bucket, key = split_uri(uri)
    return s3().get_object(Bucket=bucket, Key=key)['Body'].read()


def write_bytes(uri, data, content_type='application/octet-stream'):
    bucket, key = split_uri(uri)
    s3().put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_parquet(uri):
    return pd.read_parquet(io.BytesIO(read_bytes(uri)))


def write_parquet(df, uri):
    """parquet으로 올리고 그 바이트의 sha256을 돌려준다."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    data = buf.getvalue()
    write_bytes(uri, data)
    return sha256(data)


def feature_store():
    from feast import FeatureStore

    _s3_env()
    return FeatureStore(repo_path=str(REPO_DIR))


def hides_valkey_password(func):
    """단계 함수에서 난 예외 메시지의 Valkey 비밀번호를 지운다.

    Feast의 설정 오류 메시지에는 비밀번호가 펼쳐진 connection_string이 들어가고, KFP는 Pod 로그를 보관한다."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            secret = os.environ.get('VALKEY_PASSWORD')
            message = f'{type(e).__name__}: {e}'
            if secret and secret in message:
                raise RuntimeError(message.replace(secret, '[valkey password]')) from None
            raise
    return wrapper

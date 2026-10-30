"""캡스톤의 엔티티와 피처 뷰. 엔티티는 URL 하나의 url_id, 피처는 그 URL의 토큰 id 목록이다."""
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.data_format import ParquetFormat
from feast.types import Array, Int64, String

from capstone_kfp import config

url = Entity(
    name='url',
    join_keys=['url_id'],
    value_type=ValueType.STRING,
    description='URL 문자열의 sha256 앞 16자리 (features.url_id)',
)


def url_tokens_view(source):
    # schema를 모두 적는다. 빠진 칸이 있으면 feast apply가 parquet을 열어 타입을 추론한다.
    return FeatureView(
        name=config.FEATURE_VIEW,
        entities=[url],
        schema=[
            Field(name='url_id', dtype=String),
            Field(name='token_ids', dtype=Array(Int64)),  # max_length에서 자른 토큰 id, 패딩 없음
            Field(name='token_count', dtype=Int64),  # 자르기 전 토큰 수
        ],
        ttl=timedelta(days=365),
        online=True,
        source=source,
    )


url_tokens_source = FileSource(
    name='url_tokens_source',
    path=config.FEATURES_URI,
    file_format=ParquetFormat(),
    timestamp_field='event_timestamp',
    s3_endpoint_override=config.S3_ENDPOINT,
)

url_tokens = url_tokens_view(url_tokens_source)

"""token_ids(Array(Int64))가 오프라인과 온라인 스토어를 지나도 같은 정수 목록으로 돌아오는지.

클러스터 설정(S3, Valkey) 대신 로컬 parquet, 로컬 레지스트리, SQLite 온라인 스토어를 쓴다.
온라인 값은 Redis와 같은 ValueProto로 직렬화된다. Valkey 쪽은 materialize 단계가 실행 중에 확인한다.
"""
from feast import FeatureStore, FileSource, RepoConfig
from feast.data_format import ParquetFormat

from capstone_kfp import config, features
from capstone_kfp.feature_repo import definitions
from capstone_kfp.materialize import FEATURES, online_window, same_tokens

from test_features import NOW


def local_store(tmp_path, parquet):
    cfg = RepoConfig(
        project='capstone_test', provider='local', registry=str(tmp_path / 'registry.db'),
        offline_store={'type': 'file'}, online_store={'type': 'sqlite', 'path': str(tmp_path / 'online.db')},
        entity_key_serialization_version=2,
    )
    store = FeatureStore(repo_path=str(tmp_path), config=cfg)
    source = FileSource(name='url_tokens_source', path=str(parquet), file_format=ParquetFormat(),
                        timestamp_field='event_timestamp')
    store.apply([definitions.url, definitions.url_tokens_view(source)])
    return store


def test_token_ids_round_trip_both_stores(tmp_path, raw, tokenizer):
    feats, labels = features.build_frames(raw, tokenizer, max_length=24, seed=42, n_online=40, now=NOW)
    feats.to_parquet(tmp_path / 'url_tokens.parquet', index=False)
    store = local_store(tmp_path, tmp_path / 'url_tokens.parquet')
    expected = dict(zip(feats['url_id'], feats['token_ids']))

    # 오프라인: 학습 단계와 같은 엔티티 df(라벨 parquet의 url_id, event_timestamp)
    entity_df = labels[['url_id', 'event_timestamp', 'label', 'split']]
    hist = store.get_historical_features(entity_df=entity_df, features=FEATURES).to_df()
    assert len(hist) == len(labels) and hist['token_ids'].notna().all()
    assert all(same_tokens(expected[i], t) for i, t in zip(hist['url_id'], hist['token_ids']))
    counts = dict(zip(feats['url_id'], feats['token_count']))
    assert all(counts[i] == c for i, c in zip(hist['url_id'], hist['token_count']))

    # 온라인: materialize 단계와 같은 구간 → online 행만 올라간다
    start, end = online_window(labels)
    store.materialize(start_date=start, end_date=end, feature_views=[config.FEATURE_VIEW])
    online_ids = labels.loc[labels['online'], 'url_id'].tolist()
    got = store.get_online_features(features=FEATURES, entity_rows=[{'url_id': i} for i in online_ids]).to_dict()
    assert all(same_tokens(expected[i], t) for i, t in zip(got['url_id'], got['token_ids']))
    assert all(type(x) is int for x in got['token_ids'][0])
    others = labels.loc[~labels['online'], 'url_id'].tolist()
    missing = store.get_online_features(features=FEATURES, entity_rows=[{'url_id': i} for i in others]).to_dict()
    assert all(t is None for t in missing['token_ids'])

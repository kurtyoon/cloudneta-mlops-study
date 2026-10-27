"""4단계: materialize. 오프라인 스토어의 피처를 온라인 스토어(Valkey)에 올린다.

Valkey는 CH4, CH5 프로젝트와 같이 쓰고 maxmemory 96 MB, noeviction이다(꽉 차면 쓰기가 실패한다).
그래서 추론 요청으로 쓸 행(라벨 parquet의 online)만 올린다. materialize는 기간으로만 고르므로,
피처 단계가 그 행들에만 이번 run 시각을 주었다. 올리기 전에 이 뷰의 이전 키를 지워
Valkey에는 마지막 run의 키만 남긴다. 지우는 범위는 프로젝트 capstone의 url_tokens 키뿐이다.
"""
from datetime import timedelta

from capstone_kfp import common, config

FEATURES = [f'{config.FEATURE_VIEW}:token_ids', f'{config.FEATURE_VIEW}:token_count']


def online_window(labels):
    """online 행의 event_timestamp를 모두 담는 [start, end). Feast는 start <= t < end로 읽는다."""
    ts = labels.loc[labels['online'], 'event_timestamp']
    return ts.min().to_pydatetime(), ts.max().to_pydatetime() + timedelta(seconds=1)


def same_tokens(offline, online):
    """오프라인(parquet)과 온라인(Valkey)에서 읽은 토큰 id가 같은 정수 목록인지."""
    return online is not None and [int(x) for x in offline] == [int(x) for x in online]


def read_online(store, url_ids):
    return store.get_online_features(features=FEATURES, entity_rows=[{'url_id': i} for i in url_ids]).to_dict()


def footprint(store):
    """Valkey에서 프로젝트 capstone의 키 수와 MEMORY USAGE 합(바이트).
    Feast의 Redis 키는 엔티티 키 직렬화 뒤에 프로젝트 이름이 붙는다(feast 0.40.1)."""
    from feast.infra.online_stores.redis import RedisOnlineStore

    client = RedisOnlineStore()._get_client(store.config.online_store)
    keys = list(client.scan_iter(match=b'*' + store.config.project.encode(), count=1000))
    return len(keys), int(sum(client.memory_usage(k) or 0 for k in keys))


@common.hides_valkey_password
def run():
    from feast.infra.online_stores.redis import RedisOnlineStore

    labels = common.read_parquet(config.LABELS_URI)
    features = common.read_parquet(config.FEATURES_URI).set_index('url_id')
    start, end = online_window(labels)
    online_ids = labels.loc[labels['online'], 'url_id'].tolist()

    store = common.feature_store()
    view = store.get_feature_view(config.FEATURE_VIEW)
    RedisOnlineStore().delete_table(store.config, view)
    store.materialize(start_date=start, end_date=end, feature_views=[config.FEATURE_VIEW])

    # 올라간 값이 오프라인 스토어의 값과 같은지 online 행 모두 확인한다.
    got = read_online(store, online_ids)
    matched = sum(same_tokens(features.at[i, 'token_ids'], t) for i, t in zip(got['url_id'], got['token_ids']))
    if matched != len(online_ids):
        raise RuntimeError(f'온라인 스토어 값이 parquet과 다르다: {matched}/{len(online_ids)}')
    # online이 아닌 행은 올라가지 않았어야 한다(기간으로 고른 결과). 몇 개만 본다.
    others = labels.loc[~labels['online'], 'url_id'].head(5).tolist()
    if others and any(t is not None for t in read_online(store, others)['token_ids']):
        raise RuntimeError('online이 아닌 행이 온라인 스토어에 올라갔다')

    keys, nbytes = footprint(store)
    result = {'window_start': start.isoformat(), 'window_end': end.isoformat(), 'online_rows': len(online_ids),
              'matched': int(matched), 'keys': keys, 'bytes': nbytes}
    print(result)
    return result

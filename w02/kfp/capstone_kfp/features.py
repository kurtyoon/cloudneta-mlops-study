"""1단계: 피처 생성. 원본 CSV → 중복 제거 → 표본 → 토큰 id → 오프라인 스토어 → feast apply.

표본, 토큰화, train/test 분할은 analysis/의 preprocessing.py, classifier.py와 같은 방식이다.
"""
import hashlib
import io

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from capstone_kfp import common, config


def url_id(url):
    """URL 문자열의 sha256 앞 16자리. 추론 요청에 URL만 와도 같은 키로 온라인 스토어를 찾는다."""
    return hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]


def dedupe(df):
    """url_id가 엔티티 키라 URL 하나에 행 하나만 남긴다. 라벨이 서로 다른 URL은 모두 뺀다."""
    conflicting = df.groupby('url')['type'].transform('nunique') > 1
    kept = df[~conflicting].drop_duplicates('url')
    stats = {
        'raw_rows': len(df),
        'conflicting_urls': int(df.loc[conflicting, 'url'].nunique()),
        'duplicate_rows': int(len(df) - int(conflicting.sum()) - len(kept)),
    }
    return kept, stats


def stratified_sample(df, sample_size, seed):
    if sample_size >= len(df):
        return df
    sample, _ = train_test_split(df, train_size=sample_size, stratify=df['type'], random_state=seed)
    return sample


def assign_split(labels, seed):
    # 분할 결과는 라벨 parquet에 적고 학습 단계가 그대로 쓴다.
    labels = np.asarray(labels)
    _, test_idx = train_test_split(range(len(labels)), test_size=0.2, stratify=labels, random_state=seed)
    split = np.full(len(labels), 'train', dtype=object)
    split[test_idx] = 'test'
    return split


def pick_online(labels, split, n_online, seed):
    """test 중 n_online개를 클래스 비율대로 골라 추론 요청으로 쓴다. 이 행들만 온라인 스토어에 올라간다."""
    labels = np.asarray(labels)
    test_pos = np.flatnonzero(np.asarray(split) == 'test')
    if n_online >= len(test_pos):
        chosen = test_pos
    else:
        try:
            chosen, _ = train_test_split(test_pos, train_size=n_online, stratify=labels[test_pos],
                                         random_state=seed)
        except ValueError:
            # 표본이 작아 클래스별로 나눌 수 없을 때
            chosen = np.random.default_rng(seed).choice(test_pos, n_online, replace=False)
    online = np.zeros(len(labels), dtype=bool)
    online[chosen] = True
    return online


def tokenize(tokenizer, url, max_length):
    """preprocessing.py와 같은 토큰화. 자르기만 하고 패딩은 학습과 추론이 모델 입력을 만들 때 한다."""
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(url))
    return ids[:max_length], len(ids)


def build_frames(df, tokenizer, max_length, seed, n_online, now):
    """피처 parquet과 라벨 parquet에 쓸 DataFrame을 만든다."""
    df = df.reset_index(drop=True)
    labels = df['type'].map(config.LABELS.index).to_numpy()
    split = assign_split(labels, seed)
    online = pick_online(labels, split, n_online, seed)

    # materialize는 event_timestamp 구간으로만 행을 고르므로 online 행에만 이번 run 시각을 준다.
    # Feast의 Redis 쓰기는 기존 키보다 시각이 늦지 않으면 건너뛰므로 시각은 run마다 새로 준다.
    event_timestamp = pd.Series(now - pd.Timedelta(days=1), index=df.index)
    event_timestamp[online] = now

    tokens = [tokenize(tokenizer, u, max_length) for u in df['url']]
    ids = df['url'].map(url_id)
    features = pd.DataFrame({
        'url_id': ids,
        'event_timestamp': event_timestamp,
        'token_ids': [t[0] for t in tokens],
        'token_count': np.array([t[1] for t in tokens], dtype=np.int64),
    })
    labels_df = pd.DataFrame({
        'url_id': ids,
        'url': df['url'],
        'event_timestamp': event_timestamp,
        'label': labels.astype(np.int64),
        'split': split,
        'online': online,
    })
    return features, labels_df


@common.hides_valkey_password
def run(raw_csv, sample_size, seed, max_length, n_online):
    from transformers import BertTokenizer

    from capstone_kfp.feature_repo import definitions

    raw = pd.read_csv(io.BytesIO(common.read_bytes(raw_csv)))
    deduped, stats = dedupe(raw)
    sample = stratified_sample(deduped, sample_size, seed)
    tokenizer = BertTokenizer.from_pretrained(config.BERT_DIR)
    now = pd.Timestamp.now(tz='UTC').floor('s')
    features, labels = build_frames(sample, tokenizer, max_length, seed, n_online, now)
    if not features['url_id'].is_unique:
        raise RuntimeError('url_id가 겹친다')

    features_sha256 = common.write_parquet(features, config.FEATURES_URI)
    labels_sha256 = common.write_parquet(labels, config.LABELS_URI)

    # 뷰를 지우는 apply는 Valkey의 키도 지우므로 이 단계도 VALKEY_PASSWORD를 받는다.
    store = common.feature_store()
    store.apply([definitions.url, definitions.url_tokens])

    result = dict(stats)
    result.update({
        'deduped_rows': len(deduped),
        'rows': len(features),
        'train_rows': int((labels['split'] == 'train').sum()),
        'test_rows': int((labels['split'] == 'test').sum()),
        'online_rows': int(labels['online'].sum()),
        'truncated_urls': int((features['token_count'] > max_length).sum()),
        'class_counts': {config.LABELS[k]: int(v) for k, v in labels['label'].value_counts().sort_index().items()},
        'online_timestamp': now.isoformat(),
        'features_sha256': features_sha256,
        'labels_sha256': labels_sha256,
    })
    print(result)
    return result

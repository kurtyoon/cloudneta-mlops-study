import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from capstone_kfp import config, features
from capstone_kfp.materialize import online_window
from capstone_kfp.train import model_inputs

NOW = pd.Timestamp('2026-10-27T06:00:00Z')


def test_url_id_is_sha256_prefix_and_order_free(raw):
    url = 'http://example.com/login'
    assert features.url_id(url) == hashlib.sha256(url.encode()).hexdigest()[:16]
    ids = raw['url'].map(features.url_id)
    shuffled = raw.sample(frac=1, random_state=0)['url'].map(features.url_id)
    # 행 순서가 바뀌어도 URL마다 같은 id
    assert dict(zip(raw['url'], ids)) == dict(zip(raw.loc[shuffled.index, 'url'], shuffled))
    assert ids.is_unique and ids.str.fullmatch('[0-9a-f]{16}').all()


def test_dedupe_keeps_one_row_per_url_and_drops_conflicts():
    df = pd.DataFrame({'url': ['a', 'a', 'b', 'b', 'c'],
                       'type': ['benign', 'benign', 'benign', 'phishing', 'malware']})
    kept, stats = features.dedupe(df)
    assert kept['url'].tolist() == ['a', 'c']
    assert stats == {'raw_rows': 5, 'conflicting_urls': 1, 'duplicate_rows': 1}


def test_stratified_sample_keeps_ratio_and_seed(raw):
    a = features.stratified_sample(raw, 100, seed=42)
    b = features.stratified_sample(raw, 100, seed=42)
    assert len(a) == 100 and a.index.equals(b.index)
    assert a['type'].value_counts().to_dict() == {'benign': 66, 'phishing': 15, 'defacement': 14, 'malware': 5}
    assert len(features.stratified_sample(raw, 10_000, seed=42)) == len(raw)


def test_split_matches_classifier_py(raw):
    labels = raw['type'].map(config.LABELS.index).to_numpy()
    split = features.assign_split(labels, seed=42)
    # classifier.py: train_test_split(range(n), test_size=0.2, stratify=labels, random_state=seed)
    _, test_idx = train_test_split(range(len(labels)), test_size=0.2, stratify=labels, random_state=42)
    assert set(np.flatnonzero(split == 'test')) == set(test_idx)
    assert (split == 'test').sum() == 80
    for k in range(4):  # 클래스마다 20%
        assert (split[labels == k] == 'test').sum() == round((labels == k).sum() * 0.2)


def test_online_rows_are_stratified_test_rows(raw):
    labels = raw['type'].map(config.LABELS.index).to_numpy()
    split = features.assign_split(labels, seed=42)
    online = features.pick_online(labels, split, n_online=40, seed=42)
    assert online.sum() == 40 and set(split[online]) == {'test'}
    # 클래스마다 test 안의 비율대로(반올림 차이 1까지), 네 클래스 모두 들어간다
    test_counts = np.bincount(labels[split == 'test'], minlength=4)
    assert (np.abs(np.bincount(labels[online], minlength=4) - test_counts * 40 / 80) <= 1).all()
    assert (np.bincount(labels[online], minlength=4) > 0).all()
    # test보다 많이 달라면 test 전부
    assert features.pick_online(labels, split, n_online=10_000, seed=42).sum() == 80


def test_tokenize_truncates_without_padding(tokenizer):
    ids, count = features.tokenize(tokenizer, 'abcdef', max_length=4)
    assert ids == [97, 98, 99, 100] and count == 6
    ids, count = features.tokenize(tokenizer, 'ab', max_length=4)
    assert ids == [97, 98] and count == 2


def test_baked_tokenizer_matches_post2():
    """이미지의 토크나이저가 2편 환경(transformers 4.40.1, 같은 revision)과 같은 id를 낸다.
    기대값은 2편의 venv에서 BertTokenizer.from_pretrained('bert-base-cased', revision=...)로 뽑았다."""
    from transformers import BertTokenizer

    t = BertTokenizer.from_pretrained(config.BERT_DIR)
    ids, count = features.tokenize(t, 'br-icloud.com.br', max_length=128)
    assert ids == [9304, 118, 178, 1665, 24491, 1181, 119, 3254, 119, 9304] and count == 10
    url = 'http://www.garage-pirenne.be/index.php?option=com_content&view=article&id=70&vsig70_0=15'
    ids, count = features.tokenize(t, url, max_length=8)
    assert ids == [8413, 131, 120, 120, 7001, 119, 7419, 118] and count == 40


def test_frames_put_only_online_rows_in_the_materialize_window(raw, tokenizer):
    feats, labels = features.build_frames(raw, tokenizer, max_length=16, seed=42, n_online=40, now=NOW)
    assert list(feats.columns) == ['url_id', 'event_timestamp', 'token_ids', 'token_count']
    assert list(labels.columns) == ['url_id', 'url', 'event_timestamp', 'label', 'split', 'online']
    assert (feats['url_id'] == labels['url_id']).all()
    assert feats['token_ids'].map(len).max() == 16
    start, end = online_window(labels)
    inside = (labels['event_timestamp'] >= start) & (labels['event_timestamp'] < end)
    assert inside.equals(labels['online']) and inside.sum() == 40
    assert set(labels.loc[~labels['online'], 'event_timestamp']) == {NOW - pd.Timedelta(days=1)}


def test_model_inputs_pad_like_classifier_py():
    x = model_inputs([[5, 6, 7], [1, 2, 3, 4, 5, 6]], max_length=4)
    assert x['input_ids'].tolist() == [[5, 6, 7, 0], [1, 2, 3, 4]]
    # 패딩까지 1인 attention_mask: classifier.py 그대로(4주차에 다룬다)
    assert x['attention_mask'].tolist() == [[1, 1, 1, 1], [1, 1, 1, 1]]
    assert x['token_type_ids'].sum() == 0

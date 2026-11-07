"""tools/backfill_reference.py: 피처 단계가 만든 라벨 parquet을 run 파라미터로 다시 만들고, 시각을 찾아
labels_sha256과 맞춘 뒤에만 run에 쓴다."""
import sys
from pathlib import Path

import pandas as pd
import pytest

from capstone_kfp import common, config, features
from test_drift import FakeClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import backfill_reference  # noqa: E402

NOW = pd.Timestamp('2026-10-27T09:01:37Z')  # 피처 단계가 돈 시각(run에는 남지 않는다)


def features_step(raw, tokenizer, sample_size=300, seed=42, n_online=40):
    """features.run과 같은 순서로 라벨 parquet을 만들고 run이 남기는 파라미터를 돌려준다."""
    deduped, _ = features.dedupe(raw)
    sample = features.stratified_sample(deduped, sample_size, seed)
    _, labels = features.build_frames(sample, tokenizer, 16, seed, n_online, NOW)
    params = {'labels_sha256': common.sha256(common.parquet_bytes(labels)), 'sample_size': str(sample_size),
              'seed': str(seed), 'n_online': str(n_online), 'train_size': str(int((labels['split'] == 'train').sum()))}
    return labels, params


def test_rebuild_finds_the_feature_step_time(raw, tokenizer):
    labels, params = features_step(raw, tokenizer)
    started = NOW + pd.Timedelta(seconds=95, milliseconds=420)  # 학습 run은 피처 단계 뒤에 시작한다
    got, at = backfill_reference.rebuild_labels(raw, params, started, search_seconds=600)
    assert at == NOW
    pd.testing.assert_frame_equal(got, labels)


def test_rebuild_gives_up_outside_the_window_or_on_other_data(raw, tokenizer):
    _, params = features_step(raw, tokenizer)
    late = NOW + pd.Timedelta(minutes=30)
    assert backfill_reference.rebuild_labels(raw, params, late, search_seconds=60) == (None, None)
    other = dict(params, seed='7')  # 다른 표본
    assert backfill_reference.rebuild_labels(raw, other, NOW + pd.Timedelta(seconds=5), 60) == (None, None)


class BackfillClient(FakeClient):
    def __init__(self, params, **kw):
        super().__init__(**kw)
        self.params = params

    def get_run(self, run_id):
        run = super().get_run(run_id)
        run.info.run_name = 'kfp-3f0e7468'
        run.info.start_time = int((NOW + pd.Timedelta(seconds=95)).timestamp() * 1000)
        run.data = type('Data', (), {'params': self.params})()
        return run


def write_csv(raw, tmp_path):
    path = tmp_path / 'malicious_phish.csv'
    raw.to_csv(path, index=False)
    return str(path)


def test_main_logs_reference_of_the_train_urls(raw, tokenizer, tmp_path):
    labels, params = features_step(raw, tokenizer)
    client = BackfillClient(params)
    assert backfill_reference.main(['--raw-csv', write_csv(raw, tmp_path), '--search-minutes', '5'], client) == 0
    reference = client.artifacts[config.REFERENCE_ARTIFACT]
    assert len(reference) == int(params['train_size'])
    expected = backfill_reference.url_drift.url_stats(labels.loc[labels['split'] == 'train', 'url'])
    pd.testing.assert_frame_equal(reference, expected)
    assert [d.dataset.name for d in client.inputs] == [config.REFERENCE_DATASET]
    # 두 번째 실행은 아무것도 쓰지 않는다
    assert backfill_reference.main(['--raw-csv', write_csv(raw, tmp_path)], client) == 0
    assert len(client.inputs) == 1


def test_dry_run_writes_nothing(raw, tokenizer, tmp_path):
    _, params = features_step(raw, tokenizer)
    client = BackfillClient(params)
    assert backfill_reference.main(['--raw-csv', write_csv(raw, tmp_path), '--dry-run'], client) == 0
    assert client.artifacts == {} and client.inputs == []


def test_notebook_run_without_labels_sha_is_refused(raw, tmp_path):
    client = BackfillClient({'seed': '42', 'data_sha256': 'e4dea0a6'})
    with pytest.raises(SystemExit, match='labels_sha256'):
        backfill_reference.main(['--raw-csv', write_csv(raw, tmp_path)], client)
    assert client.inputs == []


def test_sha_mismatch_is_refused(raw, tokenizer, tmp_path):
    _, params = features_step(raw, tokenizer)
    client = BackfillClient(dict(params, labels_sha256='0' * 64))
    with pytest.raises(SystemExit, match='아무것도 쓰지 않았다'):
        backfill_reference.main(['--raw-csv', write_csv(raw, tmp_path), '--search-minutes', '1'], client)
    assert client.inputs == [] and client.artifacts == {}

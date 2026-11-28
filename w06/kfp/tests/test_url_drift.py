import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
from scipy import stats as scipy_stats

from capstone_kfp import features, url_drift

SAMPLE = Path(__file__).parent / 'data' / 'url_stats_sample.csv'  # tests/data/make_url_stats_sample.py가 만든다


def stats(url):
    return url_drift.url_stats([url]).iloc[0].to_dict()


def test_columns_and_types():
    df = url_drift.url_stats(['example.com/a', 'http://b.org'])
    assert list(df.columns) == url_drift.COLUMNS
    assert (df.dtypes == 'int64').all() and len(df) == 2
    empty = url_drift.url_stats([])
    assert list(empty.columns) == url_drift.COLUMNS and len(empty) == 0


def test_without_scheme_the_host_is_not_empty():
    # 원본 데이터 대부분이 이런 꼴이다. urlsplit은 호스트를 비우고 전체를 경로로 본다.
    assert stats('mp3raid.com/music/krizz_kaliko.html') == {
        'url_length': 35, 'host_length': 11, 'path_length': 24, 'n_digits': 1, 'n_special': 5, 'n_dots': 2,
        'n_slashes': 2, 'n_equals': 0}


def test_scheme_is_skipped_and_ip_host_with_port_counts_as_host():
    s = stats('http://192.168.0.1:8080/login.php?id=1&x=2#top')
    assert s['host_length'] == len('192.168.0.1:8080')
    assert s['path_length'] == len('/login.php')
    assert s['n_digits'] == 14 and s['n_dots'] == 4 and s['n_equals'] == 2
    assert s['n_slashes'] == 3  # 스킴의 // 도 센다


def test_empty_path_and_query_only():
    assert stats('br-icloud.com.br')['path_length'] == 0
    s = stats('example.com?a=1&b=2')
    assert s['host_length'] == len('example.com') and s['path_length'] == 0 and s['n_equals'] == 2
    assert stats('example.com/?a=1')['path_length'] == 1


def test_special_counts_every_non_ascii_alphanumeric():
    assert stats('a-b_c.d/é')['n_special'] == 5  # - _ . / é
    assert stats('')['url_length'] == 0 and stats('')['host_length'] == 0


def frames(seed=0):
    """reference 2000행, 같은 분포의 current 200행, 드리프트가 난 current 200행.
    칸마다 값이 퍼져 있어야 한다(분산이 0인 칸은 정규화한 거리가 정의되지 않는다)."""
    rng = np.random.default_rng(seed)

    def word(chars, lo, hi):
        return ''.join(rng.choice(list(chars), rng.integers(lo, hi)))

    def urls(n, drifted=False):
        out = []
        for _ in range(n):
            host = '.'.join(word('abcdefgh123', 3, 10) for _ in range(rng.integers(1, 4)))
            path = '/'.join(word('xyz-_09', 2, 8) for _ in range(rng.integers(0, 4)))
            query = '&'.join(f'{word("ab", 1, 3)}={rng.integers(0, 99)}' for _ in range(rng.integers(0, 3)))
            if drifted:  # 긴 쿼리: 숫자와 = 가 훨씬 많다
                query = '&'.join(f'k{rng.integers(1000, 99999)}={rng.integers(10 ** 6, 10 ** 9)}'
                                 for _ in range(rng.integers(3, 8)))
            out.append(f'{host}.com/{path}' + (f'?{query}' if query else ''))
        return out

    return url_drift.url_stats(urls(2000)), url_drift.url_stats(urls(200)), url_drift.url_stats(urls(200, True))


def test_report_same_distribution_has_no_dataset_drift():
    reference, same, _ = frames()
    _, s = url_drift.drift_report(reference, same, 0.5)
    assert s['rows'] == 200 and not s['drift_detected'] and s['drifted_share'] < 0.5
    assert set(s['columns']) == set(url_drift.COLUMNS)
    assert all(0 <= v['p_value'] <= 1 and v['drifted'] == (v['p_value'] < 0.05) for v in s['columns'].values())


def test_report_drifted_frame_is_detected_and_threshold_matters():
    reference, _, drifted = frames()
    snapshot, s = url_drift.drift_report(reference, drifted, 0.5, tags=['kfp'], metadata={'source': 'test'})
    assert s['drift_detected'] and s['drifted_share'] >= 0.5
    assert s['drifted_columns'] == round(s['drifted_share'] * len(url_drift.COLUMNS))
    # 드리프트가 난 칸이 다 그대로여도 기준을 1보다 높이면 데이터셋 드리프트가 아니다
    _, strict = url_drift.drift_report(reference, drifted, 1.01)
    assert strict['drifted_share'] == s['drifted_share'] and not strict['drift_detected']
    # 패널이 읽는 metric이 보고서에 있다(PanelMetric은 이름을 evidently:metric_v2:RowCount 꼴로 바꾸기도 한다)
    types = {m['config']['type'].rsplit(':', 1)[1] for m in snapshot.dict()['metrics']}
    for panel in url_drift.dashboard_panels():
        assert {v.metric.rsplit(':', 1)[-1] for v in panel.values} <= types


@pytest.fixture(scope='module')
def real():
    """원본 CSV에서 피처 단계와 같은 방법으로 뽑은 6,000행의 url_stats. train 4,800행이 reference,
    test 1,200행이 요청을 뽑는 곳이다."""
    df = pd.read_csv(SAMPLE)
    return df[df['split'] == 'train'][url_drift.COLUMNS], df[df['split'] == 'test']


@pytest.mark.parametrize('n', [100, 200])
def test_real_same_distribution_windows_are_not_dataset_drift(real, n):
    # Evidently가 고르는 기본 검정(정규화한 바서슈타인 거리 0.1)으로는 이 20개 중 n=100에서 17개,
    # n=200에서 5개가 데이터셋 드리프트였다. KS p<0.05로는 0개다.
    reference, test = real
    flagged = [url_drift.drift_report(reference, features.stratified_sample(test, n, seed)[url_drift.COLUMNS],
                                      0.5)[1]['drift_detected'] for seed in range(20)]
    assert sum(flagged) <= 1


def test_real_defacement_only_window_is_dataset_drift(real):
    reference, test = real
    defacement = test[test['type'] == 'defacement']
    for seed in range(5):
        _, s = url_drift.drift_report(reference, defacement.sample(100, random_state=seed)[url_drift.COLUMNS], 0.5)
        assert s['drift_detected']


def test_gate_and_bar_panel_use_the_same_ks_test(real):
    reference, test = real
    current = test[test['type'] == 'phishing'].head(100)[url_drift.COLUMNS]
    snapshot, s = url_drift.drift_report(reference, current, 0.5)
    configs = [m['config'] for m in snapshot.dict()['metrics']]
    gate = [c for c in configs if c['type'].endswith(':DriftedColumnsCount')]
    bars = [c for c in configs if c['type'].endswith(':ValueDrift')]
    assert [(c['num_method'], c['num_threshold']) for c in gate] == [('ks', 0.05)]
    assert {(c['method'], c['threshold']) for c in bars} == {('ks', 0.05)} and len(bars) == len(url_drift.COLUMNS)
    # 막대의 값은 scipy의 2표본 KS p-value와 같고, 게이트가 센 칸 수는 p<0.05인 막대 수와 같다
    for c, v in s['columns'].items():
        assert v['p_value'] == pytest.approx(scipy_stats.ks_2samp(reference[c], current[c]).pvalue)
    assert s['drifted_columns'] == sum(v['drifted'] for v in s['columns'].values()) > 0


class FakeDashboard:
    def __init__(self):
        self.panels = []

    def model(self):
        return type('Model', (), {'panels': list(self.panels)})()

    def add_panel(self, panel):
        self.panels.append(panel)


class FakeProject:
    def __init__(self, name):
        self.id, self.name, self.dashboard = f'id-{name}', name, FakeDashboard()


class FakeWorkspace:
    """search_project는 Evidently 서버처럼 이름 일부로도 찾는다."""

    def __init__(self, *names):
        self.projects = [FakeProject(n) for n in names]
        self.created = 0

    def search_project(self, name):
        return [p for p in self.projects if name in p.name]

    def create_project(self, name, description=None):
        self.created += 1
        self.projects.append(FakeProject(name))
        return self.projects[-1]


def test_project_is_created_once_with_all_panels():
    ws = FakeWorkspace('capstone-url-classifier-old')  # 이름 일부만 같은 프로젝트는 고르지 않는다
    p = url_drift.evidently_project(ws, 'capstone-url-classifier')
    assert ws.created == 1 and p.name == 'capstone-url-classifier'
    titles = [x.title for x in p.dashboard.panels]
    assert len(titles) == len(url_drift.dashboard_panels()) == len(set(titles))
    again = url_drift.evidently_project(ws, 'capstone-url-classifier')
    assert again is p and ws.created == 1 and len(p.dashboard.panels) == len(titles)


def test_missing_panels_are_added_to_an_existing_project():
    ws = FakeWorkspace('capstone-url-classifier')
    ws.projects[0].dashboard.add_panel(url_drift.dashboard_panels()[0])
    p = url_drift.evidently_project(ws, 'capstone-url-classifier')
    assert ws.created == 0 and len(p.dashboard.panels) == len(url_drift.dashboard_panels())


def test_oldest_of_same_name_projects_is_used():
    ws = FakeWorkspace('capstone-url-classifier', 'capstone-url-classifier')
    # 서버는 uuid7 id를 주므로 문자열 순서가 만든 순서다. 검색 결과 순서와 상관없이 가장 앞의 것을 고른다
    ws.projects[0].id, ws.projects[1].id = '019a5b00-0000-7000-8000-000000000002', '019a5b00-0000-7000-8000-000000000001'
    assert url_drift.evidently_project(ws, 'capstone-url-classifier') is ws.projects[1] and ws.created == 0


def test_bar_panel_has_one_series_per_column():
    bar = url_drift.dashboard_panels()[-1]
    assert [v.metric_labels for v in bar.values] == [{'column': c} for c in url_drift.COLUMNS]
    assert 'p-value' in bar.title


def test_remote_workspace_gives_up_on_a_server_that_does_not_answer():
    """연결은 받고 응답하지 않는 서버. timeout이 없으면 RemoteWorkspace가 만들 때(verify) 한없이 기다린다."""
    with socket.socket() as server:
        server.bind(('127.0.0.1', 0))
        server.listen()
        started = time.monotonic()
        with pytest.raises(requests.exceptions.ReadTimeout):
            url_drift.remote_workspace(f'http://127.0.0.1:{server.getsockname()[1]}', 'x', timeout=(1, 0.5))
        assert time.monotonic() - started < 5

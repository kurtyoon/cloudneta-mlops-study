"""드리프트 감시의 window와 Evidently 쪽 동작. Evidently 서버 대신 가짜 workspace를 쓴다.

빌드 환경(build-requirements.lock)에 pytest만 더해 pipeline/bento에서 돌린다:

    uv pip install --python .venv/bin/python pytest==8.3.4
    PYTHONPATH=. .venv/bin/python -m pytest -q tests
"""
import threading
import types
from pathlib import Path

import pytest

import drift_monitor
import url_drift

HERE = Path(__file__).resolve().parents[1]


def urls(n, start=0):
    return [f'{"www." * (i % 2)}site{i % 37}.example/{"p/" * (i % 5)}{i}' + '&a=1' * (i % 3)
            for i in range(start, start + n)]


class FakeWorkspace:
    def __init__(self, fail_runs=0, block=None):
        self.runs = []
        self.fail_runs = fail_runs
        self.block = block  # add_run이 이 Event를 기다린다(보고서가 오래 걸리는 경우)
        self.entered = threading.Event()  # add_run에 들어오면 선다
        panels = []
        self.project = types.SimpleNamespace(
            id='project-1', name=drift_monitor.PROJECT,
            dashboard=types.SimpleNamespace(model=lambda: types.SimpleNamespace(panels=list(panels)),
                                            add_panel=panels.append))

    def search_project(self, name):
        return []

    def create_project(self, name, description=None):
        return self.project

    def add_run(self, project_id, snapshot, name=None):
        self.entered.set()
        if self.block is not None:
            self.block.wait(10)
        if self.fail_runs:
            self.fail_runs -= 1
            raise ConnectionError('evidently down')
        self.runs.append((project_id, snapshot, name))


def monitor(ws, window=10, calls=None, **kwargs):
    reference = url_drift.url_stats(urls(300))

    def factory():
        if calls is not None:
            calls.append(1)
        return ws
    return drift_monitor.DriftMonitor(reference, window, factory,
                                      metadata={'source': 'bento', 'model_version': '3'}, **kwargs)


def test_report_only_when_the_window_is_full_and_then_it_empties():
    ws, calls = FakeWorkspace(), []
    m = monitor(ws, window=10, calls=calls)
    assert all(m.add(u) is None for u in urls(9))
    assert calls == [] and ws.runs == []  # window보다 적으면 Evidently를 부르지도 않는다
    summary = m.add('one-more.example/x').result(timeout=30)
    assert summary['rows'] == 10
    ((project_id, snapshot, name),) = ws.runs
    assert project_id == 'project-1' and name == 'bento-v3'
    # 다음 window는 비어서 다시 10건을 모은다(겹치지 않는다)
    assert all(m.add(u) is None for u in urls(9, 100))
    m.add('x.example').result(timeout=30)
    assert len(ws.runs) == 2 and calls == [1]  # workspace는 한 번만 만든다


def fill(m, i, n=4):
    for u in urls(n, 10 * i):
        assert m.add(u) is None
    return m.add(f'w{i}.example')


def test_request_does_not_wait_and_windows_queue_up_to_the_limit():
    gate = threading.Event()
    ws = FakeWorkspace(block=gate)
    m = monitor(ws, window=5, max_pending=1)
    first = fill(m, 0)
    assert ws.entered.wait(30)  # 첫 보고서가 add_run에서 멈춰 있다
    second, third = fill(m, 1), fill(m, 2)
    # 둘째는 줄을 서 있고, 셋째는 줄이 차서 버린다
    assert not first.done() and second is not None and not second.done() and third is None
    gate.set()
    assert [f.result(timeout=30)['rows'] for f in (first, second)] == [5, 5] and len(ws.runs) == 2


def test_worker_thread_does_not_block_shutdown():
    m = monitor(FakeWorkspace(), window=5)
    assert m._thread.daemon and m._thread.is_alive()


@pytest.mark.parametrize('window', [0, 1])
def test_window_must_hold_at_least_two_requests(window):
    with pytest.raises(ValueError, match='2 이상'):
        monitor(FakeWorkspace(), window=window)


def test_evidently_failure_is_logged_and_the_next_window_retries():
    ws, calls = FakeWorkspace(fail_runs=1), []
    m = monitor(ws, window=5, calls=calls)
    assert fill(m, 0).result(timeout=30) is None and ws.runs == []
    assert m._workspace is None and m._project_id is None  # add_run이 실패하면 다음 window에서 다시 찾는다
    assert fill(m, 1).result(timeout=30)['rows'] == 5 and len(ws.runs) == 1
    assert calls == [1, 1]


def test_off_without_evidently_env(monkeypatch, tmp_path, caplog):
    monkeypatch.delenv('EVIDENTLY_URL', raising=False)
    monkeypatch.setenv('EVIDENTLY_SECRET', 'x')
    assert drift_monitor.from_env(tmp_path / 'missing.parquet', {}) is None
    assert '드리프트 감시를 끈다' in caplog.text


def test_on_with_env_reads_reference_and_window(monkeypatch, tmp_path):
    path = tmp_path / 'reference.parquet'
    url_drift.url_stats(urls(50)).to_parquet(path, index=False)
    monkeypatch.setenv('EVIDENTLY_URL', 'http://evidently.test:8000')
    monkeypatch.setenv('EVIDENTLY_SECRET', 'throwaway')
    monkeypatch.setenv('DRIFT_WINDOW', '7')
    made = []
    monkeypatch.setattr(url_drift, 'remote_workspace', lambda url, secret: made.append((url, secret)) or 'ws')
    m = drift_monitor.from_env(path, {'model_version': '3'})
    assert m.window == 7 and len(m.reference) == 50 and m.project == 'capstone-url-classifier'
    assert made == [] and m.workspace_factory() == 'ws'  # 첫 보고서 때 timeout이 있는 클라이언트를 만든다
    assert made == [('http://evidently.test:8000', 'throwaway')]


@pytest.mark.parametrize('window, reference', [('1', True), ('abc', True), ('7', False)])
def test_setup_failure_turns_monitoring_off(monkeypatch, tmp_path, caplog, window, reference):
    path = tmp_path / 'reference.parquet'
    if reference:
        url_drift.url_stats(urls(50)).to_parquet(path, index=False)
    monkeypatch.setenv('EVIDENTLY_URL', 'http://evidently.test:8000')
    monkeypatch.setenv('EVIDENTLY_SECRET', 'throwaway')
    monkeypatch.setenv('DRIFT_WINDOW', window)
    assert drift_monitor.from_env(path, {}) is None
    assert '드리프트 감시를 켜지 못해 끈다' in caplog.text


def test_url_drift_is_the_kfp_copy():
    source = HERE.parents[1] / 'kfp' / 'capstone_kfp' / 'url_drift.py'
    assert (HERE / 'url_drift.py').read_bytes() == source.read_bytes(), (
        'cp kfp/capstone_kfp/url_drift.py pipeline/bento/url_drift.py')


@pytest.mark.parametrize('name', ['service.py', 'drift_monitor.py', 'url_drift.py'])
def test_bentofile_includes_the_service_modules(name):
    assert f'- "/{name}"' in (HERE / 'bentofile.yaml').read_text()


def test_service_runs_one_worker():
    assert "@bentoml.service(name='capstone_url_classifier', workers=1)" in (HERE / 'service.py').read_text()

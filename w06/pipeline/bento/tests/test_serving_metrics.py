"""서빙 지표(serving_metrics.py)와 드리프트 감시가 지표에 남기는 순서.

test_drift_monitor.py와 같은 환경에서 돈다:

    PYTHONPATH=. .venv/bin/python -m pytest -q tests
"""
import os
import subprocess
import sys
from pathlib import Path

from prometheus_client import CollectorRegistry

import serving_metrics
from test_drift_monitor import FakeWorkspace, fill, monitor

HERE = Path(__file__).resolve().parents[1]
LABELS = ['Benign', 'Phishing', 'Defacement', 'Malware']


def value(registry, name, **labels):
    v = registry.get_sample_value(name, labels)
    return 0.0 if v is None else v


def test_label_series_exist_before_the_first_request():
    reg = CollectorRegistry()
    serving_metrics.Metrics(LABELS, registry=reg)
    for label in LABELS:
        assert reg.get_sample_value('capstone_predictions_total', {'prediction': label}) == 0.0
        assert reg.get_sample_value('capstone_prediction_confidence_count', {'prediction': label}) == 0.0
    for result in serving_metrics.WINDOW_RESULTS:
        assert reg.get_sample_value('capstone_drift_windows_total', {'result': result}) == 0.0
    # 판정 전에는 p-value 시계열이 없다. 0으로 두면 모든 칸이 드리프트로 보인다
    assert reg.get_sample_value('capstone_drift_p_value', {'column': 'url_length'}) is None


def test_prediction_counts_and_confidence():
    reg = CollectorRegistry()
    m = serving_metrics.Metrics(LABELS, registry=reg)
    m.prediction('Phishing', 0.97)
    m.prediction('Phishing', 0.42)
    m.prediction('Benign', 0.88)
    assert value(reg, 'capstone_predictions_total', prediction='Phishing') == 2
    assert value(reg, 'capstone_prediction_confidence_bucket', prediction='Phishing', le='0.5') == 1
    assert value(reg, 'capstone_prediction_confidence_sum', prediction='Benign') == 0.88


def summary(detected, share=0.625):
    return {'rows': 100, 'drifted_columns': 5, 'drifted_share': share, 'drift_detected': detected,
            'columns': {'url_length': {'p_value': 0.001, 'drifted': True},
                        'n_dots': {'p_value': 0.4, 'drifted': False}}}


def test_drift_judgment_sets_the_last_window_gauges():
    reg = CollectorRegistry()
    m = serving_metrics.Metrics(LABELS, registry=reg)
    m.drift_judged(summary(True))
    assert value(reg, 'capstone_drift_windows_total', result='drifted') == 1
    assert value(reg, 'capstone_drift_detected') == 1 and value(reg, 'capstone_drift_share') == 0.625
    assert value(reg, 'capstone_drift_p_value', column='url_length') == 0.001
    assert value(reg, 'capstone_drift_last_window_timestamp_seconds') > 0
    m.drift_judged(summary(False, share=0.0))
    assert value(reg, 'capstone_drift_detected') == 0 and value(reg, 'capstone_drift_share') == 0
    assert value(reg, 'capstone_drift_windows_total', result='not_drifted') == 1
    m.drift_judged(None)
    assert value(reg, 'capstone_drift_windows_total', result='error') == 1


def metered(ws, **kwargs):
    reg = CollectorRegistry()
    return reg, monitor(ws, metrics=serving_metrics.Metrics(LABELS, registry=reg), **kwargs)


def judged(reg):
    return sum(value(reg, 'capstone_drift_windows_total', result=r) for r in ('drifted', 'not_drifted'))


def test_judgment_is_recorded_even_when_evidently_is_down():
    reg, m = metered(FakeWorkspace(fail_runs=1), window=5)
    assert fill(m, 0).result(timeout=30) is None
    assert judged(reg) == 1 and value(reg, 'capstone_drift_uploads_total', result='failed') == 1
    assert fill(m, 1).result(timeout=30)['rows'] == 5
    assert judged(reg) == 2 and value(reg, 'capstone_drift_uploads_total', result='ok') == 1


def test_judgment_is_recorded_when_the_workspace_cannot_be_made():
    reg = CollectorRegistry()

    def broken():
        raise ConnectionError('evidently down')

    m = monitor(FakeWorkspace(), window=5, metrics=serving_metrics.Metrics(LABELS, registry=reg))
    m.workspace_factory = broken
    assert fill(m, 0).result(timeout=30) is None
    assert judged(reg) == 1 and value(reg, 'capstone_drift_uploads_total', result='failed') == 1


def test_dropped_window_is_counted():
    import threading

    gate = threading.Event()
    ws = FakeWorkspace(block=gate)
    reg, m = metered(ws, window=5, max_pending=1)
    first = fill(m, 0)
    assert ws.entered.wait(30)
    fill(m, 1)
    assert fill(m, 2) is None
    assert value(reg, 'capstone_drift_windows_dropped_total') == 1
    gate.set()
    first.result(timeout=30)


def test_a_failing_metric_does_not_stop_the_report():
    class Broken:
        def drift_judged(self, summary):
            raise RuntimeError('broken metric')

        def drift_uploaded(self, ok):
            raise RuntimeError('broken metric')

        def drift_dropped(self):
            raise RuntimeError('broken metric')

    ws = FakeWorkspace()
    m = monitor(ws, window=5, metrics=Broken())
    assert fill(m, 0).result(timeout=30)['rows'] == 5 and len(ws.runs) == 1


def run(code, **env):
    return subprocess.run([sys.executable, '-c', code], cwd=HERE, env={**os.environ, **env}, capture_output=True,
                          text=True, check=True).stdout


def test_importing_the_module_does_not_import_prometheus_client():
    # BentoML이 PROMETHEUS_MULTIPROC_DIR을 정하기 전에 prometheus_client가 초기화되면 지표가 /metrics에 없다
    out = run('import sys, serving_metrics, drift_monitor; print("prometheus_client" in sys.modules)')
    assert out.strip() == 'False'


def test_multiprocess_mode_exposes_the_metrics(tmp_path):
    # BentoML 워커와 같은 조건: PROMETHEUS_MULTIPROC_DIR이 정해진 뒤 Metrics를 만들고, 다른 registry가
    # MultiProcessCollector로 모아 읽는다(/metrics가 하는 일).
    out = run('''
import serving_metrics
m = serving_metrics.Metrics(["Benign", "Phishing"])
m.prediction("Phishing", 0.9)
m.drift_judged({"rows": 2, "drifted_columns": 1, "drifted_share": 0.5, "drift_detected": True,
                "columns": {"url_length": {"p_value": 0.01, "drifted": True}}})
m.drift_judged({"rows": 2, "drifted_columns": 0, "drifted_share": 0.0, "drift_detected": False,
                "columns": {"url_length": {"p_value": 0.7, "drifted": False}}})
from prometheus_client import CollectorRegistry, generate_latest, multiprocess
reg = CollectorRegistry()
multiprocess.MultiProcessCollector(reg)
print(generate_latest(reg).decode())
''', PROMETHEUS_MULTIPROC_DIR=str(tmp_path))
    lines = set(out.splitlines())
    assert 'capstone_predictions_total{prediction="Phishing"} 1.0' in lines
    assert 'capstone_drift_windows_total{result="drifted"} 1.0' in lines
    # mostrecent: 마지막 window의 값 하나만 나온다(pid 라벨 없음)
    assert 'capstone_drift_detected 0.0' in lines
    assert 'capstone_drift_p_value{column="url_length"} 0.7' in lines

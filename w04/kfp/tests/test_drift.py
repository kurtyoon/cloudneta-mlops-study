"""드리프트 단계: reference를 run에 남기고(학습, 백필) champion run에서 다시 찾는 길, 그리고 단계 전체.

MLflow 서버 대신 MlflowClient 메서드만 흉내 낸 가짜를 쓴다. 데이터셋 입력은 진짜 mlflow entity다.
"""
import json
import types

import mlflow
import pandas as pd
import pytest
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.entities import DatasetInput, InputTag, RunInputs
from mlflow.exceptions import MlflowException

from capstone_kfp import config, drift, url_drift

ARTIFACT_URI = 'mlflow-artifacts:/2/0123abcd/artifacts'


class FakeClient:
    def __init__(self, champion='3', run_id='0123abcd', inputs=(), artifact_uri=ARTIFACT_URI):
        self.champion, self.run_id, self.artifact_uri = champion, run_id, artifact_uri
        self.inputs = list(inputs)
        self.artifacts = {}  # artifact_path/파일 이름 → DataFrame

    def get_model_version_by_alias(self, name, alias):
        if alias != config.CHAMPION or self.champion is None:
            raise MlflowException('alias not found')
        return types.SimpleNamespace(version=self.champion, run_id=self.run_id)

    def get_run(self, run_id):
        return types.SimpleNamespace(info=types.SimpleNamespace(run_id=run_id, artifact_uri=self.artifact_uri),
                                     inputs=RunInputs(dataset_inputs=list(self.inputs)))

    def log_artifact(self, run_id, local_path, artifact_path=None):
        self.artifacts[f'{artifact_path}/{local_path.rsplit("/", 1)[-1]}'] = pd.read_parquet(local_path)

    def log_inputs(self, run_id, datasets=None, models=None):
        self.inputs.extend(datasets)


def local_input(tmp_path, reference, name=config.REFERENCE_DATASET):
    """로컬 파일을 출처로 한 데이터셋 입력. 클러스터에서는 HTTP 출처지만 읽는 코드는 같다(get_source().load())."""
    path = tmp_path / 'reference.parquet'
    reference.to_parquet(path, index=False)
    dataset = mlflow.data.from_pandas(reference, source=str(path), name=name)
    return DatasetInput(dataset._to_mlflow_entity(), tags=[InputTag('mlflow.data.context', 'training')])


def urls(n):
    # 칸마다 값이 퍼지게 한다(분산이 0인 칸은 정규화한 거리가 정의되지 않는다)
    return [f'{"www." * (i % 2)}site{i % 37}.example/{"p/" * (i % 5)}{i}' + '&a=1' * (i % 3) for i in range(n)]


@pytest.fixture
def reference():
    return url_drift.url_stats(urls(500))


def test_artifact_url_points_at_the_artifact_proxy_of_the_tracking_server():
    assert drift.artifact_url('http://mlflow.test:5000/', ARTIFACT_URI, 'drift/reference.parquet') == (
        'http://mlflow.test:5000/api/2.0/mlflow-artifacts/artifacts/2/0123abcd/artifacts/drift/reference.parquet')
    with pytest.raises(RuntimeError, match='프록시'):
        drift.artifact_url(config.MLFLOW_TRACKING_URI, 's3://mlflow/2/0123abcd/artifacts', 'drift/reference.parquet')


def test_log_reference_writes_artifact_and_training_dataset_input(reference):
    client = FakeClient()
    url = drift.log_reference(client, client.run_id, reference)
    # 기본값은 드리프트 단계가 읽는 클러스터 안 주소다(백필이 밖에서 붙어도 같다)
    assert url.startswith(f'{config.MLFLOW_TRACKING_URI}/api/2.0/mlflow-artifacts/artifacts/2/0123abcd/')
    pd.testing.assert_frame_equal(client.artifacts[config.REFERENCE_ARTIFACT], reference)
    (logged,) = client.inputs
    assert logged.dataset.name == config.REFERENCE_DATASET
    assert [(t.key, t.value) for t in logged.tags] == [('mlflow.data.context', 'training')]
    assert json.loads(logged.dataset.profile)['num_rows'] == len(reference)
    # 읽는 쪽(mlflow.data.get_source)이 같은 HTTP 주소를 돌려받는다
    source = mlflow.data.get_source(logged)
    assert isinstance(source, HTTPDatasetSource) and source.url == url
    assert drift.reference_input(client, client.run_id) is logged


def test_log_reference_uses_the_given_tracking_uri(reference):
    client = FakeClient()
    url = drift.log_reference(client, client.run_id, reference, tracking_uri='http://mlflow.test')
    assert url == f'http://mlflow.test/api/2.0/mlflow-artifacts/artifacts/2/0123abcd/artifacts/{config.REFERENCE_ARTIFACT}'


def test_log_reference_writes_nothing_when_artifacts_are_not_proxied(reference):
    client = FakeClient(artifact_uri='s3://mlflow/2/0123abcd/artifacts')
    with pytest.raises(RuntimeError, match='프록시'):
        drift.log_reference(client, client.run_id, reference)
    assert client.artifacts == {} and client.inputs == []


def test_champion_reference_reads_the_dataset_input_source(tmp_path, reference):
    (tmp_path / 'other').mkdir()
    other = local_input(tmp_path / 'other', reference.head(3), name='something-else')
    client = FakeClient(inputs=[other, local_input(tmp_path, reference)])
    version, got = drift.champion_reference(client, config.REGISTERED_MODEL)
    assert version == '3'
    pd.testing.assert_frame_equal(got, reference)


def test_champion_without_reference_stops_with_backfill_hint():
    with pytest.raises(RuntimeError, match='backfill_reference'):
        drift.champion_reference(FakeClient(inputs=[]), config.REGISTERED_MODEL)


class FakeProject:
    def __init__(self):
        self.id, self.name = 'project-1', config.EVIDENTLY_PROJECT
        self.panels = []
        self.dashboard = types.SimpleNamespace(model=lambda: types.SimpleNamespace(panels=list(self.panels)),
                                               add_panel=self.panels.append)


class FakeWorkspace:
    instances = []

    def __init__(self, url, secret=None):
        self.url, self.secret, self.runs, self.project = url, secret, [], FakeProject()
        FakeWorkspace.instances.append(self)

    def search_project(self, name):
        return []

    def create_project(self, name, description=None):
        return self.project

    def add_run(self, project_id, snapshot, name=None):
        self.runs.append((project_id, snapshot, name))


@pytest.fixture
def step(tmp_path, reference, monkeypatch):
    """drift.run 앞뒤의 MLflow, 라벨 parquet, Evidently를 가짜로 바꾼다. 요청 100건은 reference와 같은 생성기."""
    labels = pd.DataFrame({'url': urls(600), 'online': [i >= 500 for i in range(600)]})
    monkeypatch.setattr(drift, 'MlflowClient', lambda uri: FakeClient(inputs=[local_input(tmp_path, reference)]))
    monkeypatch.setattr(drift.common, 'read_parquet', lambda uri: labels)
    monkeypatch.setattr(url_drift, 'remote_workspace', FakeWorkspace)
    monkeypatch.setenv('EVIDENTLY_SECRET', 'throwaway-secret')

    def run(champion_version='3'):
        return drift.run('http://mlflow.test', config.REGISTERED_MODEL, 'http://evidently.test',
                         config.EVIDENTLY_PROJECT, 0.5, 'abcdef12-run', champion_version)
    return run


def test_run_reports_to_evidently_and_returns_the_verdict(step, reference):
    r = step()
    (ws,) = FakeWorkspace.instances[-1:]
    assert ws.url == 'http://evidently.test' and ws.secret == 'throwaway-secret'
    ((project_id, snapshot, name),) = ws.runs
    assert project_id == 'project-1' and name == 'kfp-abcdef12'
    assert len(ws.project.panels) == len(url_drift.dashboard_panels())
    assert r['rows'] == 100 and r['reference_rows'] == len(reference) and r['champion_version'] == '3'
    assert isinstance(r['drift_detected'], bool) and 0 <= r['drifted_share'] <= 1 and r['uploaded']
    assert 'champion v3' in r['markdown'] and '| n_equals |' in r['markdown'] and 'KS p-value' in r['markdown']
    assert '올리지 못했다' not in r['markdown']


@pytest.mark.parametrize('error', [ConnectionError('connection refused'), TimeoutError('read timed out')])
def test_run_returns_the_verdict_when_evidently_upload_fails(step, monkeypatch, error):
    def unavailable(url, secret):
        raise error
    monkeypatch.setattr(url_drift, 'remote_workspace', unavailable)
    r = step()
    assert r['uploaded'] is False and isinstance(r['drift_detected'], bool) and r['rows'] == 100
    assert '올리지 못했다' in r['markdown'] and type(error).__name__ in r['markdown']
    assert 'throwaway-secret' not in r['markdown']


def test_run_returns_the_verdict_when_the_secret_is_missing(step, monkeypatch):
    monkeypatch.delenv('EVIDENTLY_SECRET')
    r = step()
    assert r['uploaded'] is False and 'EVIDENTLY_SECRET' in r['markdown']


def test_run_stops_when_the_champion_moved_since_promotion(step):
    with pytest.raises(RuntimeError, match='v2'):
        step(champion_version='2')

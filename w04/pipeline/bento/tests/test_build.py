"""build.py가 champion run에서 reference를 받기 전에 확인하는 것. MLflow 서버 대신 가짜 client를 쓴다."""
import types

import pytest

import build


class FakeClient:
    def __init__(self, artifact=True, dataset_input=True):
        self.artifact, self.dataset_input, self.downloaded = artifact, dataset_input, []

    def get_run(self, run_id):
        inputs = [types.SimpleNamespace(dataset=types.SimpleNamespace(name='something-else'))]
        if self.dataset_input:
            inputs.append(types.SimpleNamespace(dataset=types.SimpleNamespace(name=build.REFERENCE_DATASET)))
        return types.SimpleNamespace(inputs=types.SimpleNamespace(dataset_inputs=inputs))

    def list_artifacts(self, run_id, path):
        return [types.SimpleNamespace(path=build.REFERENCE_ARTIFACT)] if self.artifact else []

    def download_artifacts(self, run_id, path, dst_dir):
        self.downloaded.append(path)
        return f'{dst_dir}/{path}'


def test_downloads_when_the_run_has_the_file_and_the_dataset_input(tmp_path):
    client = FakeClient()
    assert build.download_reference(client, 'run-1', str(tmp_path)) == f'{tmp_path}/{build.REFERENCE_ARTIFACT}'
    assert client.downloaded == [build.REFERENCE_ARTIFACT]


@pytest.mark.parametrize('artifact, dataset_input, missing', [
    (False, True, build.REFERENCE_ARTIFACT), (True, False, build.REFERENCE_DATASET), (False, False, 'backfill')])
def test_stops_before_download_when_either_is_missing(tmp_path, artifact, dataset_input, missing):
    client = FakeClient(artifact, dataset_input)
    with pytest.raises(SystemExit, match=missing):
        build.download_reference(client, 'run-1', str(tmp_path))
    assert client.downloaded == []

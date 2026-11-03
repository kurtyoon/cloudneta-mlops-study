import pytest
from mlflow.exceptions import MlflowException

from capstone_kfp import promote


class FakeVersion:
    def __init__(self, version):
        self.version = version


class FakeClient:
    """승격 단계가 부르는 MlflowClient 메서드만 흉내 낸다."""

    def __init__(self, champion):
        self.aliases = {'champion': champion} if champion else {}
        self.tags = {}

    def get_model_version_by_alias(self, name, alias):
        if alias not in self.aliases:
            raise MlflowException('alias not found')
        return FakeVersion(self.aliases[alias])

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[alias] = str(version)

    def set_model_version_tag(self, name, version, key, value):
        self.tags[(version, key)] = value


@pytest.mark.parametrize('macro_f1, threshold, expected', [
    (0.9, 0.9, True),  # 같으면 통과
    (0.8999, 0.9, False),
    (0.9432, 0.9, True),
    (0.0, 0.0, True),
])
def test_gate(macro_f1, threshold, expected):
    assert promote.gate(macro_f1, threshold) is expected


def run_with(monkeypatch, champion, macro_f1, threshold):
    client = FakeClient(champion)
    monkeypatch.setattr(promote, 'MlflowClient', lambda uri: client)
    return client, promote.run('http://mlflow.test', 'capstone-url-classifier', '2', macro_f1, threshold)


def test_pass_moves_champion(monkeypatch):
    client, r = run_with(monkeypatch, champion='1', macro_f1=0.95, threshold=0.9)
    assert r['promoted'] and r['champion_version'] == '2' and r['previous_champion'] == '1'
    assert client.aliases['champion'] == '2'
    assert client.tags[('2', 'gate')].startswith('passed')


def test_fail_keeps_champion(monkeypatch):
    client, r = run_with(monkeypatch, champion='1', macro_f1=0.85, threshold=0.9)
    assert not r['promoted'] and r['champion_version'] == '1'
    assert client.aliases['champion'] == '1'
    assert client.tags[('2', 'gate')].startswith('failed')
    assert 'v1 그대로' in r['message']


def test_fail_without_champion_stops(monkeypatch):
    with pytest.raises(RuntimeError, match='champion이 없어'):
        run_with(monkeypatch, champion=None, macro_f1=0.5, threshold=0.9)

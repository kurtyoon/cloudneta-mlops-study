import pytest

from capstone_kfp import common


def test_password_is_removed_from_errors(monkeypatch):
    monkeypatch.setenv('VALKEY_PASSWORD', 'throwaway-pw-123')
    @common.hides_valkey_password
    def step():
        raise ValueError('bad online_store: valkey:6379,password=throwaway-pw-123')

    with pytest.raises(RuntimeError) as e:
        step()
    assert 'throwaway-pw-123' not in str(e.value)
    assert '[valkey password]' in str(e.value) and 'ValueError' in str(e.value)
    assert e.value.__suppress_context__  # 비밀번호가 든 원래 예외는 traceback에 나오지 않는다


def test_other_errors_pass_through(monkeypatch):
    monkeypatch.setenv('VALKEY_PASSWORD', 'throwaway-pw-123')
    @common.hides_valkey_password
    def step():
        raise KeyError('url_id')

    with pytest.raises(KeyError):
        step()


def test_split_uri():
    assert common.split_uri('s3://feature-data-sets/capstone/url_tokens.parquet') == (
        'feature-data-sets', 'capstone/url_tokens.parquet')


def test_feast_config_error_hides_password(tmp_path, monkeypatch):
    # online_store를 문자열로 잘못 적으면 Feast의 설정 오류 메시지에 펼친 connection_string이 들어간다
    monkeypatch.setenv('VALKEY_PASSWORD', 'throwaway-pw-123')
    (tmp_path / 'feature_store.yaml').write_text(
        'project: capstone\nregistry: registry.db\nprovider: local\n'
        'online_store: "valkey:6379,password=${VALKEY_PASSWORD}"\n')
    monkeypatch.setattr(common, 'REPO_DIR', tmp_path)
    step = common.hides_valkey_password(lambda: common.feature_store().config.online_store)
    with pytest.raises(RuntimeError) as e:
        step()
    assert 'throwaway-pw-123' not in str(e.value) and '[valkey password]' in str(e.value)

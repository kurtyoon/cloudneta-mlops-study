"""파이프라인이 컴파일되고, 온라인 추론이 드리프트 단계의 출력에 걸린 dsl.If 안에 있는지.

컴파일하는 venv(.venv-kfp, kfp 2.9.0 + kfp-kubernetes 1.3.0)에서 돈다. 컴포넌트 이미지에는 kfp가
의존성 없이 들어 있어 컴파일러를 import하지 못하므로 tests/와 따로 둔다:

    uv pip install --python .venv-kfp/bin/python pytest==8.3.4
    .venv-kfp/bin/python -m pytest -q tests_compile
"""
import yaml
from kfp import compiler

import pipeline


def compiled(tmp_path):
    out = tmp_path / 'pipeline.yaml'
    compiler.Compiler().compile(pipeline.make_pipeline('capstone-kfp:test'), str(out))
    return list(yaml.safe_load_all(out.read_text()))


def test_infer_runs_only_without_drift(tmp_path):
    spec, platform = compiled(tmp_path)
    tasks = spec['root']['dag']['tasks']
    assert 'infer-online' not in tasks
    (name,) = [n for n, t in tasks.items() if 'triggerPolicy' in t]
    condition = tasks[name]
    assert condition['triggerPolicy']['condition'] == (
        "inputs.parameter_values['pipelinechannel--detect-drift-drift_detected'] == false")
    assert condition['inputs']['parameters']['pipelinechannel--detect-drift-drift_detected'] == {
        'taskOutputParameter': {'outputParameterKey': 'drift_detected', 'producerTask': 'detect-drift'}}
    assert list(spec['components'][condition['componentRef']['name']]['dag']['tasks']) == ['infer-online']
    # 드리프트 단계는 승격과 materialize 뒤에 돈다
    assert set(tasks['detect-drift']['dependentTasks']) == {'promote-model', 'materialize-online'}


def test_drift_step_gets_only_its_secrets_and_no_tracking(tmp_path):
    spec, platform = compiled(tmp_path)
    executors = platform['platforms']['kubernetes']['deploymentSpec']['executors']
    secrets = {s['secretName'] for s in executors['exec-detect-drift']['secretAsEnv']}
    assert secrets == {'kfp-pipeline-s3', 'evidently-secret'}
    assert all('evidently-secret' not in {s['secretName'] for s in e.get('secretAsEnv', [])}
               for name, e in executors.items() if name != 'exec-detect-drift')
    container = spec['deploymentSpec']['executors']['exec-detect-drift']['container']
    assert {'name': 'DO_NOT_TRACK', 'value': '1'} in container['env']
    assert container['resources']['memoryLimit'] > 0 and container['resources']['cpuLimit'] == 1.0


def test_drift_share_is_a_pipeline_parameter(tmp_path):
    spec, _ = compiled(tmp_path)
    params = spec['root']['inputDefinitions']['parameters']
    assert params['drift_share'] == {'defaultValue': 0.5, 'isOptional': True, 'parameterType': 'NUMBER_DOUBLE'}
    assert params['evidently_url']['defaultValue'] == 'http://evidently.evidently.svc.cluster.local:8000'

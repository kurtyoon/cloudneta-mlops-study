"""컴파일된 파이프라인(IR YAML)의 단계를 KFP 없이 docker 컨테이너로 차례대로 돌린다.

KFP 런처가 하는 일 중 필요한 만큼만 한다: 단계마다 컴파일된 이미지, command, args를 그대로 쓰고,
{{$}} 자리에 executor input(JSON)을 넣는다. 입력 파라미터는 파이프라인 기본값(또는 --param),
앞 단계의 출력, 상수 자리표시자(run id)에서 채우고, 출력 파일과 아티팩트는 공유 볼륨 아래에 둔다.
환경 변수는 kfp-kubernetes가 적은 secretAsEnv를 따라 --secrets 파일의 값으로 넣고,
메모리 상한도 컴파일된 값으로 준다. secretAsEnv가 없는 단계는 그 키를 받지 않는다.

KFP의 드라이버, 런처, Argo, API 서버는 다루지 않는다. 그 부분은 클러스터에서 확인한다.

사용법: python run_compiled.py <pipeline.yaml> --network <docker 네트워크> --secrets <json> --work <디렉터리>
        [--param 이름=값 ...]
"""
import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

import yaml

PLACEHOLDERS = {'{{$.pipeline_job_uuid}}', '{{$.pipeline_job_name}}'}


def topological(tasks):
    order, done = [], set()
    while len(order) < len(tasks):
        ready = [n for n, t in tasks.items() if n not in done and set(t.get('dependentTasks', [])) <= done]
        if not ready:
            sys.exit('순환 의존이 있다')
        for n in sorted(ready):
            order.append(n)
            done.add(n)
    return order


def cast(value, type_name):
    return {'NUMBER_INTEGER': int, 'NUMBER_DOUBLE': float, 'BOOLEAN': lambda v: str(v).lower() == 'true'}.get(
        type_name, str)(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pipeline')
    ap.add_argument('--network', required=True)
    ap.add_argument('--secrets', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--param', action='append', default=[])
    a = ap.parse_args()

    docs = list(yaml.safe_load_all(open(a.pipeline)))
    spec = docs[0]
    executors_k8s = (docs[1] if len(docs) > 1 else {}).get('platforms', {}).get('kubernetes', {}).get(
        'deploymentSpec', {}).get('executors', {})
    secrets = json.load(open(a.secrets))
    root_params = spec['root']['inputDefinitions']['parameters']
    params = {k: v.get('defaultValue') for k, v in root_params.items()}
    for kv in a.param:
        k, v = kv.split('=', 1)
        params[k] = cast(v, root_params[k]['parameterType'])
    run_id = str(uuid.uuid4())
    work = Path(a.work).resolve()
    print(f'run id {run_id}, parameters {params}', flush=True)

    tasks = spec['root']['dag']['tasks']
    outputs = {}  # (task, name) -> 값
    for name in topological(tasks):
        task = tasks[name]
        comp = spec['components'][task['componentRef']['name']]
        exe = comp['executorLabel']
        container = spec['deploymentSpec']['executors'][exe]['container']

        values = {}
        types = {p: d['parameterType'] for p, d in comp.get('inputDefinitions', {}).get('parameters', {}).items()}
        for p, v in task['inputs'].get('parameters', {}).items():
            if 'componentInputParameter' in v:
                values[p] = params[v['componentInputParameter']]
            elif 'taskOutputParameter' in v:
                ref = v['taskOutputParameter']
                values[p] = outputs[(ref['producerTask'], ref['outputParameterKey'])]
            else:
                constant = v['runtimeValue']['constant']
                values[p] = run_id if constant in PLACEHOLDERS else constant
            # IR은 정수 기본값을 42.0처럼 적는다. KFP 드라이버(Go)는 JSON에 42로 넘기므로 같게 맞춘다.
            if types.get(p) == 'NUMBER_INTEGER':
                values[p] = int(values[p])
        task_dir = work / name
        task_dir.mkdir(parents=True, exist_ok=True)
        out_defs = comp.get('outputDefinitions', {})
        executor_input = {
            'inputs': {'parameterValues': values},
            'outputs': {
                'parameters': {p: {'outputFile': str(task_dir / 'parameters' / p)}
                               for p in out_defs.get('parameters', {})},
                'artifacts': {art: {'artifacts': [{'name': art, 'type': d['artifactType'],
                                                   'uri': str(task_dir / 'artifacts' / art), 'metadata': {}}]}
                              for art, d in out_defs.get('artifacts', {}).items()},
                'outputFile': str(task_dir / 'executor_output.json'),
            },
        }
        argv = [json.dumps(executor_input) if x == '{{$}}' else x
                for x in container.get('command', []) + container.get('args', [])]

        env = []
        for s in executors_k8s.get(exe, {}).get('secretAsEnv', []):
            for kv in s['keyToEnv']:
                env += ['-e', f"{kv['envVar']}={secrets[s['secretName']][kv['secretKey']]}"]
        resources = container.get('resources', {})
        limit = ['--memory', f"{int(resources['memoryLimit'] * 1e9)}b"] if 'memoryLimit' in resources else []
        names = [e.split('=', 1)[0] for e in env if e != '-e']
        print(f'== {name}: {container["image"]}, env from Secret {names or "없음"}, '
              f'memory limit {limit[1] if limit else "없음"}', flush=True)
        r = subprocess.run(['docker', 'run', '--rm', '--name', f'kfp-{name}-{run_id[:8]}', '--network', a.network,
                            '-v', f'{work}:{work}', *limit, *env, container['image'], *argv])
        if r.returncode != 0:
            sys.exit(f'{name} 실패 (exit {r.returncode})')

        result = json.loads((task_dir / 'executor_output.json').read_text())
        for p, v in result.get('parameterValues', {}).items():
            outputs[(name, p)] = v
        for art, lst in result.get('artifacts', {}).items():
            for x in lst.get('artifacts', []):
                if x.get('metadata'):
                    print(f'   {art}: {json.dumps(x["metadata"], ensure_ascii=False)}')
        print(f'   outputs: {result.get("parameterValues", {})}', flush=True)
    print(json.dumps({'run_id': run_id, 'outputs': {f'{t}.{p}': v for (t, p), v in outputs.items()}},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()

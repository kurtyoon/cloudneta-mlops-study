"""컴파일한 pipeline.yaml을 KFP에 올리고 run을 만든다. 같은 이름의 파이프라인이 있으면 새 버전으로 올린다.

KFP 주소는 authentik forward auth 뒤에 있어 요청마다 authentik 사용자 이름과 app password를 HTTP Basic으로 보낸다.

    KFP_HOST=https://<kfp 호스트> KFP_USERNAME=<사용자> KFP_PASSWORD=<app password> \
        .venv-kfp/bin/python tools/submit.py --package pipeline.yaml [--param sample_size=40000 ...] [--wait]
"""
import argparse
import base64
import datetime
import os
import sys

import kfp
from kfp.client import TokenCredentialsBase


class AuthentikBasic(TokenCredentialsBase):
    def __init__(self, user, password):
        self.value = base64.b64encode(f'{user}:{password}'.encode()).decode()

    def refresh_api_key_hook(self, config):
        config.api_key['authorization'] = self.value
        config.api_key_prefix['authorization'] = 'Basic'


def parse_param(text):
    # 숫자로 읽히는 값은 숫자로 넘긴다.
    name, _, value = text.partition('=')
    for cast in (int, float):
        try:
            return name, cast(value)
        except ValueError:
            pass
    return name, value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', default='pipeline.yaml')
    parser.add_argument('--pipeline', default='capstone-url-classifier')
    parser.add_argument('--experiment', default='capstone')
    parser.add_argument('--param', action='append', default=[], type=parse_param)
    parser.add_argument('--wait', action='store_true', help='run이 끝날 때까지 기다리고 상태를 출력한다')
    args = parser.parse_args()

    host = os.environ['KFP_HOST'].rstrip('/')
    client = kfp.Client(host=host, credentials=AuthentikBasic(os.environ['KFP_USERNAME'], os.environ['KFP_PASSWORD']))

    version = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    pipeline_id = client.get_pipeline_id(args.pipeline)
    if pipeline_id is None:
        p = client.upload_pipeline(args.package, pipeline_name=args.pipeline)
        pipeline_id = p.pipeline_id
        version_id = client.list_pipeline_versions(pipeline_id).pipeline_versions[0].pipeline_version_id
    else:
        version_id = client.upload_pipeline_version(args.package, pipeline_version_name=version,
                                                    pipeline_id=pipeline_id).pipeline_version_id

    experiment = client.create_experiment(args.experiment)  # 있으면 그 실험을 돌려준다
    run = client.run_pipeline(experiment_id=experiment.experiment_id, job_name=f'{args.pipeline}-{version}',
                              params=dict(args.param), pipeline_id=pipeline_id, version_id=version_id)
    print(f'run {run.run_id}')
    print(f'{host}/#/runs/details/{run.run_id}')

    if args.wait:
        done = client.wait_for_run_completion(run.run_id, timeout=3 * 60 * 60, sleep_duration=30)
        print(f'state {done.state}')
        sys.exit(0 if done.state == 'SUCCEEDED' else 1)

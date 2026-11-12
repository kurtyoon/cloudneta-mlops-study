"""KFP run이 MLflow에 남긴 것을 출력한다: 학습 run의 파라미터와 지표, 등록된 버전과 champion 별칭.

    MLFLOW_TRACKING_URI=... MLFLOW_TRACKING_USERNAME=... MLFLOW_TRACKING_PASSWORD=... \
        python tools/show_run.py [--kfp-run-id <id>]

--kfp-run-id를 주지 않으면 실험의 마지막 run을 보여 준다.
"""
import argparse

from mlflow import MlflowClient

from capstone_kfp import config

PARAMS = ['sample_size', 'epochs', 'lr', 'train_size', 'test_size', 'device', 'torch_threads']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kfp-run-id')
    args = parser.parse_args()

    client = MlflowClient()
    experiment = client.get_experiment_by_name(config.EXPERIMENT)
    query = f"tags.kfp_run_id = '{args.kfp_run_id}'" if args.kfp_run_id else ''
    run = client.search_runs([experiment.experiment_id], query, order_by=['start_time DESC'], max_results=1)[0]
    seconds = (run.info.end_time - run.info.start_time) / 1000
    print(f'run {run.info.run_name} ({run.info.status}, {seconds:.0f}s)')
    for k in PARAMS:
        print(f'  {k}: {run.data.params.get(k)}')
    for k in sorted(run.data.metrics):
        print(f'  {k}: {run.data.metrics[k]:.4f}')

    champion = client.get_model_version_by_alias(config.REGISTERED_MODEL, 'champion').version
    print(f'{config.REGISTERED_MODEL}: champion v{champion}')
    for v in client.search_model_versions(f"name = '{config.REGISTERED_MODEL}'"):
        print(f'  v{v.version} run={v.run_id[:8]} gate={v.tags.get("gate")}')

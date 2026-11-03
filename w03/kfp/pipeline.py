"""캡스톤 악성 URL 분류기를 Feast와 Kubeflow Pipelines로 잇는 파이프라인.

1. 피처 생성: S3의 원본 CSV → 표본 → 토큰 id → 오프라인 스토어와 라벨 parquet → feast apply
2. 학습: 라벨 parquet을 엔티티로 get_historical_features → 작은 BERT → MLflow run, 모델 등록
3. 승격: macro F1이 기준 이상이면 champion 별칭을 새 버전으로
4. materialize: 추론 요청으로 쓸 행만 Valkey에 올린다
5. 온라인 추론: Valkey에서 get_online_features → champion 모델 → 예측은 S3에, 지표는 KFP 출력으로

2와 4는 1 뒤에 나란히 돌고, 5는 3과 4가 모두 끝난 뒤에 돈다. 단계 코드는 이미지의 capstone_kfp에 있다.
kfp도 이미지에 들어 있어 Pod가 시작할 때 pip install을 하지 않는다.

컴파일:
    uv venv --python 3.11 .venv-kfp && uv pip install --python .venv-kfp/bin/python kfp==2.9.0 kfp-kubernetes==1.3.0
    .venv-kfp/bin/python pipeline.py --image <레지스트리>/<owner>/capstone-kfp:<커밋 sha> --out pipeline.yaml
"""
import argparse
from typing import NamedTuple

from kfp import compiler, dsl, kubernetes

from capstone_kfp import config

# 키는 Secret에서 환경 변수로만 받는다. KFP는 파이프라인 파라미터를 실행 이력에 평문으로 남긴다.
S3_SECRET = 'kfp-pipeline-s3'
S3_ENV = {'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY'}
VALKEY_SECRET = 'feast-valkey'
VALKEY_ENV = {'VALKEY_PASSWORD': 'VALKEY_PASSWORD'}


def build_features(raw_csv: str, sample_size: int, seed: int, max_length: int, n_online: int) -> NamedTuple(
        'Outputs', [('rows', int), ('online_rows', int), ('features_sha256', str), ('labels_sha256', str)]):
    from capstone_kfp import features
    r = features.run(raw_csv, sample_size, seed, max_length, n_online)
    return r['rows'], r['online_rows'], r['features_sha256'], r['labels_sha256']


def train_model(features_sha256: str, labels_sha256: str, kfp_run_id: str, sample_size: int, n_online: int,
          num_layers: int, hidden_size: int, num_heads: int, max_length: int, epochs: int, batch_size: int,
          lr: float, seed: int, tracking_uri: str, experiment: str, registered_model: str,
          metrics: dsl.Output[dsl.Metrics]) -> NamedTuple(
        'Outputs', [('model_version', str), ('run_id', str), ('accuracy', float), ('macro_f1', float)]):
    from types import SimpleNamespace

    from capstone_kfp import train
    args = SimpleNamespace(num_layers=num_layers, hidden_size=hidden_size, num_heads=num_heads,
                           max_length=max_length, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed,
                           sample_size=sample_size, n_online=n_online, tracking_uri=tracking_uri,
                           experiment=experiment, registered_model=registered_model)
    r = train.run(args, features_sha256, labels_sha256, kfp_run_id)
    metrics.log_metric('test_accuracy', r['accuracy'])
    metrics.log_metric('test_macro_f1', r['macro_f1'])
    for label, v in r['per_class'].items():
        metrics.log_metric(f'{label}_recall', v['recall'])
    return r['model_version'], r['run_id'], r['accuracy'], r['macro_f1']


def promote_model(model_version: str, macro_f1: float, threshold: float, tracking_uri: str,
            registered_model: str) -> NamedTuple('Outputs', [('promoted', bool), ('champion_version', str),
                                                            ('message', str)]):
    from capstone_kfp import promote
    r = promote.run(tracking_uri, registered_model, model_version, macro_f1, threshold)
    return r['promoted'], r['champion_version'], r['message']


def materialize_online(labels_sha256: str) -> NamedTuple('Outputs', [('keys', int), ('bytes', int), ('matched', int)]):
    # labels_sha256은 피처 단계 뒤에 돌게 하는 입력이다. 읽는 parquet은 config.LABELS_URI다.
    from capstone_kfp import materialize
    r = materialize.run()
    return r['keys'], r['bytes'], r['matched']


def infer_online(champion_version: str, online_keys: int, kfp_run_id: str, tracking_uri: str, registered_model: str,
          metrics: dsl.Output[dsl.Metrics], report: dsl.Output[dsl.Markdown]) -> NamedTuple(
        'Outputs', [('output', str), ('accuracy', float)]):
    # champion_version과 online_keys는 승격과 materialize 뒤에 돌게 하는 입력이다.
    # 모델은 별칭으로 다시 찾고, 찾은 버전을 출력에 적는다.
    from capstone_kfp import infer
    r = infer.run(tracking_uri, registered_model, kfp_run_id)
    metrics.log_metric('accuracy', r['accuracy'])
    metrics.log_metric('macro_f1', r['macro_f1'])
    for label, v in r['per_class'].items():
        metrics.log_metric(f'{label}_accuracy', v['recall'])
    with open(report.path, 'w') as f:
        f.write(r['markdown'])
    return r['output'], r['accuracy']


def make_pipeline(image):
    def component(func):
        return dsl.component(func, base_image=image, install_kfp_package=False)

    build_op, train_op, promote_op, materialize_op, infer_op = map(
        component, (build_features, train_model, promote_model, materialize_online, infer_online))

    @dsl.pipeline(name='capstone-url-classifier',
                  description='캡스톤 악성 URL 분류기: Feast 오프라인 스토어로 학습, 온라인 스토어로 추론')
    def capstone_pipeline(
            raw_csv: str = config.RAW_CSV,
            sample_size: int = 10000,
            seed: int = 42,
            max_length: int = 128,
            n_online: int = 200,
            num_layers: int = 4,
            hidden_size: int = 256,
            num_heads: int = 4,
            epochs: int = 2,
            batch_size: int = 16,
            # classifier.py 기본값 1e-5는 랜덤 초기화한 작은 모델에 너무 느리다.
            lr: float = 1e-4,
            macro_f1_threshold: float = 0.9,
            tracking_uri: str = config.MLFLOW_TRACKING_URI,
            experiment: str = config.EXPERIMENT,
            registered_model: str = config.REGISTERED_MODEL,
    ):
        feats = build_op(raw_csv=raw_csv, sample_size=sample_size, seed=seed, max_length=max_length,
                         n_online=n_online)
        trained = train_op(
            features_sha256=feats.outputs['features_sha256'], labels_sha256=feats.outputs['labels_sha256'],
            kfp_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER, sample_size=sample_size, n_online=n_online,
            num_layers=num_layers, hidden_size=hidden_size, num_heads=num_heads, max_length=max_length,
            epochs=epochs, batch_size=batch_size, lr=lr, seed=seed, tracking_uri=tracking_uri,
            experiment=experiment, registered_model=registered_model)
        promoted = promote_op(model_version=trained.outputs['model_version'],
                              macro_f1=trained.outputs['macro_f1'], threshold=macro_f1_threshold,
                              tracking_uri=tracking_uri, registered_model=registered_model)
        online = materialize_op(labels_sha256=feats.outputs['labels_sha256'])
        inferred = infer_op(champion_version=promoted.outputs['champion_version'],
                            online_keys=online.outputs['keys'], kfp_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
                            tracking_uri=tracking_uri, registered_model=registered_model)

        # Secret은 그 값을 쓰는 단계에만 준다. 승격은 MLflow만 부르므로 둘 다 받지 않는다.
        for task in (feats, trained, online, inferred):
            kubernetes.use_secret_as_env(task, secret_name=S3_SECRET, secret_key_to_env=S3_ENV)
        for task in (feats, online, inferred):
            kubernetes.use_secret_as_env(task, secret_name=VALKEY_SECRET, secret_key_to_env=VALKEY_ENV)

        # 상한 없이 2Gi보다 큰 요청을 주면 LimitRange의 기본 상한에 걸려 거부되므로 상한을 같이 준다.
        # torch는 노드의 코어 수만큼 스레드를 열므로 OMP_NUM_THREADS를 CPU 상한에 맞춘다.
        feats.set_cpu_request('250m').set_cpu_limit('1').set_memory_request('512Mi').set_memory_limit('1Gi')
        trained.set_cpu_request('1').set_cpu_limit('2').set_memory_request('2Gi').set_memory_limit('2Gi')
        trained.set_env_variable('OMP_NUM_THREADS', '2')
        promoted.set_cpu_request('100m').set_cpu_limit('500m').set_memory_request('256Mi').set_memory_limit('512Mi')
        online.set_cpu_request('100m').set_cpu_limit('1').set_memory_request('256Mi').set_memory_limit('1Gi')
        inferred.set_cpu_request('250m').set_cpu_limit('1').set_memory_request('768Mi').set_memory_limit('1536Mi')
        inferred.set_env_variable('OMP_NUM_THREADS', '1')

        # 모든 단계가 S3, 레지스트리, MLflow, Valkey의 상태를 바꾸므로 캐시하지 않는다.
        for task, name in ((feats, '1. 피처 생성'), (trained, '2. 학습'), (promoted, '3. 승격'),
                           (online, '4. materialize'), (inferred, '5. 온라인 추론')):
            task.set_caching_options(False)
            task.set_display_name(name)

    return capstone_pipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='컴포넌트 이미지 (레지스트리/owner/이름:태그)')
    parser.add_argument('--out', default='pipeline.yaml')
    args = parser.parse_args()
    compiler.Compiler().compile(make_pipeline(args.image), args.out)
    print(f'compiled with image {args.image}: {args.out}')

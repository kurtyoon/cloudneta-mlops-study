"""영화 추천 모델의 배포 파이프라인: MLflow staging → PVC models → KServe.

movie-train의 승격 단계가 recommender_production의 staging alias를 옮기면, 이
파이프라인이 그 버전을 서빙용 파일로 만들어 PVC models의 movie/v<레지스트리 버전>/에
쓴다. InferenceService movie-recommender는 STORAGE_URI pvc://models/movie/v<버전>으로
이 파일을 복사 없이 읽는다. 서빙할 버전을 바꾸는 것은 STORAGE_URI를 바꾸는 PR이다.

어떤 모델을 내보낼지는 이 파이프라인이 정하지 않는다. 학습 파이프라인의 승격 게이트를
통과해 staging alias를 받은 버전만 내보낸다.

단계:
- catalog (python:3.11): inference-datasets/<dataset>/에서 영화 목록과 세 분할의 평점을
  읽어 movies.json(영화 ID와 제목)과 seen.json(사용자별로 평점을 남긴 영화 ID)을 만든다.
  서버가 추천 후보와 이미 본 영화를 정할 때 쓴다.
- export (ultralytics 이미지의 torch, 단계 안에서 mlflow 설치): alias가 가리키는 버전을
  찾아 모델을 읽고 TorchScript로 내보낸 뒤, 원래 모델과 출력이 같은지 확인하고 게시한다.
  - 학습 단계는 모델 클래스를 단계 함수 안에서 정의하고 MLflow가 cloudpickle로 저장한다.
    이 파일을 읽으려면 mlflow나 cloudpickle이 있어야 한다. TorchScript는 torch만 있으면
    읽히므로 서빙 컨테이너에 패키지를 설치하지 않고, egress도 필요 없다.
  - 같은 버전을 두 번 게시하지 않는다(idcard-publish와 같은 규칙).

컴파일: python pipeline_publish.py 를 실행하면 pipeline_publish.yaml이 생긴다.
"""
import os
import sys

from kfp import compiler, dsl, kubernetes
from kfp.dsl import Artifact, Input, Output

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_train import MLFLOW_URI, S3_ENV, S3_SECRET, YOLO_IMAGE  # noqa: E402


@dsl.component(base_image='python:3.11', packages_to_install=['pyarrow', 'pandas'])
def build_catalog(bucket: str, dataset: str, catalog: Output[Artifact]):
    import json
    import os

    import pandas as pd
    from pyarrow import fs, parquet

    s3 = fs.S3FileSystem(
        endpoint_override='http://minio-service.kubeflow:9000',
        access_key=os.environ['AWS_ACCESS_KEY_ID'],
        secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        scheme='http',
    )

    def read(name):
        return parquet.read_table(s3.open_input_file(f'{bucket}/{dataset}/{name}.parquet.gzip')).to_pandas()

    movies = read('movies')
    ratings = pd.concat([read(split) for split in ['train', 'val', 'test']], ignore_index=True)
    seen = ratings.groupby('userId')['movieId'].apply(lambda ids: sorted(int(i) for i in ids))

    os.makedirs(catalog.path, exist_ok=True)
    with open(os.path.join(catalog.path, 'movies.json'), 'w') as file:
        json.dump({str(int(row.movieId)): row.title for row in movies.itertuples()}, file)
    with open(os.path.join(catalog.path, 'seen.json'), 'w') as file:
        json.dump({str(int(user)): ids for user, ids in seen.items()}, file)
    print(f'movies {len(movies)} users {len(seen)} ratings {len(ratings)}')


@dsl.component(base_image=YOLO_IMAGE, packages_to_install=['kfp==2.17.0', '--no-deps'])
def export_model(registered_model_name: str, alias: str, mlflow_uri: str, publish_dir: str,
                 catalog: Input[Artifact]) -> str:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-warn-script-location',
                           'mlflow==3.17.2'])

    import datetime
    import json
    import os
    import shutil

    import mlflow
    import mlflow.pytorch
    import torch
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(uri=mlflow_uri)
    client = MlflowClient()
    version = client.get_model_version_by_alias(registered_model_name, alias)
    target = os.path.join('/models', publish_dir, f'v{version.version}')
    if os.path.exists(os.path.join(target, 'model.pt')):
        raise ValueError(f'{target} already holds a model; {registered_model_name}@{alias} is v{version.version}, '
                         'which is published already')

    model = mlflow.pytorch.load_model(f'models:/{registered_model_name}@{alias}')
    model.eval()
    n_users = model.user_factors.num_embeddings - 1
    n_items = model.item_factors.num_embeddings - 1

    # Embedding lookups and the MLP have no data-dependent branches, so tracing records the whole forward.
    example = (torch.tensor([0, 1]), torch.tensor([0, 1]))
    with torch.no_grad():
        scripted = torch.jit.trace(model, example)
        generator = torch.Generator().manual_seed(0)
        users = torch.randint(0, n_users, (4096,), generator=generator)
        items = torch.randint(0, n_items, (4096,), generator=generator)
        max_diff = float((model(users, items) - scripted(users, items)).abs().max())
    if max_diff > 1e-5:
        raise ValueError(f'TorchScript output differs from the model by {max_diff}')

    os.makedirs(target, exist_ok=True)
    scripted.save(os.path.join(target, 'model.pt'))
    for name in ['movies.json', 'seen.json']:
        shutil.copy2(os.path.join(catalog.path, name), os.path.join(target, name))
    run = client.get_run(version.run_id)
    meta = {
        'registered_model_name': registered_model_name,
        'alias': alias,
        'model_version': f'v{version.version}',
        'run_id': version.run_id,
        'source': version.source,
        'metrics': run.data.metrics,
        'promotion': run.data.tags.get('promotion'),
        'n_users': n_users,
        'n_items': n_items,
        'torchscript_max_abs_diff': max_diff,
        'exported_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    }
    with open(os.path.join(target, 'meta.json'), 'w') as file:
        json.dump(meta, file, indent=2)
    print(json.dumps(meta))
    return meta['model_version']


@dsl.pipeline(name='movie-publish', description='MLflow staging 영화 추천 모델을 TorchScript로 내보내 PVC models에 게시한다')
def publish_pipeline(
        bucket: str = 'inference-datasets',
        dataset: str = 'ml-latest-small',
        registered_model_name: str = 'recommender_production',
        alias: str = 'staging',
        mlflow_uri: str = MLFLOW_URI,
        publish_dir: str = 'movie'):
    catalog_op = build_catalog(bucket=bucket, dataset=dataset).set_display_name('catalog')
    kubernetes.use_secret_as_env(catalog_op, secret_name=S3_SECRET, secret_key_to_env=S3_ENV)

    export_op = export_model(
        registered_model_name=registered_model_name,
        alias=alias,
        mlflow_uri=mlflow_uri,
        publish_dir=publish_dir,
        catalog=catalog_op.outputs['catalog'],
    ).set_display_name('export')
    export_op.set_cpu_request('500m').set_cpu_limit('1').set_memory_request('1Gi').set_memory_limit('2Gi')
    export_op.set_env_variable('OMP_NUM_THREADS', '1')
    kubernetes.mount_pvc(export_op, pvc_name='models', mount_path='/models')

    for op in [catalog_op, export_op]:
        op.set_caching_options(False)


if __name__ == '__main__':
    compiler.Compiler().compile(pipeline_func=publish_pipeline, package_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline_publish.yaml'))

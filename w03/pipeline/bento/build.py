"""MLflow 레지스트리의 champion을 BentoML 모델로 저장한다.

토크나이저와 max_length는 학습 run에서 읽어 모델과 같이 넣는다. 모델 폴더에는 pickle 대신
save_pretrained 결과(safetensors, config.json)를 넣어, 서빙 쪽이 임의 객체를 풀지 않게 한다.
champion run의 드리프트 reference(drift/reference.parquet)도 같은 폴더에 넣는다. run에 없으면
멈춘다. kfp/tools/backfill_reference.py로 넣은 뒤 다시 돌린다.
"""
import argparse
import os
import shutil
import sys
import tempfile

import bentoml
import mlflow.pytorch
from mlflow import MlflowClient
from transformers import BertTokenizer

MODEL_NAME = 'capstone-url-classifier'
ALIAS = 'champion'
# kfp/capstone_kfp/config.py의 값. run에 tokenizer_revision이 없을 때 쓴다.
TOKENIZER_REVISION = 'cd5ef92a9fb2f889e972770a36d4ed042daf221e'
REFERENCE_ARTIFACT = 'drift/reference.parquet'  # kfp/capstone_kfp/config.py의 REFERENCE_ARTIFACT
REFERENCE_DATASET = 'url-stats-train'  # kfp/capstone_kfp/config.py의 REFERENCE_DATASET


def save(name, model, tokenizer, reference, labels, metadata):
    with bentoml.models.create(name, labels=labels, metadata=metadata) as bento_model:
        model.save_pretrained(bento_model.path_of('model'))
        tokenizer.save_pretrained(bento_model.path_of('tokenizer'))
        dst = bento_model.path_of(REFERENCE_ARTIFACT)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(reference, dst)
    return bento_model.tag


def download_reference(client, run_id, dst_dir):
    """champion run의 drift/reference.parquet를 받아 로컬 경로를 돌려준다.

    학습 단계와 백필(kfp/capstone_kfp/drift.py의 log_reference)은 그 파일을 올린 뒤 데이터셋 입력
    url-stats-train을 남긴다. KFP 드리프트 단계는 데이터셋 입력으로 reference를 찾으므로, 파일만 있고
    입력이 없는 run(남기다 멈춘 run)도 받지 않는다. 그래야 서빙과 KFP가 같은 reference를 쓴다."""
    folder, _ = os.path.split(REFERENCE_ARTIFACT)
    inputs = client.get_run(run_id).inputs
    missing = []
    if REFERENCE_ARTIFACT not in [f.path for f in client.list_artifacts(run_id, folder)]:
        missing.append(f'아티팩트 {REFERENCE_ARTIFACT}')
    if not any(d.dataset.name == REFERENCE_DATASET for d in (inputs.dataset_inputs if inputs else [])):
        missing.append(f'데이터셋 입력 {REFERENCE_DATASET}')
    if missing:
        sys.exit(f'run {run_id}에 다음이 없다: {", ".join(missing)}. 3주차 이전에 학습한 run이면 '
                 f'kfp/tools/backfill_reference.py로 넣은 뒤 다시 돌린다')
    return client.download_artifacts(run_id, REFERENCE_ARTIFACT, dst_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--model-name', default=MODEL_NAME)
    parser.add_argument('--alias', default=ALIAS)
    args = parser.parse_args()

    client = MlflowClient()
    version = client.get_model_version_by_alias(args.model_name, args.alias)
    params = client.get_run(version.run_id).data.params
    max_length = int(params.get('max_length', 128))
    tokenizer_revision = params.get('tokenizer_revision', TOKENIZER_REVISION)

    with tempfile.TemporaryDirectory() as tmp:
        # reference가 없으면 모델을 받기 전에 멈춘다.
        reference = download_reference(client, version.run_id, tmp)
        # 2주차 v1은 mps에서 pickle됐다
        model = mlflow.pytorch.load_model(f'models:/{args.model_name}@{args.alias}', map_location='cpu')
        model.eval()
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', revision=tokenizer_revision)

        tag = save(
            args.model_name,
            model,
            tokenizer,
            reference,
            labels={'mlflow_alias': args.alias},
            metadata={
                'mlflow_model_version': version.version,
                'mlflow_run_id': version.run_id,
                'max_length': max_length,
                'tokenizer_revision': tokenizer_revision,
            },
        )
    print(f'{tag} <- models:/{args.model_name}/{version.version} (max_length {max_length}, {REFERENCE_ARTIFACT})')


if __name__ == '__main__':
    main()

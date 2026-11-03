"""MLflow 레지스트리의 champion을 BentoML 모델로 저장한다.

토크나이저와 max_length는 학습 run에서 읽어 모델과 같이 넣는다.
"""
import argparse

import bentoml
import mlflow.pytorch
from mlflow import MlflowClient
from transformers import BertTokenizer

MODEL_NAME = 'capstone-url-classifier'
ALIAS = 'champion'
# run에 tokenizer_revision이 없을 때 쓰는 값. kfp/capstone_kfp/config.py와 같다.
TOKENIZER_REVISION = 'cd5ef92a9fb2f889e972770a36d4ed042daf221e'


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

    model = mlflow.pytorch.load_model(f'models:/{args.model_name}@{args.alias}', map_location='cpu')
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', revision=tokenizer_revision)

    saved = bentoml.pytorch.save_model(
        args.model_name,
        model,
        signatures={'__call__': {'batchable': False}},
        custom_objects={'tokenizer': tokenizer},
        labels={'mlflow_alias': args.alias},
        metadata={
            'mlflow_model_version': version.version,
            'mlflow_run_id': version.run_id,
            'max_length': max_length,
            'tokenizer_revision': tokenizer_revision,
        },
    )
    print(f'{saved.tag} <- models:/{args.model_name}/{version.version} (max_length {max_length})')


if __name__ == '__main__':
    main()

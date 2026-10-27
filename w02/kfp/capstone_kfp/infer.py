"""5단계: 온라인 추론. 요청 URL의 피처를 온라인 스토어에서 읽고 champion 모델로 분류한다.

요청은 피처 단계가 고른 online 행(학습에 쓰지 않은 test URL)이다. 실시간 요청에는 URL만 온다고 보고,
엔티티 키 url_id를 URL에서 다시 계산해 Valkey를 찾는다. 모델은 별칭으로 찾는다
(models:/capstone-url-classifier@champion). 승격 단계에서 새 버전이 기준을 넘지 못했으면
기존 champion(post-2의 v1)이 예측한다.
"""
import io
import json

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
from mlflow import MlflowClient

from capstone_kfp import common, config
from capstone_kfp.features import url_id
from capstone_kfp.train import classification_metrics, limit_threads, model_inputs

FEATURES = [f'{config.FEATURE_VIEW}:token_ids']


def champion_model(tracking_uri, model_name):
    """champion 버전, 그 버전을 학습한 run의 max_length, 모델."""
    client = MlflowClient(tracking_uri)
    version = client.get_model_version_by_alias(model_name, config.CHAMPION)
    params = client.get_run(version.run_id).data.params if version.run_id else {}
    mlflow.set_tracking_uri(tracking_uri)
    # map_location: post-2의 v1은 노트북의 mps에서 학습한 채로 피클돼 있다. CPU Pod에서 열려면 필요하다.
    model = mlflow.pytorch.load_model(f'models:/{model_name}@{config.CHAMPION}', map_location='cpu')
    model.eval()
    return version.version, int(params.get('max_length', 128)), model


def predict(model, token_ids, max_length, batch_size=64):
    inputs = model_inputs(token_ids, max_length)
    preds = []
    with torch.no_grad():
        for i in range(0, len(token_ids), batch_size):
            batch = {k: v[i:i + batch_size] for k, v in inputs.items()}
            preds.extend(torch.argmax(model(**batch).logits, dim=-1).tolist())
    return preds


def report_markdown(version, metrics, n):
    rows = [f'| {label} | {v["support"]} | {v["recall"]:.4f} |' for label, v in metrics['per_class'].items()]
    return '\n'.join([
        f'## 온라인 추론: champion v{version}, 요청 {n}건',
        '',
        f'accuracy {metrics["accuracy"]:.4f}, macro F1 {metrics["macro_f1"]:.4f}',
        '',
        '| 클래스 | 건수 | 정확도 |',
        '|---|---|---|',
        *rows,
    ])


@common.hides_valkey_password
def run(tracking_uri, model_name, kfp_run_id):
    limit_threads()
    labels = common.read_parquet(config.LABELS_URI)
    requests = labels[labels['online']].sort_values('url_id').reset_index(drop=True)
    request_ids = requests['url'].map(url_id)
    if not (request_ids == requests['url_id']).all():
        raise RuntimeError('URL에서 다시 만든 url_id가 피처 단계의 값과 다르다')

    store = common.feature_store()
    online = store.get_online_features(
        features=FEATURES, entity_rows=[{'url_id': i} for i in request_ids]).to_dict()
    missing = [i for i, t in zip(online['url_id'], online['token_ids']) if t is None]
    if missing:
        raise RuntimeError(f'온라인 스토어에 없는 url_id {len(missing)}개 (materialize 단계 확인): {missing[:5]}')

    version, max_length, model = champion_model(tracking_uri, model_name)
    preds = predict(model, online['token_ids'], max_length)
    metrics = classification_metrics(requests['label'].tolist(), preds)

    out = f'{config.INFERENCE_PREFIX}/{kfp_run_id}'
    predictions = pd.DataFrame({
        'url_id': requests['url_id'], 'url': requests['url'], 'label': requests['label'],
        'prediction': preds, 'prediction_label': [config.LABELS[p] for p in preds],
        'model_version': str(version),
    })
    buf = io.BytesIO()
    predictions.to_parquet(buf, index=False)
    common.write_bytes(f'{out}/predictions.parquet', buf.getvalue())
    summary = {'model': model_name, 'champion_version': str(version), 'max_length': max_length,
               'requests': len(requests), **{k: metrics[k] for k in ('accuracy', 'macro_f1', 'per_class',
                                                                     'confusion_matrix')}}
    common.write_bytes(f'{out}/metrics.json', json.dumps(summary, ensure_ascii=False, indent=2).encode(),
                       content_type='application/json')

    result = {'output': out, 'champion_version': str(version), 'requests': len(requests),
              'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1'],
              'per_class': metrics['per_class'],
              'markdown': report_markdown(version, metrics, len(requests))}
    print({k: result[k] for k in ('output', 'champion_version', 'requests', 'accuracy', 'macro_f1')})
    return result

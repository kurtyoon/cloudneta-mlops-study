"""3단계: 승격. 새 버전의 macro F1이 기준을 넘으면 별칭 champion을 새 버전으로 옮긴다.

post-2에서 등록한 capstone-url-classifier 버전 1(run lr-1e-4, macro F1 0.9432, 표본 4만 건,
3 epoch)이 이미 champion이다. 기준을 넘지 못하면 별칭은 그대로 두고, 추론 단계는 기존
champion으로 예측한다. 어느 쪽이든 새 버전에 gate 태그를 남겨 MLflow UI에서 결과가 보이게 한다.
"""
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from capstone_kfp import config


def gate(macro_f1, threshold):
    # 기준과 같으면 통과로 본다.
    return macro_f1 >= threshold


def current_champion(client, model_name):
    try:
        return client.get_model_version_by_alias(model_name, config.CHAMPION).version
    except MlflowException:
        return None


def run(tracking_uri, model_name, model_version, macro_f1, threshold):
    client = MlflowClient(tracking_uri)
    previous = current_champion(client, model_name)
    passed = gate(macro_f1, threshold)
    client.set_model_version_tag(model_name, model_version, 'gate',
                                 f'{"passed" if passed else "failed"}: macro_f1 {macro_f1:.4f}, threshold {threshold}')
    if passed:
        client.set_registered_model_alias(model_name, config.CHAMPION, model_version)
        champion = str(model_version)
        message = (f'macro F1 {macro_f1:.4f} >= {threshold}: champion을 옮겼다(v{previous} → v{champion})'
                   if previous else
                   f'macro F1 {macro_f1:.4f} >= {threshold}: champion을 v{champion}에 붙였다')
    else:
        champion = previous
        message = (f'macro F1 {macro_f1:.4f} < {threshold}: 승격하지 않았다(새 버전 v{model_version}). '
                   f'champion은 v{previous} 그대로다' if previous else
                   f'macro F1 {macro_f1:.4f} < {threshold}: 승격하지 않았다(새 버전 v{model_version}). '
                   f'champion이 없다')
    print(message)
    if champion is None:
        raise RuntimeError(f'{model_name}에 champion이 없어 추론할 모델이 없다')
    return {'promoted': passed, 'champion_version': str(champion), 'previous_champion': previous or '',
            'message': message}

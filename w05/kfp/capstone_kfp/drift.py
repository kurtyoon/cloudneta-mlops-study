"""5단계: 드리프트 탐지(책 6.2.2의 detect_drift).

reference는 champion run의 데이터셋 입력 url-stats-train, current는 이번 run의 online 행 URL이다.
판정은 Evidently에 올리기 전에 정한다. 올리지 못해도 같은 판정을 돌려줘, 추론이 Evidently에 묶이지 않는다.
"""
import os
import tempfile
from pathlib import PurePosixPath
from urllib.parse import urlparse

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import DatasetInput, InputTag
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT

from capstone_kfp import common, config, url_drift


def artifact_url(tracking_uri, artifact_uri, path):
    """mlflow-artifacts:/ URI → MLflow 서버의 아티팩트 프록시 주소(HTTP).
    mlflow.data는 mlflow-artifacts:/와 runs:/를 데이터셋 출처로 받지 않는다."""
    parsed = urlparse(artifact_uri)
    if parsed.scheme != 'mlflow-artifacts':
        raise RuntimeError(f'MLflow 서버가 프록시하는 아티팩트가 아니다: {artifact_uri}')
    return f'{tracking_uri.rstrip("/")}/api/2.0/mlflow-artifacts/artifacts/{parsed.path.strip("/")}/{path}'


def log_reference(client, run_id, reference, tracking_uri=config.MLFLOW_TRACKING_URI):
    """reference를 run의 아티팩트로 올리고 그 파일을 출처로 한 데이터셋 입력 url-stats-train을 남긴다.
    출처는 드리프트 단계가 읽는 클러스터 안 주소다. 끝난 run(백필)의 끝 시각이 바뀌지 않게 MlflowClient로만 쓴다."""
    url = artifact_url(tracking_uri, client.get_run(run_id).info.artifact_uri, config.REFERENCE_ARTIFACT)
    artifact = PurePosixPath(config.REFERENCE_ARTIFACT)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, artifact.name)
        reference.to_parquet(path, index=False)
        client.log_artifact(run_id, path, artifact_path=str(artifact.parent))
    dataset = mlflow.data.from_pandas(reference, source=url, name=config.REFERENCE_DATASET)
    # MlflowClient.log_inputs는 entity를 받는다. mlflow.log_input도 안에서 같은 변환을 한다.
    client.log_inputs(run_id, [DatasetInput(dataset._to_mlflow_entity(),
                                            tags=[InputTag(MLFLOW_DATASET_CONTEXT, 'training')])])
    return url


def reference_input(client, run_id):
    """run의 데이터셋 입력 중 url-stats-train(여러 개면 마지막). 없으면 None."""
    inputs = client.get_run(run_id).inputs
    found = [d for d in (inputs.dataset_inputs if inputs else []) if d.dataset.name == config.REFERENCE_DATASET]
    return found[-1] if found else None


def champion_reference(client, model_name):
    """champion 버전과 그 버전을 학습한 run의 reference DataFrame."""
    version = client.get_model_version_by_alias(model_name, config.CHAMPION)
    found = reference_input(client, version.run_id) if version.run_id else None
    if found is None:
        raise RuntimeError(
            f'{model_name} champion v{version.version}(run {version.run_id})에 데이터셋 입력 '
            f'{config.REFERENCE_DATASET}이 없다. tools/backfill_reference.py로 넣은 뒤 다시 돌린다')
    with tempfile.TemporaryDirectory() as d:
        reference = pd.read_parquet(mlflow.data.get_source(found).load(dst_path=d))
    return version.version, reference


def report_markdown(version, summary, n_reference, drift_share, upload_error=None):
    rows = [f'| {c} | {v["p_value"]:.4g} | {"예" if v["drifted"] else "아니오"} |'
            for c, v in summary['columns'].items()]
    verdict = '드리프트, 온라인 추론을 건너뛴다' if summary['drift_detected'] else '드리프트 없음, 온라인 추론을 돌린다'
    lines = [
        f'## 드리프트: champion v{version}의 학습 URL {n_reference}건과 요청 {summary["rows"]}건',
        '',
        f'칸마다 KS 검정을 하고 p-value가 {url_drift.THRESHOLD}보다 작으면 드리프트로 센다. '
        f'드리프트 칸 {summary["drifted_columns"]}/{len(summary["columns"])}(비율 {summary["drifted_share"]:.3f}), '
        f'기준 {drift_share} → {verdict}',
        '',
        '| 칸 | KS p-value | 드리프트 |',
        '|---|---|---|',
        *rows,
    ]
    if upload_error:
        lines += ['', f'Evidently에 보고서를 올리지 못했다({upload_error}). 판정은 위와 같고, '
                      f'Evidently 대시보드에는 이 run의 보고서가 없다.']
    return '\n'.join(lines)


def upload(evidently_url, project_name, snapshot, kfp_run_id):
    """보고서를 Evidently 프로젝트에 run으로 올린다. 실패하면 오류 문장을, 성공하면 None을 돌려준다."""
    try:
        workspace = url_drift.remote_workspace(evidently_url, os.environ['EVIDENTLY_SECRET'])
        project = url_drift.evidently_project(workspace, project_name)
        workspace.add_run(project.id, snapshot, name=f'kfp-{kfp_run_id[:8]}')
    except Exception as e:  # 서버가 없거나, 느리거나, 쓰기를 거부해도 판정은 돌려준다
        error = f'{type(e).__name__}: {e}'[:300]
        print(f'Evidently({evidently_url})에 보고서를 올리지 못했다: {error}')
        return error
    return None


def run(tracking_uri, model_name, evidently_url, project_name, drift_share, kfp_run_id, champion_version):
    """별칭으로 다시 찾은 champion이 승격 단계가 넘긴 버전과 다르면 멈춘다. reference가 섞이지 않게 한다."""
    client = MlflowClient(tracking_uri)
    version, reference = champion_reference(client, model_name)
    if str(version) != str(champion_version):
        raise RuntimeError(f'{model_name} champion 별칭이 v{version}을 가리킨다. 승격 단계가 넘긴 '
                           f'v{champion_version}과 다르다. 이 run이 도는 사이에 별칭이 바뀌었다')
    labels = common.read_parquet(config.LABELS_URI)
    current = url_drift.url_stats(labels.loc[labels['online'], 'url'])
    snapshot, summary = url_drift.drift_report(
        reference, current, drift_share, tags=['kfp'],
        metadata={'source': 'kfp', 'kfp_run_id': kfp_run_id, 'champion_version': str(version)})

    upload_error = upload(evidently_url, project_name, snapshot, kfp_run_id)
    result = {'champion_version': str(version), 'reference_rows': len(reference), **summary,
              'uploaded': upload_error is None,
              'markdown': report_markdown(version, summary, len(reference), drift_share, upload_error)}
    print({k: result[k] for k in ('champion_version', 'reference_rows', 'rows', 'drifted_columns', 'drifted_share',
                                  'drift_detected', 'uploaded')})
    return result

"""csi-s3 측정 파이프라인 뒤에 모델 게시 단계를 붙인 버전.

학습한 best.pt를 PVC `models`의 idcard/<model_version>/ 아래에 쓴다. PVC
`models`는 버킷 csi-volumes의 prefix models를 가리키는 static PV에 묶여 있고,
serving 네임스페이스의 PVC `models`도 같은 prefix를 가리킨다. 그래서 KServe의
InferenceService가 storageUri pvc://models/idcard/<model_version>으로 이 파일을
복사 없이 읽는다.

같은 버전을 두 번 게시하지 않는다. 서빙 중인 버전의 파일이 바뀌면 어느 모델을
서빙하는지 git으로 알 수 없기 때문이다.

검증 게이트: idcard/pipeline.py의 gate_report로 val mAP50을 min_map50과 비교하고, 통과할 때만
publish 단계를 실행한다(dsl.If). 미달이면 publish를 건너뛰고 run은 성공으로 끝난다.
판정은 gate-report의 report 아티팩트에 남는다. run을 실패시키지 않는 이유는 GitOps로
돌리는 run Job이 실패하면 kfp-pipelines Application이 Degraded가 되어 다음 sync를
막기 때문이다.

컴파일: python pipeline_publish.py 를 실행하면 pipeline_publish.yaml이 생긴다.
"""
import os
import sys

from kfp import compiler, dsl, kubernetes
from kfp.dsl import Input, Metrics, Model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_storage import download_dataset, split_dataset, train_model, validate_model  # noqa: E402
from pipeline import gate_report  # noqa: E402  (idcard/pipeline.py, on the path pipeline_storage set)


@dsl.component(base_image='python:3.11')
def publish_model(model: Input[Model], metrics: Input[Metrics], model_version: str):
    import datetime
    import json
    import os
    import shutil

    target = os.path.join('/models', 'idcard', model_version)
    if os.path.exists(os.path.join(target, 'best.pt')):
        raise ValueError(f'{target} already holds a model; publish a new model_version instead')
    os.makedirs(target, exist_ok=True)
    shutil.copy2(os.path.join(model.path, 'best.pt'), os.path.join(target, 'best.pt'))
    meta = {
        'model_version': model_version,
        'source': model.uri,
        'metrics': dict(metrics.metadata),
        'published_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    }
    with open(os.path.join(target, 'meta.json'), 'w') as file:
        json.dump(meta, file, indent=2)
    print(json.dumps(meta))


@dsl.pipeline(name='idcard-publish', description='신분증 탐지 YOLO를 버킷 PVC로 학습하고, 모델을 PVC models에 게시한다')
def publish_pipeline(
    model_version: str,
    epochs: int = 1,
    batch: int = 8,
    random_state: int = 42,
    yolo_model_name: str = 'yolov8n_custom',
    min_map50: float = 0.5,
):
    pvc = kubernetes.CreatePVC(
        pvc_name_suffix='-idcard',
        access_modes=['ReadWriteMany'],
        size='3Gi',
        storage_class_name='csi-s3',
    ).set_display_name('create-pvc')

    download_op = download_dataset().set_display_name('download').set_retry(num_retries=2).after(pvc)
    download_op.set_memory_request('512Mi').set_memory_limit('2Gi')

    split_op = split_dataset(random_state=random_state).set_display_name('split').after(download_op)

    train_op = train_model(epochs=epochs, batch=batch, yolo_model_name=yolo_model_name).set_display_name('train').after(split_op)
    train_op.set_cpu_request('1').set_cpu_limit('2').set_memory_request('2Gi').set_memory_limit('4Gi')
    train_op.set_env_variable('OMP_NUM_THREADS', '2')
    train_op.set_env_variable('YOLO_CONFIG_DIR', '/tmp/ultralytics')

    validate_op = validate_model(
        data_yaml=train_op.outputs['data_yaml'],
        model=train_op.outputs['model_output'],
    ).set_display_name('validate').after(train_op)
    validate_op.set_cpu_request('500m').set_cpu_limit('1').set_memory_request('1Gi').set_memory_limit('3Gi')
    validate_op.set_env_variable('OMP_NUM_THREADS', '1')
    validate_op.set_env_variable('YOLO_CONFIG_DIR', '/tmp/ultralytics')

    for op in [download_op, split_op, train_op, validate_op]:
        kubernetes.mount_pvc(op, pvc_name=pvc.outputs['name'], mount_path='/data')

    report_op = gate_report(metrics=validate_op.outputs['metrics'], min_map50=min_map50).set_display_name('gate-report')

    with dsl.If(report_op.outputs['Output'] == True, name='gate-passed'):  # noqa: E712
        publish_op = publish_model(
            model=train_op.outputs['model_output'],
            metrics=validate_op.outputs['metrics'],
            model_version=model_version,
        ).set_display_name('publish')
        kubernetes.mount_pvc(publish_op, pvc_name='models', mount_path='/models')

    delete_pvc = kubernetes.DeletePVC(pvc_name=pvc.outputs['name']).set_display_name('delete-pvc').after(validate_op)

    for op in [pvc, download_op, split_op, train_op, validate_op, report_op, publish_op, delete_pvc]:
        op.set_caching_options(False)


if __name__ == '__main__':
    compiler.Compiler().compile(pipeline_func=publish_pipeline, package_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline_publish.yaml'))

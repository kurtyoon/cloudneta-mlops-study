# 5주차: 버킷 PVC, 모델 게시와 KServe 서빙, git으로 KFP run 실행

`../w04/`를 복사해 고친 상태입니다. `analysis/`, `kfp/`, `pipeline/`, `runpod/`, `serving/capstone-url-classifier.yaml`은 4주차와 같습니다. 이번 주에 고친 코드는 캡스톤이 아니라 책의 실습 모델 두 개(신분증 탐지, 영화 추천)입니다. 모델 설명과 원본 진단은 [baseline/README.md](../baseline/README.md)에 있습니다.

## 글

- [버킷을 PVC로 마운트하면 무엇이 달라질까](https://blog.somnitor.dev/posts/102/bucket-pvc-with-csi-s3): `idcard/`, `streaming/`, `manifests/csi-s3/`
- [MLflow에 승격한 모델은 어떻게 KServe까지 갈까](https://blog.somnitor.dev/posts/103/mlflow-model-to-kserve): `movie/`, `idcard/pipeline_publish.py`, `serving/`, `manifests/kfp-pipelines/`

## 4주차에서 바꾼 파일

- `idcard/`: 새 폴더
  - `pipeline.py`: 신분증 탐지 학습, 검증 파이프라인. 4주차에 돌린 코드이고, 아래 파일들이 import합니다.
  - `pipeline_storage.py`: 데이터를 local-path PVC나 csi-s3 버킷 PVC에 두는 버전과, split 안에서 fsync 시간을 재는 버전
  - `pipeline_publish.py`: val mAP50 게이트를 통과한 모델만 PVC `models`에 게시
- `movie/`: 새 폴더
  - `pipeline_data_prep.py`: ml-latest-small 데이터 준비. 4주차에 돌린 코드입니다.
  - `pipeline_train.py`: 학습, 검증, MLflow 기록, staging 승격 게이트
  - `pipeline_publish.py`: staging 모델을 TorchScript로 바꿔 PVC `models`에 게시
- `streaming/pipeline_streamer.py`: safetensors 모델을 로컬 디스크, 버킷 PVC, S3에서 읽는 시간 비교
- `serving/`: InferenceService 두 개와 KServe V1 서버 (`server.py`, `idcard-detector.yaml`, `movie_server.py`, `movie-recommender.yaml`, `models-pvc.yaml`, `kustomization.yaml`)
- `manifests/csi-s3/`: 버킷을 GeeseFS로 마운트하는 csi-s3 드라이버(arm64 이미지), StorageClass, prefix `models`를 kubeflow와 serving에 묶는 정적 PV
- `manifests/kfp-pipelines/`: 파이프라인과 run을 Argo CD로 배포하는 파일. `render.py`는 컴파일한 YAML을 Pipeline과 PipelineVersion으로 바꾸고, `runs/`의 Job은 `trigger/trigger.py`로 run을 하나씩 실행합니다.

## 실행

필요한 Secret은 `csi-s3/csi-s3-secret`(키 `accessKeyID`, `secretAccessKey`, `endpoint`, `region`)과 `kubeflow/kfp-pipeline-s3`(키 `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)입니다. `endpoint`는 노드에서 닿는 S3 주소입니다.

컴파일하고 `manifests/kfp-pipelines/pipelines/`에 PipelineVersion을 만듭니다.

```bash
python3 -m venv .venv
.venv/bin/pip install kfp==2.9.0 kfp-kubernetes==1.3.0 pyyaml
(cd idcard && for f in pipeline.py pipeline_storage.py pipeline_publish.py; do ../.venv/bin/python $f; done)
(cd movie && for f in pipeline_train.py pipeline_publish.py; do ../.venv/bin/python $f; done)
(cd streaming && ../.venv/bin/python pipeline_streamer.py)

cd manifests/kfp-pipelines && mkdir -p pipelines
render() { ../../.venv/bin/python render.py "$@"; }
render ../../idcard/pipeline.yaml idcard-artifact --no-cache
render ../../idcard/pipeline_local-path.yaml idcard-local-path
render ../../idcard/pipeline_csi-s3.yaml idcard-csi-s3
render ../../idcard/pipeline_csi-s3-fsync.yaml idcard-csi-s3-fsync
render ../../idcard/pipeline_publish.yaml idcard-publish --no-cache
render ../../streaming/pipeline_streamer.yaml model-streamer --no-cache
render ../../movie/pipeline_train.yaml movie-train
render ../../movie/pipeline_publish.yaml movie-publish --no-cache
```

이렇게 만든 PipelineVersion 이름은 `runs/`의 Job이 가리키는 버전과 같습니다. `idcard-csi-s3-fsync-1`과 `idcard-publish-1`만 이전 버전을 가리켜서, 이 코드로는 그 버전을 다시 만들 수 없습니다. `manifests/csi-s3/`, `manifests/kfp-pipelines/`, `serving/`을 각각 kustomize로 배포합니다.

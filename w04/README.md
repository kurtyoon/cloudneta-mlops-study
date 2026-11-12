# 4주차: 학습과 검증, RunPod GPU로 학습하기

`../w03/`을 복사해 고친 상태입니다. `analysis/`, `pipeline/`, `serving/`은 3주차와 같습니다. 모델 설명과 원본 진단은 [baseline/README.md](../baseline/README.md)에 있습니다.

## 글

- [KFP에서 객체 탐지 모델 학습하고 검증하기](https://blog.somnitor.dev/posts/95/object-detection-training-on-kfp): 코드 변경 없음
- [KFP의 Artifact와 PVC는 무엇이 다를까](https://blog.somnitor.dev/posts/96/kfp-artifact-vs-pvc): `kfp/` 변경
- [seaborn.objects로 그래프 그리기](https://blog.somnitor.dev/posts/97/seaborn-objects-plot-grammar): 코드 변경 없음
- [pandas로 데이터 변환하기](https://blog.somnitor.dev/posts/98/pandas-data-transformation): 코드 변경 없음
- [EDA에서는 무엇을 확인할까](https://blog.somnitor.dev/posts/99/exploratory-data-analysis-flow): 코드 변경 없음
- [잔차를 보면서 모델 만들기](https://blog.somnitor.dev/posts/100/statistical-modeling-with-residuals): 코드 변경 없음
- [GPU 없는 클러스터의 KFP에서 RunPod GPU로 학습하기](https://blog.somnitor.dev/posts/101/runpod-gpu-training-from-kfp): `runpod/` 추가

## 3주차에서 바꾼 파일

- `kfp/pipeline.py`: 온라인 추론 단계가 혼동 행렬을 KFP 출력으로 남깁니다. `--infer-only`로 컴파일하면 현재 champion으로 온라인 추론만 도는 파이프라인이 됩니다.
- `kfp/tests_compile/test_pipeline.py`: 추론 전용 파이프라인의 컴파일 테스트
- `runpod/`: 새 폴더. KFP 단계 하나가 RunPod GPU Pod를 만들어 신분증 탐지 YOLO 학습을 맡기고, 결과를 Model 아티팩트로 받은 뒤 Pod를 지웁니다.
  - `pipeline.py`: `runpod-train`과 `report` 두 단계. 키는 Secret `runpod-api`에서 환경 변수로 받습니다.
  - `yolo_train.py`: GPU Pod 안에서 데이터 다운로드, 분할, 학습을 한 번에 실행

## 실행

```bash
cd runpod
python3 -m venv .venv
.venv/bin/pip install kfp==2.9.0 kfp-kubernetes==1.3.0
.venv/bin/python pipeline.py
```

만들어진 `pipeline.yaml`을 KFP에 올립니다. 준비와 실행 순서는 마지막 글에 있습니다.

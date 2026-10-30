# 2주차: 학습을 MLflow에 기록하기, Feast와 Kubeflow Pipelines

`../w01/`을 복사해 고친 상태입니다. `pipeline/`은 1주차와 같습니다. 모델 설명과 원본 진단은 [baseline/README.md](../baseline/README.md)에 있습니다.

## 글

- [블로그가 도는 k3s에 4, 5장 실습 환경 만들기](https://blog.somnitor.dev/posts/89/k3s-lab-for-mlflow-feast-kfp): 코드 변경 없음
- [accuracy 0.845인데 malware는 하나도 못 맞혔다](https://blog.somnitor.dev/posts/90/capstone-training-with-mlflow): `analysis/` 변경
- [학습은 offline store에서, 추론은 online store에서](https://blog.somnitor.dev/posts/91/capstone-feast-kfp-pipeline): `kfp/` 추가

## 1주차에서 바꾼 파일

- `analysis/preprocessing/preprocessing.py`: 인자로 경로와 표본 크기를 받음, URL 데이터만 쓸 수 있게 특징 데이터를 선택으로, 클래스 비율을 지킨 표본과 분할, 토크나이저 리비전 고정
- `analysis/classifier/classifier.py`: 학습이 돌게 고침, 작은 BERT 설정과 `max_length`를 인자로, 디바이스 자동 선택, 평가 지표 계산, MLflow에 파라미터, 지표, 모델 기록과 레지스트리 등록
- `analysis/classifier/requirements.txt`: mlflow 추가
- `analysis/classifier/compare_runs.py`: 새 파일. MLflow run들을 파라미터와 지표로 나란히 출력
- `analysis/eda/`: 새 폴더. seaborn objects로 그린 URL 길이 구간별 클래스 구성 그림
- `kfp/`: 새 폴더. 다섯 step으로 된 KFP pipeline입니다. 레지스트리 주소는 컴파일할 때 인자로, KFP 주소는 제출할 때 환경 변수로 받습니다.
  - step: 피처 생성과 `feast apply`, offline store에서 학습과 MLflow 등록, 승격, Valkey online store로 materialize, online store에서 추론
  - `Dockerfile`, `requirements.lock`, `requirements-kfp.lock`: 컴포넌트 이미지와 해시로 고정한 lock
  - `.gitea/workflows/`: 이미지를 빌드하는 Gitea Actions workflow
  - `tools/submit.py`: 제출 스크립트
  - `tests/`: 단위 테스트

전체 diff는 [커밋 6ad5c2b](https://github.com/kurtyoon/cloudneta-mlops-study/commit/6ad5c2b)에서 볼 수 있습니다. 이 커밋은 폴더를 나누기 전이라 경로가 `capstone/`으로 보입니다.

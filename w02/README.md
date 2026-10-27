# 2주차: 학습을 MLflow에 기록하기, Feast와 Kubeflow Pipelines

`../w01/`을 복사해 고친 상태입니다. `pipeline/`은 1주차와 같습니다. 모델 설명과 원본 진단은 [baseline/README.md](../baseline/README.md)에 있습니다.

## 글

글을 게시하면 링크를 넣습니다.

- 실습 인프라: 코드 변경 없음
- 캡스톤 학습을 MLflow에 기록하기: `analysis/` 변경
- Feast와 Kubeflow Pipelines: `kfp/` 추가

## 1주차에서 바꾼 파일

- `analysis/preprocessing/preprocessing.py`: 인자로 경로와 표본 크기를 받음, URL 데이터만 쓸 수 있게 특징 데이터를 선택으로, 클래스 비율을 지킨 표본과 분할, 토크나이저 리비전 고정
- `analysis/classifier/classifier.py`: 학습이 돌게 고침, 작은 BERT 설정과 `max_length`를 인자로, 디바이스 자동 선택, 평가 지표 계산, MLflow에 파라미터, 지표, 모델 기록과 레지스트리 등록
- `analysis/classifier/requirements.txt`: mlflow 추가
- `analysis/classifier/compare_runs.py` (새 파일): MLflow run들을 파라미터와 지표로 나란히 출력
- `analysis/eda/` (새 폴더): URL 길이 구간별 클래스 구성 그림(seaborn objects)
- `kfp/` (새 폴더): KFP pipeline 다섯 step(피처 생성과 `feast apply`, offline store에서 학습과 MLflow 등록, 승격, Valkey online store로 materialize, online store에서 추론). 컴포넌트 이미지(`Dockerfile`, 해시로 고정한 lock), 이미지를 빌드하는 Gitea Actions workflow(`kfp/.gitea/workflows/`), 제출 스크립트(`tools/submit.py`), 단위 테스트(`tests/`). 레지스트리와 KFP 주소는 컴파일과 제출 때 인자, 환경 변수로 받는다

전체 diff는 [커밋 6ad5c2b](https://github.com/kurtyoon/cloudneta-mlops-study/commit/6ad5c2b)에서 볼 수 있습니다. 이 커밋은 폴더를 나누기 전이라 경로가 `capstone/`으로 보입니다.

# baseline — 악성 URL 분류 모델 코드 (원본)

[CSID-DGU/2024-1-CECD1-EJO-2](https://github.com/CSID-DGU/2024-1-CECD1-EJO-2)의 `main` `09b1ebe`(2024-06-30)에서 모델과 관련된 코드만 그대로 복사했습니다. 경로 구조는 원본과 같고, 코드는 한 줄도 고치지 않았습니다. 고친 버전은 이 폴더 밖에 따로 만듭니다.

이 저장소에는 모델과 관련된 코드만 올립니다. 원본의 Spring 백엔드 코드는 올리지 않았습니다.

## 무엇을 하는 모델인가

URL 문자열을 받아 Benign / Phishing / Defacement / Malware 중 하나로 분류합니다. BERT-base 구조의 `BertForSequenceClassification`(4클래스)이고, URL을 `bert-base-cased` WordPiece 토크나이저로 그대로 잘라 입력합니다. 사전학습 가중치는 쓰지 않고 랜덤 초기화부터 학습합니다(`classifier.py:42`). 사전학습 어휘를 쓰는 것은 토크나이저뿐입니다.

## 구성

| 경로 | 단계 | 내용 |
|---|---|---|
| `analysis/preprocessing/preprocessing.py` | 전처리 | URL 데이터 토큰화. 웹사이트 특징 데이터는 SMOTE로 균형을 맞추고, RandomForest 중요도로 특징을 고른 뒤 `/`로 이은 문자열을 토큰화 |
| `analysis/preprocessing/util.py` | 전처리 | 라벨 매핑 benign 0, phishing 1, defacement 2, malware 3 |
| `analysis/classifier/classifier.py` | 학습·평가 | 512 패딩, 8:2 stratified split, AdamW lr 1e-5, batch 16, 10 epoch, `mps`. state_dict를 `model/model.pth`로 저장 |
| `pipeline/serve.py` + `Dockerfile` | 서빙 A | FastAPI `POST /api/predict` → `{"prediction": "Benign"}`. CPU 추론, 모델 파일을 이미지에 넣음 |
| `pipeline/build.py`, `service.py`, `bentofile.yaml`, `deploy.sh` | 서빙 B | BentoML 1.2 Runner API. `classify` → `{"prediction": 0~3}` |

데이터셋 두 개는 원본 저장소에 없습니다(gitignore). 둘 다 모델 구조를 가져온 논문(Su & Su, "BERT-Based Approaches to Identifying Malicious URLs", Sensors 23(20) 8499, 2023)의 참고문헌 [28][29]과 같습니다(2026-10-19 확인).

- `malicious_phish.csv` (`url`, `type`): Kaggle [Malicious URLs dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset), Manu Siddhartha, CC0. 651,191개
- `dataset_full.csv` (특징 111개 + `phishing`): Vrbančič 외, [Datasets for Phishing Websites Detection](https://github.com/GregaVrbancic/Phishing-Dataset), Data in Brief 33 (2020). 88,647개

학습된 모델 파일(`model.pth`, `pipeline/model/config.json`, `model.safetensors`)도 저장소에 없습니다.

## 원본 그대로는 돌지 않는 부분

- 컬럼명이 다릅니다. 전처리는 특징 데이터를 `token_idx`로 저장하고(`preprocessing.py:63`), 학습은 `token_ids`로 읽습니다(`classifier.py:51`).
- 학습 결과와 서빙 입력의 형식이 다릅니다. 학습은 `model.pth`(state_dict)를 남기는데, 서빙은 `from_pretrained("./model")`로 HF 디렉터리를 읽습니다. 중간 변환 코드가 없습니다.
- 512 토큰을 넘는 입력을 자르지 않습니다(`classifier.py:59-60`, `serve.py:16-22`). 긴 URL이 들어오면 학습 배치 구성이나 추론에서 실패합니다.
- BentoML 경로는 정수를 돌려주는데, 백엔드는 라벨 문자열을 `EType.valueOf`로 바꿉니다(`UrlService.java:60`). 또 `to_runner()`를 두 번 불러(`service.py:8`, `:29`) 서비스에 등록한 Runner와 실제로 호출하는 Runner가 서로 다른 객체입니다.

## 돌아가더라도 품질에 영향을 주는 부분

- SMOTE를 train/test 분할 전에 적용합니다(`preprocessing.py:37-38`). 데이터 누수입니다.
- 패딩까지 attention mask를 1로 둡니다(`classifier.py:67-68`). 서빙은 패딩 없이 넣으니 학습과 서빙의 입력 분포가 다릅니다.
- 특징 문자열 데이터(`token_type_ids`=1, `classifier.py:71`)로 학습하지만, 서빙은 URL만 받습니다. 이 특징 데이터로 학습한 부분은 추론 입력과 이어지지 않습니다.
- 평가 함수가 예측만 모으고 지표를 계산하지 않습니다(`classifier.py:116-126`). 성능 수치가 남아 있지 않습니다.
- 모델 라벨 번호(0=Benign…3=Malware)와 백엔드 `EType` 코드(0=PHISHING…3=BENIGN, `EType.java:10-13`)의 순서가 다릅니다. 지금은 이름으로 매핑해서 문제가 없습니다.
- 디바이스가 `mps`로 고정되어 있고(`classifier.py:36`), torch 시드를 고정하지 않으며, 실험 기록은 `print`뿐입니다.

## 스터디에서 다룰 만한 주제

이 코드는 "모델은 있는데 MLOps는 없는" 상태라, 단계마다 바꿔 볼 거리가 분명합니다.

| 주제 | 이 코드에서 출발점 |
|---|---|
| 데이터 버전 관리 | CSV 두 개가 저장소 밖에 있고 출처 기록이 없음 → DVC 등으로 버전 고정 |
| 파이프라인 오케스트레이션 | 전처리 → 학습 → 평가 → 저장이 수동 스크립트 → Airflow / Kubeflow Pipelines / Argo로 DAG화 |
| 데이터 검증·누수 방지 | SMOTE 분할 전 적용, 컬럼명 불일치 → 분할 후 리샘플링, 스키마 검증 |
| 실험 추적 | 지표·하이퍼파라미터 기록 없음 → MLflow로 metric, param, artifact 기록 |
| 모델 레지스트리·포맷 | `.pth`와 HF 포맷 사이 변환 누락 → `save_pretrained` + 레지스트리 등록, 버전 태깅 |
| 서빙 비교 | FastAPI와 BentoML이 공존 → 둘의 계약(응답 형식)을 통일하고 KServe / Triton과 비교 |
| 학습·서빙 일관성 | 토큰화·패딩·마스크·truncation이 학습과 서빙에 따로 구현됨 → 공용 전처리 모듈로 묶기 |
| 컨테이너·배포 | 모델을 이미지에 넣는 Dockerfile → 레지스트리에서 받아오는 구조, K8s 배포, 롤백 |
| 모니터링 | 요청·지연·예측 분포 기록 없음 → Prometheus 지표, 예측 분포 드리프트 감시 |
| 피드백 루프 | 사용자 신고(`report` 테이블)가 재학습에 쓰이지 않음 → 신고 데이터로 재학습 트리거 |

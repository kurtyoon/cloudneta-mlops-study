# baseline: 악성 URL 분류 모델 원본 코드

[CSID-DGU/2024-1-CECD1-EJO-2](https://github.com/CSID-DGU/2024-1-CECD1-EJO-2)의 `main` 브랜치, 2024-06-30 커밋 `09b1ebe`에서 모델과 관련된 코드만 그대로 복사했습니다. 경로 구조는 원본과 같고, 코드는 한 줄도 고치지 않았습니다. 고친 코드는 주차 폴더에 있습니다.

원본의 Spring 백엔드 코드는 올리지 않았습니다.

## 무엇을 하는 모델인가

URL 문자열을 받아 Benign, Phishing, Defacement, Malware 중 하나로 분류합니다. BERT-base 구조의 4클래스 `BertForSequenceClassification`이고, URL을 `bert-base-cased` WordPiece 토크나이저로 그대로 잘라 입력합니다. 사전학습 가중치는 쓰지 않고, `classifier.py:42`에서 랜덤 초기화한 모델부터 학습합니다. 사전학습 어휘를 쓰는 것은 토크나이저뿐입니다.

## 구성

- `analysis/preprocessing/preprocessing.py`: 전처리. URL 데이터를 토큰화합니다. 웹사이트 특징 데이터는 SMOTE로 균형을 맞추고, RandomForest 중요도로 특징을 고른 뒤 `/`로 이은 문자열을 토큰화합니다.
- `analysis/preprocessing/util.py`: 라벨 매핑 benign 0, phishing 1, defacement 2, malware 3
- `analysis/classifier/classifier.py`: 학습과 평가. 512 패딩, 8:2 stratified split, AdamW lr 1e-5, batch 16, 10 epoch, `mps`. state_dict를 `model/model.pth`로 저장합니다.
- `pipeline/serve.py`, `Dockerfile`: 서빙 A. FastAPI `POST /api/predict` → `{"prediction": "Benign"}`. CPU로 추론하고 모델 파일을 이미지에 넣습니다.
- `pipeline/build.py`, `service.py`, `bentofile.yaml`, `deploy.sh`: 서빙 B. BentoML 1.2 Runner API. `classify` → `{"prediction": 0~3}`

데이터셋 두 개는 원본 저장소에서 gitignore로 빠져 있습니다. 둘 다 모델 구조를 가져온 논문 Su & Su, "BERT-Based Approaches to Identifying Malicious URLs", Sensors 23(20) 8499, 2023의 참고문헌 [28], [29]와 같습니다.

- `malicious_phish.csv`: 컬럼 `url`, `type`. Kaggle [Malicious URLs dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset), Manu Siddhartha, CC0, 651,191개
- `dataset_full.csv`: 특징 111개와 `phishing`. Vrbančič 외, [Datasets for Phishing Websites Detection](https://github.com/GregaVrbancic/Phishing-Dataset), Data in Brief 33, 2020, 88,647개

학습된 모델 파일 `model.pth`, `pipeline/model/config.json`, `model.safetensors`도 저장소에 없습니다.

## 원본 그대로는 돌지 않는 부분

- 컬럼명이 다릅니다. 전처리 `preprocessing.py:63`은 특징 데이터를 `token_idx`로 저장하고, 학습 `classifier.py:51`은 `token_ids`로 읽습니다.
- 학습 결과와 서빙 입력의 형식이 다릅니다. 학습은 state_dict인 `model.pth`를 남기는데, 서빙은 `from_pretrained("./model")`로 HF 디렉터리를 읽습니다. 중간 변환 코드가 없습니다.
- `classifier.py:59-60`과 `serve.py:16-22`는 512 토큰을 넘는 입력을 자르지 않습니다. 긴 URL이 들어오면 학습 배치 구성이나 추론에서 실패합니다.
- BentoML 경로는 정수를 돌려주는데, 백엔드 `UrlService.java:60`은 라벨 문자열을 `EType.valueOf`로 바꿉니다. 또 `service.py:8`과 `:29`에서 `to_runner()`를 두 번 불러, 서비스에 등록한 Runner와 실제로 호출하는 Runner가 서로 다른 객체입니다.

## 돌아가더라도 품질에 영향을 주는 부분

- `preprocessing.py:37-38`에서 SMOTE를 train/test 분할 전에 적용합니다. 데이터 누수입니다.
- `classifier.py:67-68`은 패딩 자리까지 attention mask를 1로 둡니다. 서빙은 패딩 없이 넣으니 학습과 서빙의 입력 분포가 다릅니다.
- `classifier.py:71`은 특징 문자열 데이터를 `token_type_ids`=1로 넣어 함께 학습하지만, 서빙은 URL만 받습니다. 이 특징 데이터로 학습한 부분은 추론 입력과 이어지지 않습니다.
- `classifier.py:116-126`의 평가 함수는 예측만 모으고 지표를 계산하지 않습니다. 성능 수치가 남아 있지 않습니다.
- 모델 라벨 번호는 0=Benign부터 3=Malware까지, 백엔드 `EType.java:10-13`의 코드는 0=PHISHING부터 3=BENIGN까지로 순서가 다릅니다. 지금은 이름으로 매핑해서 문제가 없습니다.
- `classifier.py:36`에서 디바이스가 `mps`로 고정되어 있고, torch 시드를 고정하지 않으며, 실험 기록은 `print`뿐입니다.

## 그 밖에 없는 것

- 전처리, 학습, 평가, 저장을 스크립트로 하나씩 손으로 돌립니다.
- 토큰화, 패딩, 마스크, 자르기가 학습과 서빙에 따로 구현되어 있습니다.
- 서빙에 요청, 지연, 예측 분포 기록이 없습니다.
- 사용자 신고가 쌓이는 백엔드 `report` 테이블은 재학습에 쓰이지 않습니다.

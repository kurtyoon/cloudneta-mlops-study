# 3주차: 캡스톤 Bento를 Yatai로 배포하기

`../w02/`를 복사해 고친 상태입니다. `analysis/`와 `kfp/`는 2주차와 같습니다. 모델 설명과 원본 진단은 [baseline/README.md](../baseline/README.md)에 있습니다.

## 글

- [amd64 이미지뿐인 Yatai 1.x를 arm64 k3s에 올리기](https://blog.somnitor.dev/posts/92/yatai-on-arm64-k3s): 코드 변경 없음
- [2024년에 멈춘 캡스톤 Bento, MLflow champion에서 Yatai 배포까지](https://blog.somnitor.dev/posts/93/capstone-bento-on-yatai): `pipeline/bento/` 추가

## 2주차에서 바꾼 파일

- `pipeline/bentofile.yaml`, `build.py`, `deploy.sh`, `service.py`: 2024년의 BentoML 파일입니다. `pipeline/bento/`로 옮기며 다시 썼습니다.
- `pipeline/bento/`: 새 폴더. BentoML 1.1.10으로 고정한 Bento입니다. Yatai 1.x가 BentoML 1.0, 1.1만 지원해서입니다.
  - `build.py`: MLflow의 `capstone-url-classifier@champion`을 받아 토크나이저 revision, `max_length`와 함께 BentoML 모델로 저장
  - `service.py`: Runner 하나, FastAPI 서버와 같은 계약(`POST /api/predict` → `{"prediction": "Benign"}`), 입력은 KFP 학습 step처럼 `max_length`에서 자르고 0으로 채움
  - `requirements.lock`: 이미지에 들어가는 패키지, 2024-06-30 기준 해시 포함. torch와 transformers는 학습 이미지와 같은 버전
  - `build-requirements.lock`: `build.py`와 `bentoml build`를 돌리는 노트북 환경

## 실행

```bash
cd pipeline/bento
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --require-hashes -r build-requirements.lock

export MLFLOW_TRACKING_URI=<MLflow 주소> MLFLOW_TRACKING_USERNAME=<사용자> MLFLOW_TRACKING_PASSWORD=<비밀번호>
.venv/bin/python build.py
.venv/bin/bentoml build
.venv/bin/bentoml serve capstone_url_classifier:latest --port 3000
curl -X POST localhost:3000/api/predict -H 'content-type: application/json' -d '{"url": "google.com"}'
```

lock은 `requirements.in`, `build-requirements.in`을 고친 뒤 각 lock 파일 머리말의 명령으로 다시 만듭니다.

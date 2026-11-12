# 3주차: 캡스톤 Bento를 KServe로 배포하고 드리프트를 Evidently로 보기

`../w02/`를 복사해 고친 상태입니다. `analysis/`는 2주차와 같습니다. 모델 설명과 원본 진단은 [baseline/README.md](../baseline/README.md)에 있습니다.

Yatai를 한 번 써 본 뒤 KServe로 옮겼습니다. 이 폴더는 KServe로 옮긴 뒤의 상태이고, Yatai 때의 코드는 [`553573f`의 `w03/`](https://github.com/kurtyoon/cloudneta-mlops-study/tree/553573f3e70f118f765b7858a5b74d42a1c20acd/w03), 드리프트를 넣기 전 KServe 상태는 [`5aba67e`의 `w03/`](https://github.com/kurtyoon/cloudneta-mlops-study/tree/5aba67e/w03)에 있습니다.

## 글

- [amd64 이미지뿐인 Yatai 1.x를 arm64 k3s에 올리기](https://blog.somnitor.dev/posts/92/yatai-on-arm64-k3s): 코드 변경 없음
- [2024년에 멈춘 캡스톤 Bento, MLflow champion에서 Yatai 배포까지](https://blog.somnitor.dev/posts/93/capstone-bento-on-yatai): `pipeline/bento/` 추가, Yatai 때의 코드
- [KServe 서빙과 Evidently 드리프트 탐지](https://blog.somnitor.dev/posts/94/kserve-evidently-drift): KServe로 옮긴 `pipeline/bento/`와 `serving/`, `kfp/`의 드리프트 단계, Bento의 드리프트 감시

## 2주차에서 바꾼 파일

- `pipeline/bentofile.yaml`, `build.py`, `deploy.sh`, `service.py`: 2024년의 BentoML 파일입니다. `pipeline/bento/`로 옮기며 다시 썼습니다.
- `pipeline/bento/`: 새 폴더. BentoML 1.4.40으로 만든 Bento입니다.
  - `build.py`: MLflow의 `capstone-url-classifier@champion`을 받아 토크나이저 revision, `max_length`와 함께 BentoML 모델로 저장. 모델 폴더는 `save_pretrained` 결과
  - `service.py`: `@bentoml.service` 클래스 하나, FastAPI 서버와 같은 계약(`POST /api/predict` → `{"prediction": "Benign"}`), 입력은 KFP 학습 step처럼 `max_length`에서 자르고 0으로 채움
  - `requirements.lock`: 이미지에 들어가는 패키지, 2026-11-03 기준 해시 포함. torch와 transformers는 학습 이미지와 같은 버전
  - `build-requirements.lock`: `build.py`, `bentoml build`, `bentoml serve`를 돌리는 환경
  - `.gitea/workflows/image.yaml`, `ci-requirements.lock`: Gitea 러너가 champion으로 이미지를 빌드해 레지스트리에 올림
  - `drift_monitor.py`, `url_drift.py`: 요청 URL 100건마다 드리프트 보고서를 Evidently에 넣는 감시. `url_drift.py`는 `kfp/capstone_kfp/url_drift.py`의 복사본
- `serving/capstone-url-classifier.yaml`: 새 파일. 그 이미지를 다이제스트로 가리키는 KServe InferenceService. `EVIDENTLY_URL`, `EVIDENTLY_SECRET`이 있으면 드리프트 감시가 켜짐
- `kfp/`: 드리프트 탐지 단계를 넣었습니다.
  - `capstone_kfp/url_drift.py`: URL에서 센 숫자 8칸(`url_stats`), 칸마다 KS 검정(p-value 0.05 미만이 드리프트)으로 만드는 Evidently 0.7 보고서와 대시보드
  - `capstone_kfp/drift.py`: champion run의 reference와 요청 URL을 비교해 `drift_detected`를 출력. 학습 단계는 학습 URL의 `url_stats`를 run에 reference로 남김
  - `pipeline.py`: `dsl.If`로 드리프트가 없을 때만 온라인 추론
  - `tools/backfill_reference.py`: 3주차 전에 학습한 champion run에 reference를 넣음

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

.venv/bin/bentoml containerize capstone_url_classifier:latest -t capstone-bento:local --opt platform=linux/arm64
```

lock은 `requirements.in`, `build-requirements.in`을 고친 뒤 각 lock 파일 머리말의 명령으로 다시 만듭니다. `requirements.in`을 바꿨을 때는 `--upgrade`를 붙입니다. 붙이지 않으면 uv가 기존 lock의 버전을 그대로 둡니다.

# 6주차: 서빙 모니터링과 드리프트 알림

`../w05/`를 복사해 고친 상태입니다. `analysis/`, `idcard/`, `kfp/`, `manifests/`, `movie/`, `runpod/`, `streaming/`은 5주차와 같습니다. 이번 주에는 세 predictor에 지표를 붙이고, 캡스톤의 드리프트 판정을 Prometheus 알림으로 Discord까지 보냈습니다. 모델 설명과 원본 진단은 [baseline/README.md](../baseline/README.md)에 있습니다.

## 글

- [KServe predictor 모니터링과 드리프트 알림](https://blog.somnitor.dev/posts/104/kserve-predictor-monitoring): `pipeline/bento/`, `serving/`, `monitoring/`

6주차 2편(팀 문서 검색 RAG의 설계와 운영)은 실습 없이 쓴 글이라 코드가 없습니다.

## 5주차에서 바꾼 파일

- `pipeline/bento/`
  - `serving_metrics.py`: 새 파일. 예측 라벨, 예측 확률, 드리프트 window 판정, 버린 window, Evidently 업로드 결과를 지표로 만듭니다. BentoML 워커 안에서 만들어야 `/metrics`에 나옵니다.
  - `service.py`: 예측마다 라벨과 softmax 확률을 지표에 남깁니다.
  - `drift_monitor.py`: window 판정을 Evidently에 넣기 전에 지표에 남깁니다. Evidently가 내려가도 드리프트 알림은 나갑니다.
  - `.gitea/workflows/image.yaml`: 이미지 스모크 테스트가 `/metrics`의 서빙 지표와 판정 수를 확인합니다.
  - `bentofile.yaml`, `tests/`: `serving_metrics.py`를 Bento에 넣고 테스트를 더했습니다.
- `serving/`
  - `promtext.py`: 새 파일. ultralytics 이미지에 `prometheus_client`가 없어서, 표준 라이브러리로 Counter, Gauge, Histogram을 텍스트 형식으로 내보냅니다.
  - `server.py`, `movie_server.py`: `/metrics`를 붙이고 요청 수, 지연, 모델 출력을 셉니다. ConfigMap 이름을 `-2`로 바꿨습니다.
  - `podmonitor.yaml`: 새 파일. Prometheus가 세 predictor의 `/metrics`를 읽습니다.
  - `capstone-url-classifier.yaml`: 지표를 내는 이미지로 바꿨습니다.
- `monitoring/`: 새 폴더
  - `prometheusrule-serving.yaml`: 책의 규칙 두 개(PredictorDown, PredictorMissing)에 준비 상태, 5xx 비율, 지연, 입력 드리프트, 드리프트 보고서 실패 규칙 다섯 개를 더했습니다.
  - `alertmanager-values.yaml`: kube-prometheus-stack values 중 Alertmanager의 Discord 경로
  - `dashboards/build.py`, `dashboards/mlops-serving.json`: 서빙 대시보드. 패널을 고치면 `python3 build.py`로 JSON을 다시 씁니다.
  - `send_urls.py`: 캡스톤 분류기에 정상 구간과 드리프트 구간의 요청을 차례로 보내는 실험 스크립트

## 실행

kube-prometheus-stack이 `monitoring` 네임스페이스에 있고, 다른 네임스페이스의 PodMonitor와 PrometheusRule을 고르도록 설정되어 있어야 합니다. `serving`에 NetworkPolicy가 있으면 Prometheus Pod에서 3000, 8080 포트로 오는 요청을 열어야 합니다.

필요한 Secret은 `monitoring/alertmanager-discord`(키 `webhook-url`, Discord webhook URL)입니다. `alertmanager-values.yaml`을 kube-prometheus-stack values에 합치고, `serving/`과 `monitoring/`을 각각 kustomize로 배포합니다.

캡스톤 이미지는 `pipeline/bento/.gitea/workflows/image.yaml`로 빌드합니다. 드리프트 실험은 `send_urls.py` 머리말의 명령으로 돌립니다. 학습 데이터 `malicious_phish.csv`는 이 저장소에 없어서 `--data`로 위치를 줍니다.

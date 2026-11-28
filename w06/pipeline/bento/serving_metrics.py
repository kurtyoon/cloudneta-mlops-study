"""캡스톤 서빙 지표. Prometheus가 BentoML의 /metrics에서 BentoML 기본 지표와 같이 읽는다.

BentoML 기본 지표(bentoml_service_request_total, bentoml_service_request_duration_seconds 등)는 요청 수와
지연까지다. 여기서는 그 위에 모델이 무엇을 내놓았는지(예측 라벨과 그 확률)와 드리프트 window의 판정을 더한다.
드리프트 판정은 Evidently에 보고서로 남지만 Evidently에는 알림을 보낼 길이 없다. 판정을 Prometheus에도
내놓아야 알림 규칙(PrometheusRule)이 걸 수 있다.

prometheus_client는 Metrics를 만들 때 불러온다. BentoML 1.4는 워커가 PROMETHEUS_MULTIPROC_DIR을 정한 뒤
서비스 클래스를 만든다. 그 전에(모듈을 import할 때) 불러오면 prometheus_client가 단일 프로세스 모드로
초기화되어 지표가 /metrics에 나오지 않는다(bentoml/metrics.py의 설명). 그래서 서비스 __init__에서 만든다.
"""

# 예측한 라벨의 softmax 확률 구간. 4클래스라 가장 큰 확률은 0.25 이상이다.
CONFIDENCE_BUCKETS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)

# 판정을 마친 window의 결과. error는 보고서(판정)를 만들지 못한 window다.
WINDOW_RESULTS = ('drifted', 'not_drifted', 'error')


class Metrics:
    """registry를 주지 않으면 prometheus_client의 기본 registry(BentoML /metrics가 읽는 것)에 등록한다.
    테스트는 CollectorRegistry를 따로 준다. 같은 registry에 두 번 만들면 이름이 겹쳐 실패한다."""

    def __init__(self, labels, registry=None):
        import prometheus_client as prom

        kw = {} if registry is None else {'registry': registry}
        self.predictions = prom.Counter('capstone_predictions', '예측 수(예측 라벨별)', ['prediction'], **kw)
        self.confidence = prom.Histogram('capstone_prediction_confidence', '예측한 라벨의 softmax 확률',
                                         ['prediction'], buckets=CONFIDENCE_BUCKETS, **kw)
        self.windows = prom.Counter('capstone_drift_windows', '드리프트 판정을 마친 window 수(판정별)', ['result'],
                                    **kw)
        self.dropped = prom.Counter('capstone_drift_windows_dropped', '보고서 줄이 차서 판정 없이 버린 window 수',
                                    **kw)
        self.uploads = prom.Counter('capstone_drift_uploads', 'Evidently에 보고서를 넣은 결과(ok, failed)',
                                    ['result'], **kw)
        # Gauge는 마지막 window의 값이다. 워커가 하나라 mostrecent로 둔다(멀티프로세스 모드의 기본 all은 워커
        # pid마다 따로 내놓는다).
        self.share = prom.Gauge('capstone_drift_share', '마지막 window에서 드리프트가 난 칸의 비율',
                                multiprocess_mode='mostrecent', **kw)
        self.detected = prom.Gauge('capstone_drift_detected', '마지막 window의 데이터셋 드리프트 판정(1 드리프트, 0 아님)',
                                   multiprocess_mode='mostrecent', **kw)
        self.last_window = prom.Gauge('capstone_drift_last_window_timestamp_seconds',
                                      '마지막으로 판정한 window의 시각(Unix 초)', multiprocess_mode='mostrecent', **kw)
        self.p_value = prom.Gauge('capstone_drift_p_value', '마지막 window의 칸별 KS 검정 p-value', ['column'],
                                  multiprocess_mode='mostrecent', **kw)
        # 라벨 조합을 미리 만든다. 한 번도 나오지 않은 라벨도 0으로 있어야 rate()와 비율 패널이 비지 않고,
        # 알림 식의 increase()가 첫 증가를 놓치지 않는다.
        for label in labels:
            self.predictions.labels(label)
            self.confidence.labels(label)
        for result in WINDOW_RESULTS:
            self.windows.labels(result)
        for result in ('ok', 'failed'):
            self.uploads.labels(result)
        # p_value는 미리 만들지 않는다. 0으로 만들어 두면 판정 전부터 모든 칸이 드리프트로 보인다.

    def prediction(self, label, confidence):
        self.predictions.labels(label).inc()
        self.confidence.labels(label).observe(confidence)

    def drift_judged(self, summary):
        """url_drift.drift_report의 summary. None이면 보고서를 만들지 못한 window다."""
        if summary is None:
            self.windows.labels('error').inc()
            return
        self.windows.labels('drifted' if summary['drift_detected'] else 'not_drifted').inc()
        self.share.set(summary['drifted_share'])
        self.detected.set(1 if summary['drift_detected'] else 0)
        for column, result in summary['columns'].items():
            self.p_value.labels(column).set(result['p_value'])
        self.last_window.set_to_current_time()

    def drift_dropped(self):
        self.dropped.inc()

    def drift_uploaded(self, ok):
        self.uploads.labels('ok' if ok else 'failed').inc()

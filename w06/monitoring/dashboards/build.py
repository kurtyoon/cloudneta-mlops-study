"""Writes mlops-serving.json, the Grafana dashboard of the three serving predictors.

Run after changing a panel: python3 build.py (standard library only). The
Grafana sidecar loads the JSON from the ConfigMap that kustomization.yaml
generates with the label grafana_dashboard: "1".

Series come from PodMonitor serving/predictors. The capstone Bento reports
requests as bentoml_service_request_*, the idcard and movie servers as
predictor_request*; the request panels read both with `or`.
"""
import json
from pathlib import Path

DS = {'type': 'prometheus', 'uid': 'prometheus'}
JOB = 'job="serving/predictors"'
RATE = '$__rate_interval'


def target(expr, legend, ref):
    return {'datasource': DS, 'expr': expr, 'legendFormat': legend, 'refId': ref}


def panel(kind, title, x, y, w, h, targets, unit=None, description=None, instant=False, **extra):
    p = {'type': kind, 'title': title, 'datasource': DS, 'gridPos': {'x': x, 'y': y, 'w': w, 'h': h},
         'targets': [target(e, l, chr(65 + i)) for i, (e, l) in enumerate(targets)],
         'fieldConfig': {'defaults': {}, 'overrides': []}, 'options': {}}
    if instant:
        for t in p['targets']:
            t.update(instant=True, range=False, format='table')
    if unit:
        p['fieldConfig']['defaults']['unit'] = unit
    if description:
        p['description'] = description
    for key, value in extra.items():
        if key == 'defaults':
            p['fieldConfig']['defaults'].update(value)
        else:
            p[key] = value
    return p


def row(title, y):
    return {'type': 'row', 'title': title, 'collapsed': False, 'gridPos': {'x': 0, 'y': y, 'w': 24, 'h': 1},
            'panels': []}


def threshold(value, color='red'):
    return {'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': None},
                                                         {'color': color, 'value': value}]}}


requests_total = (f'sum by (inferenceservice) (rate(bentoml_service_request_total{{{JOB}, path="/api/predict"}}[{RATE}]))'
                  f' or sum by (inferenceservice) (rate(predictor_requests_total{{{JOB}}}[{RATE}]))')
errors_5xx = (f'sum by (inferenceservice) (rate(bentoml_service_request_total{{{JOB}, path="/api/predict", http_response_code=~"5.."}}[{RATE}]))'
              f' or sum by (inferenceservice) (rate(predictor_requests_total{{{JOB}, code=~"5.."}}[{RATE}]))')
p95 = (f'histogram_quantile(0.95, sum by (inferenceservice, le) (rate(bentoml_service_request_duration_seconds_bucket{{{JOB}, path="/api/predict"}}[{RATE}])))'
       f' or histogram_quantile(0.95, sum by (inferenceservice, le) (rate(predictor_request_duration_seconds_bucket{{{JOB}}}[{RATE}])))')

panels = [
    row('Predictors', 0),
    panel('stat', 'Scrape state', 0, 1, 6, 4, [(f'max by (inferenceservice) (up{{{JOB}}})', '{{inferenceservice}}')],
          description='1 when Prometheus reads /metrics. Alerts PredictorDown, PredictorMissing',
          defaults={**threshold(1, 'green'), 'mappings': [
              {'type': 'value', 'options': {'0': {'text': 'DOWN', 'color': 'red'}, '1': {'text': 'UP', 'color': 'green'}}}]}),
    panel('timeseries', 'Predict requests', 6, 1, 6, 10, [(requests_total, '{{inferenceservice}}')], unit='reqps',
          description='Predict calls only; BentoML probe paths (/readyz, /livez) left out'),
    panel('timeseries', '5xx ratio', 12, 1, 6, 10, [(f'({errors_5xx}) / ({requests_total})', '{{inferenceservice}}')],
          unit='percentunit', description='Alert PredictorErrorRatio: above 5% for 10 minutes',
          defaults={**threshold(0.05), 'min': 0, 'max': 1}),
    panel('timeseries', 'Predict latency p95', 18, 1, 6, 10, [(p95, '{{inferenceservice}}')], unit='s',
          description='Alert PredictorSlow: above 1 s (capstone) or 2.5 s (idcard, movie) for 10 minutes'),
    # Instant query: a range query gives one row per step in the table.
    panel('table', 'Served model versions', 0, 5, 6, 6,
          [('max by (inferenceservice, version) (predictor_model_info{' + JOB + '})', '')], instant=True,
          description='Reported by the idcard and movie servers; the capstone version is v<MLflow version> in its image tag',
          transformations=[{'id': 'labelsToFields', 'options': {}},
                           {'id': 'organize', 'options': {'excludeByName': {'Time': True, 'Value': True}}}],
          options={'showHeader': True}),

    row('Capstone URL classifier: predictions and input drift', 11),
    panel('timeseries', 'Predicted label share', 0, 12, 8, 8,
          [(f'sum by (prediction) (rate(capstone_predictions_total{{{JOB}}}[{RATE}])) / ignoring (prediction) group_left '
            f'sum(rate(capstone_predictions_total{{{JOB}}}[{RATE}]))', '{{prediction}}')],
          unit='percentunit', description='Which label request URLs get. Serving has no ground truth, so this shows whether the output distribution moves',
          defaults={'custom': {'stacking': {'mode': 'normal'}, 'fillOpacity': 60}, 'max': 1}),
    panel('timeseries', 'Predicted label probability (softmax) quantiles', 8, 12, 8, 8,
          [(f'histogram_quantile(0.5, sum by (le) (rate(capstone_prediction_confidence_bucket{{{JOB}}}[{RATE}])))', 'p50'),
           (f'histogram_quantile(0.1, sum by (le) (rate(capstone_prediction_confidence_bucket{{{JOB}}}[{RATE}])))', 'p10')],
          unit='percentunit', description='Probability of the predicted label. A falling p10 means more requests the model is unsure about',
          defaults={'min': 0, 'max': 1}),
    panel('timeseries', 'Drifted column share of the last window', 16, 12, 8, 8,
          [(f'max(capstone_drift_share{{{JOB}}})', 'drifted share')], unit='percentunit',
          description='Every 100 request URLs, 8 columns KS-tested against the training URLs. 0.5 or more is dataset drift (alert CapstoneInputDrift)',
          defaults={**threshold(0.5), 'min': 0, 'max': 1, 'custom': {'drawStyle': 'line', 'lineInterpolation': 'stepAfter',
                                                                     'thresholdsStyle': {'mode': 'dashed'}}}),
    panel('stat', 'Judged windows (dashboard range)', 0, 20, 8, 5,
          [(f'sum by (result) (increase(capstone_drift_windows_total{{{JOB}}}[$__range]))', '{{result}}')],
          description='drifted, not_drifted, error; error is a window whose report could not be made',
          defaults={'decimals': 0}),
    panel('stat', 'Evidently uploads (dashboard range)', 8, 20, 8, 5,
          [(f'sum by (result) (increase(capstone_drift_uploads_total{{{JOB}}}[$__range]))', '{{result}}'),
           (f'sum(increase(capstone_drift_windows_dropped_total{{{JOB}}}[$__range]))', 'dropped')],
          description='failed raises CapstoneDriftReportsFailing. Judgments feed the drift alert whether or not the upload works',
          defaults={'decimals': 0}),
    panel('bargauge', 'KS p-value per column, last window', 16, 20, 8, 5,
          [(f'max by (column) (capstone_drift_p_value{{{JOB}}})', '{{column}}')],
          description='A column below 0.05 drifted',
          defaults={'min': 0, 'max': 1, 'thresholds': {'mode': 'absolute', 'steps': [
              {'color': 'red', 'value': None}, {'color': 'green', 'value': 0.05}]}},
          options={'orientation': 'horizontal', 'displayMode': 'basic'}),

    row('ID card detector and movie recommender: model output', 25),
    panel('timeseries', 'ID card: mean detections per image', 0, 26, 8, 7,
          [(f'sum(rate(idcard_detections_per_image_sum{{{JOB}}}[{RATE}])) / sum(rate(idcard_detections_per_image_count{{{JOB}}}[{RATE}]))', 'mean')],
          description='Near 1 when each image holds one ID card. More zeros mean more missed images'),
    panel('timeseries', 'ID card: detection confidence quantiles', 8, 26, 8, 7,
          [(f'histogram_quantile(0.5, sum by (le) (rate(idcard_detection_confidence_bucket{{{JOB}}}[{RATE}])))', 'p50'),
           (f'histogram_quantile(0.1, sum by (le) (rate(idcard_detection_confidence_bucket{{{JOB}}}[{RATE}])))', 'p10')],
          unit='percentunit', defaults={'min': 0, 'max': 1}),
    panel('timeseries', 'Movie: predicted rating quantiles of recommended movies', 16, 26, 8, 7,
          [(f'histogram_quantile(0.5, sum by (le) (rate(movie_recommendation_score_bucket{{{JOB}}}[{RATE}])))', 'p50'),
           (f'histogram_quantile(0.9, sum by (le) (rate(movie_recommendation_score_bucket{{{JOB}}}[{RATE}])))', 'p90')],
          description='Predicted ratings (0.5 to 5) of the top-k movies returned', defaults={'min': 0, 'max': 5}),
]

for i, p in enumerate(panels, 1):
    p['id'] = i

dashboard = {
    'uid': 'mlops-serving',
    'title': 'MLOps serving: predictors',
    'tags': ['mlops', 'serving'],
    'timezone': 'browser',
    'schemaVersion': 41,
    'refresh': '30s',
    'time': {'from': 'now-3h', 'to': 'now'},
    'editable': True,
    'panels': panels,
}

out = Path(__file__).with_name('mlops-serving.json')
out.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2) + '\n')
print(f'wrote {out} with {len(panels)} panels')

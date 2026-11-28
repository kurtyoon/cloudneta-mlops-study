"""KFP 드리프트 단계와 Bento 서비스가 같이 쓰는 칸(url_stats), Evidently 보고서, 대시보드.

pipeline/bento/url_drift.py는 이 파일의 사본이다. Bento 저장소는 kfp/를 보지 못한다.
토큰 id는 길이가 제각각이라 칸별 검정에 넣을 수 없어, URL 문자열에서 센 숫자 8개를 비교한다.
evidently는 함수 안에서 import한다. url_stats만 쓰는 학습 단계가 Evidently를 읽지 않게 한다.
"""
import functools

import pandas as pd

COLUMNS = ['url_length', 'host_length', 'path_length', 'n_digits', 'n_special', 'n_dots', 'n_slashes',
           'n_equals']

# 칸마다 KS 검정, p-value가 THRESHOLD보다 작으면 드리프트. Evidently 기본 검정(정규화한 바서슈타인 0.1)은
# test split의 window 100개 중 n=200에서 52개, n=100에서 89개를 드리프트로 판정했고, KS는 둘 다 0개였다.
METHOD = 'ks'
THRESHOLD = 0.05

# Evidently 클라이언트 요청의 (연결, 읽기) timeout, 초.
HTTP_TIMEOUT = (5, 30)

# urlsplit은 스킴 없는 'example.com/a'의 호스트를 비운다. 스킴만 건너뛰고 첫 '/', '?', '#'에서 나눈다.
_PARTS = r'^(?:[^:/?#]*://)?(?P<host>[^/?#]*)(?P<path>[^?#]*)'


def url_stats(urls):
    """URL 목록 → COLUMNS 칸의 정수 DataFrame. 스킴은 url_length, n_special, n_slashes에만 들어간다."""
    s = pd.Series(list(urls), dtype=object)
    parts = s.str.extract(_PARTS)
    df = pd.DataFrame({
        'url_length': s.str.len(),
        'host_length': parts['host'].str.len(),
        'path_length': parts['path'].str.len(),
        'n_digits': s.str.count(r'[0-9]'),
        'n_special': s.str.count(r'[^0-9A-Za-z]'),
        'n_dots': s.str.count(r'\.'),
        'n_slashes': s.str.count('/'),
        'n_equals': s.str.count('='),
    })
    return df.astype('int64')


def drift_report(reference, current, drift_share, tags=(), metadata=None):
    """reference와 current를 비교한 Evidently 스냅샷과 요약. 드리프트 칸 비율이 drift_share 이상이면
    데이터셋 드리프트다. dashboard_panels가 이 metric을 읽으므로 같이 고친다."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.metrics import DatasetMissingValueCount, DriftedColumnsCount, RowCount, ValueDrift

    definition = DataDefinition(numerical_columns=COLUMNS)
    drifted = DriftedColumnsCount(columns=COLUMNS, drift_share=drift_share, num_method=METHOD,
                                  num_threshold=THRESHOLD)
    rows = RowCount()
    # url_stats는 빈 값을 만들지 않아 지금은 늘 0이다.
    missing = DatasetMissingValueCount()
    per_column = {c: ValueDrift(column=c, method=METHOD, threshold=THRESHOLD) for c in COLUMNS}
    report = Report([drifted, rows, missing, *per_column.values()], tags=list(tags), metadata=metadata or {})
    snapshot = report.run(reference_data=Dataset.from_pandas(reference[COLUMNS], data_definition=definition),
                          current_data=Dataset.from_pandas(current[COLUMNS], data_definition=definition))

    results = snapshot.metric_results
    share = float(results[drifted.get_metric_id()].share.value)
    columns = {}
    for c, m in per_column.items():
        p_value = float(results[m.get_metric_id()].value)  # KS의 값은 p-value다. 낮을수록 드리프트
        columns[c] = {'p_value': p_value, 'drifted': p_value < THRESHOLD}
    summary = {
        'rows': int(results[rows.get_metric_id()].value),
        'drifted_columns': int(results[drifted.get_metric_id()].count.value),
        'drifted_share': share,
        'drift_detected': share >= drift_share,
        'columns': columns,
    }
    return snapshot, summary


def dashboard_panels():
    """책 6.2.1과 같은 구성의 패널."""
    from evidently.sdk.models import PanelMetric
    from evidently.sdk.panels import bar_plot_panel, counter_panel, line_plot_panel, text_panel

    share = PanelMetric(metric='DriftedColumnsCount', metric_labels={'value_type': 'share'}, legend='드리프트 칸 비율')
    return [
        text_panel('캡스톤 URL 분류기: 요청 URL 드리프트'),
        counter_panel('확인한 요청 수', [PanelMetric(metric='RowCount', legend='요청')], aggregation='sum',
                      size='half'),
        counter_panel('드리프트 칸 비율', [share], aggregation='last', size='half'),
        line_plot_panel('드리프트 칸 비율과 빈 값 비율', [
            share,
            PanelMetric(metric='DatasetMissingValueCount', metric_labels={'value_type': 'share'}, legend='빈 값 비율'),
        ]),
        bar_plot_panel('칸별 KS 검정 p-value', [
            PanelMetric(metric='ValueDrift', metric_labels={'column': c}, legend=c) for c in COLUMNS],
            description=f'낮을수록 reference와 분포가 다르다. {THRESHOLD}보다 작은 칸이 드리프트다'),
    ]


def evidently_project(workspace, name):
    """이름이 name인 프로젝트를 찾거나 만들고, 없는 패널만 더한다. 같은 이름이 여럿이면 가장 먼저 만든 것
    (id가 uuid7이라 만든 순서로 정렬된다)."""
    found = [p for p in workspace.search_project(name) if p.name == name]
    project = min(found, key=lambda p: str(p.id)) if found else workspace.create_project(
        name, description='캡스톤 악성 URL 분류기: 학습 URL과 요청 URL의 url_stats 드리프트')
    have = {p.title for p in project.dashboard.model().panels}
    for panel in dashboard_panels():
        if panel.title not in have:
            project.dashboard.add_panel(panel)
    return project


def remote_workspace(url, secret, timeout=HTTP_TIMEOUT):
    """요청마다 timeout을 주는 RemoteWorkspace. 만들 때 서버에 버전을 묻는다(verify)."""
    return _timeout_workspace_class()(url, secret=secret, timeout=timeout)


@functools.cache
def _timeout_workspace_class():
    from evidently._pydantic_compat import parse_obj_as
    from evidently.errors import EvidentlyError
    from evidently.ui.workspace import RemoteWorkspace
    from requests import Session

    class TimeoutRemoteWorkspace(RemoteWorkspace):
        """0.7.23의 RemoteBase._request에 timeout만 더했다. 원래는 서버가 답하지 않으면 한없이 기다린다.
        evidently를 올릴 때 원래 함수와 다시 맞춘다."""

        def __init__(self, base_url, secret=None, timeout=HTTP_TIMEOUT):
            self.timeout = timeout  # 부모의 __init__이 verify()로 첫 요청을 보내므로 먼저 둔다
            super().__init__(base_url, secret=secret)

        def _request(self, path, method, query_params=None, body=None, response_model=None, cookies=None,
                     headers=None, form_data=False):
            request = self._prepare_request(path, method, query_params, body, cookies, headers, form_data=form_data)
            with Session() as session:
                response = session.send(request.prepare(), timeout=self.timeout)
            if response.status_code >= 400:
                try:
                    details = response.json()['detail']
                    raise EvidentlyError(details)
                except ValueError:
                    pass
            response.raise_for_status()
            if response_model is not None:
                return parse_obj_as(response_model, response.json())
            return response

    return TimeoutRemoteWorkspace

"""URL 드리프트 공용 모듈: 드리프트를 볼 칸(url_stats), Evidently 보고서, 프로젝트와 대시보드, 클라이언트.

KFP의 드리프트 단계(drift.py)와 Bento 서비스(pipeline/bento/service.py)가 같은 칸과 같은 보고서를
같은 Evidently 프로젝트에 넣는다. 원본은 이 파일(kfp/capstone_kfp/url_drift.py)이고
pipeline/bento/url_drift.py는 그대로 복사한 사본이다. 두 폴더는 Gitea에 서로 다른 저장소로
올라가서(git subtree split) Bento 빌드는 kfp/를 보지 못한다. 고칠 때는 이 파일을 고치고
`cp kfp/capstone_kfp/url_drift.py pipeline/bento/url_drift.py`로 복사한다.
pipeline/bento/tests/test_drift_monitor.py가 두 파일이 같은지 확인한다.

드리프트는 모델 입력(토큰 id) 대신 URL 문자열에서 센 숫자 8개로 본다. 토큰 id는 길이가 제각각인
정수 목록이라 칸별 분포 검정에 넣을 수 없고, 서빙 쪽은 요청 URL만 받는다. 이 모듈은 pandas만
import하고 evidently는 함수 안에서 불러온다. url_stats만 쓰는 곳(학습 단계)이 Evidently를 읽지 않게 한다.
"""
import functools

import pandas as pd

COLUMNS = ['url_length', 'host_length', 'path_length', 'n_digits', 'n_special', 'n_dots', 'n_slashes',
           'n_equals']

# 칸마다의 드리프트 검정: KS 검정, p-value가 THRESHOLD보다 작으면 그 칸은 드리프트다(11-05 결정).
# Evidently가 고르는 기본 검정(reference가 1000행을 넘으면 정규화한 바서슈타인 거리 0.1)은 실제 데이터에서
# 같은 분포의 요청도 데이터셋 드리프트로 판정했다: 학습 split과 test split에서 뽑은 window 100개 중
# n=200에서 52개, n=100에서 89개. KS p<0.05로 고정하면 둘 다 0개였고, phishing만이나 defacement만으로 된
# window는 20개 중 20개를 잡았다(benign만으로 된 window는 대부분 놓친다).
METHOD = 'ks'
THRESHOLD = 0.05

# Evidently 클라이언트 요청의 (연결, 읽기) timeout, 초.
HTTP_TIMEOUT = (5, 30)

# 스킴(http:// 등)은 호스트와 경로를 나눌 때만 건너뛴다. 원본 데이터 대부분은 스킴 없이 'example.com/a'처럼
# 적혀 있고, urllib.parse.urlsplit은 그런 URL의 호스트를 비우고 전체를 경로로 본다.
# 호스트: 스킴 뒤부터 첫 '/', '?', '#' 앞까지(포트와 사용자 정보 포함). 경로: 그 뒤부터 '?', '#' 앞까지.
_PARTS = r'^(?:[^:/?#]*://)?(?P<host>[^/?#]*)(?P<path>[^?#]*)'


def url_stats(urls):
    """URL 목록 → COLUMNS 칸의 정수 DataFrame(행 순서는 입력 순서).

    n_digits는 ASCII 숫자, n_special은 ASCII 영문자와 숫자가 아닌 모든 글자 수다.
    url_length, n_special, n_slashes는 URL 전체에서 세므로 스킴('http://'의 7글자, 특수 문자 3개, '/' 2개)도
    들어간다. host_length와 path_length에는 들어가지 않는다.
    """
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
    """reference(학습 URL의 url_stats)와 current를 비교한 Evidently 스냅샷과 요약.

    칸마다 KS 검정을 하고 p-value가 THRESHOLD보다 작으면 그 칸을 드리프트로 센다. 칸별 막대(ValueDrift)와
    데이터셋 판정(DriftedColumnsCount)에 같은 검정과 임곗값을 준다. 드리프트가 난 칸의 비율이 drift_share
    이상이면 데이터셋 드리프트다(Evidently의 DatasetDriftMetric과 같은 기준, 책 6.2.1의 데이터셋 임곗값).
    대시보드 패널(dashboard_panels)이 이 보고서의 metric을 읽으므로 둘을 같이 고친다.
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.metrics import DatasetMissingValueCount, DriftedColumnsCount, RowCount, ValueDrift

    definition = DataDefinition(numerical_columns=COLUMNS)
    drifted = DriftedColumnsCount(columns=COLUMNS, drift_share=drift_share, num_method=METHOD,
                                  num_threshold=THRESHOLD)
    rows = RowCount()
    # 책의 DatasetMissingValuesMetric. url_stats는 빈 값을 만들지 않으므로 지금은 늘 0이다.
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
    """책 6.2.1의 패널과 같은 구성: 제목, 확인한 요청 수(합), 드리프트 칸 비율(마지막),
    드리프트 칸 비율과 빈 값 비율(선), 칸별 KS p-value(막대)."""
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
    """이름이 name인 프로젝트를 찾고, 없으면 만든다. 대시보드에 없는 제목의 패널만 더한다.

    여러 번 불러도 프로젝트와 패널이 늘지 않는다. KFP 단계와 Bento 서비스가 같은 프로젝트를 쓴다.
    search_project는 이름 일부로도 찾으므로 이름이 같은 것만 고른다. 둘이 처음에 동시에 만들어 같은 이름이
    여럿이면 가장 먼저 만든 것을 쓴다(프로젝트 id는 만든 시각순인 uuid7이다).
    """
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
        """evidently 0.7.23의 RemoteWorkspace는 RemoteBase._request(evidently/legacy/ui/workspace/remote.py)로
        모든 요청을 보내는데, 그 함수는 timeout 없이 보낸다. 서버가 연결을 받고 응답하지 않으면 한없이 기다린다.
        아래 _request는 그 함수와 같고 timeout만 더했다. evidently를 올릴 때 원래 함수와 다시 맞춘다."""

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

"""Bento 서비스의 마이크로배치 드리프트 감시(책 6.2.3의 MonitoringService).

요청 URL이 window개 차면 build.py가 넣은 reference와 비교한 보고서를 KFP 단계와 같은 프로젝트에 넣는다.
책과 달리 window가 찰 때마다 한 번 만들고 비운다. 보고서는 백그라운드 스레드가 만들고, 줄 선 window가
max_pending개를 넘으면 버린다. window가 프로세스 메모리에 있어 서비스는 워커 하나로 돈다.
"""
import logging
import os
import queue
import threading
from concurrent.futures import Future

import pandas as pd

import url_drift

# BentoML은 'bentoml' 로거만 서버 로그로 내보낸다(다른 로거는 WARNING 미만을 버린다).
log = logging.getLogger('bentoml')

PROJECT = 'capstone-url-classifier'  # kfp/capstone_kfp/config.py의 EVIDENTLY_PROJECT
DEFAULT_WINDOW = 100  # 스터디장 환경의 window. 책은 테스트용으로 50
MAX_PENDING = 10  # 줄을 서는 window 수의 상한. 지금 만들고 있는 보고서는 세지 않는다


class DriftMonitor:
    def __init__(self, reference, window, workspace_factory, project=PROJECT, drift_share=0.5, metadata=None,
                 max_pending=MAX_PENDING):
        """workspace_factory는 첫 보고서를 보낼 때 부른다. 서비스 시작이 Evidently에 묶이지 않게 한다."""
        if window < 2:
            raise ValueError(f'window는 2 이상이어야 한다. 요청 1건으로는 분포를 비교하지 못한다: {window}')
        self.reference = reference
        self.window = window
        self.workspace_factory = workspace_factory
        self.project = project
        self.drift_share = drift_share
        self.metadata = metadata or {}
        self.max_pending = max_pending
        self._urls = []
        self._lock = threading.Lock()  # 동기 API는 스레드 풀에서 돈다. 요청 여럿이 동시에 붙인다
        self._queue = queue.Queue(maxsize=max_pending)
        self._workspace = None
        self._project_id = None
        self._thread = threading.Thread(target=self._worker, name='drift-report', daemon=True)
        self._thread.start()

    def add(self, url):
        """window가 차서 줄에 넣었으면 그 보고서의 Future, 아니면 None."""
        with self._lock:
            self._urls.append(url)
            if len(self._urls) < self.window:
                return None
            batch, self._urls = self._urls, []
        future = Future()
        try:
            self._queue.put_nowait((batch, future))
        except queue.Full:
            log.warning('drift: 보고서를 기다리는 window가 %d개라 요청 %d건의 window를 버린다',
                        self.max_pending, len(batch))
            return None
        return future

    def _worker(self):
        while True:
            urls, future = self._queue.get()
            future.set_result(self._report(urls))

    def _report(self, urls):
        try:
            if self._workspace is None:
                workspace = self.workspace_factory()
                self._project_id = url_drift.evidently_project(workspace, self.project).id
                self._workspace = workspace
            snapshot, summary = url_drift.drift_report(self.reference, url_drift.url_stats(urls), self.drift_share,
                                                       tags=['bento'], metadata=self.metadata)
            self._workspace.add_run(self._project_id, snapshot, name=f'bento-v{self.metadata.get("model_version")}')
            log.info('drift: 요청 %d건, 드리프트 칸 %d/%d(비율 %.3f)', summary['rows'], summary['drifted_columns'],
                     len(summary['columns']), summary['drifted_share'])
            return summary
        except Exception:
            # 이 window는 버리고 다음 window에서 workspace와 프로젝트를 다시 찾는다.
            self._workspace = self._project_id = None
            log.exception('drift: 요청 %d건의 보고서를 Evidently에 넣지 못했다', len(urls))
            return None


def from_env(reference_path, metadata):
    """EVIDENTLY_URL과 EVIDENTLY_SECRET이 있으면 감시를 켠다. 없거나 켜다 실패하면 None, 예측만 한다."""
    url, secret = os.environ.get('EVIDENTLY_URL'), os.environ.get('EVIDENTLY_SECRET')
    if not url or not secret:
        log.warning('drift: EVIDENTLY_URL 또는 EVIDENTLY_SECRET이 없어 드리프트 감시를 끈다. 예측은 그대로 한다')
        return None
    try:
        window = int(os.environ.get('DRIFT_WINDOW', DEFAULT_WINDOW))
        monitor = DriftMonitor(pd.read_parquet(reference_path), window,
                               lambda: url_drift.remote_workspace(url, secret), metadata=metadata)
    except Exception:
        log.exception('drift: 드리프트 감시를 켜지 못해 끈다. 예측은 그대로 한다')
        return None
    log.info('drift: window %d건마다 %s의 프로젝트 %s에 보고서를 넣는다', window, url, PROJECT)
    return monitor

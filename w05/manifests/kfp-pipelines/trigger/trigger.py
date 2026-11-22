"""Creates one KFP run of a PipelineVersion declared in git and waits for it.

Run by a Job in runs/ (one Job per run). The Job's name is the run's display
name, and a run with that name in the experiment is reused instead of created
again, so a retried or re-synced Job never starts a second run. The Job ends
when the run does: 0 for SUCCEEDED, 1 otherwise. Argo CD treats a running Job
as progressing, so Jobs in later sync waves start only after it finishes, one
run at a time on this node.

At the end it prints each task's start and end from the run details, the same
times the KFP UI shows, so the Job's log keeps them after the run's pods are
gone.

Environment:
  KFP_API         API server base URL (in-cluster, no auth in standalone KFP)
  PIPELINE        Pipeline object name
  VERSION         PipelineVersion object name
  RUN_NAME        run display name (the Job's name)
  EXPERIMENT      experiment display name, created when missing
  PARAMETERS      JSON object of pipeline parameters, default {}
  POLL_SECONDS    default 20
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

API = os.environ['KFP_API'].rstrip('/') + '/apis/v2beta1'
DONE = {'SUCCEEDED', 'FAILED', 'CANCELED', 'SKIPPED'}


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(API + path, data=data, method=method, headers={'Content-Type': 'application/json'})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors='replace')
            if error.code < 500 or attempt == 4:
                sys.exit(f'{method} {path}: HTTP {error.code} {detail}')
        except OSError as error:  # URLError, a dropped connection, a timeout
            if attempt == 4:
                sys.exit(f'{method} {path}: {error}')
        time.sleep(5 * (attempt + 1))


def list_all(path, key):
    """Every item of a list call, matched by name here rather than with the
    API's filter: in Kubernetes native mode pipelines are listed from the
    Pipeline objects, where server-side filtering is limited."""
    items, token = [], ''
    while True:
        sep = '&' if '?' in path else '?'
        page = call('GET', f'{path}{sep}page_size=100' + (f'&page_token={urllib.parse.quote(token)}' if token else ''))
        items += page.get(key, [])
        token = page.get('next_page_token', '')
        if not token:
            return items


def find_pipeline_version(pipeline, version):
    match = [p for p in list_all('/pipelines', 'pipelines') if pipeline in (p.get('name'), p.get('display_name'))]
    if not match:
        sys.exit(f'pipeline {pipeline} not found')
    pipeline_id = match[0]['pipeline_id']
    versions = list_all(f'/pipelines/{pipeline_id}/versions', 'pipeline_versions')
    match = [v for v in versions if version in (v.get('name'), v.get('display_name'))]
    if not match:
        sys.exit(f'version {version} of {pipeline} not found among {[v.get("name") for v in versions]}')
    return pipeline_id, match[0]['pipeline_version_id']


def experiment_id(name):
    found = [e for e in list_all('/experiments', 'experiments') if e.get('display_name') == name]
    if found:
        return found[0]['experiment_id']
    return call('POST', '/experiments', {'display_name': name})['experiment_id']


def parse_time(value):
    if not value or value.startswith('1970'):
        return None
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def print_tasks(run):
    rows = []
    for task in run.get('run_details', {}).get('task_details', []):
        # One row per pipeline step: the run details also list each step's
        # driver, its executor and loop iterations, under the same times.
        name = task.get('display_name', '')
        if not name or 'driver' in name or name.startswith('executor') or name.endswith(')'):
            continue
        start, end = parse_time(task.get('start_time')), parse_time(task.get('end_time'))
        rows.append((task.get('display_name', ''), task.get('state', ''), start, end))
    rows.sort(key=lambda r: (r[2].timestamp() if r[2] else float('inf'), r[0]))
    print(f'{"task":40} {"state":10} {"start (UTC)":20} {"seconds":>8}')
    for name, state, start, end in rows:
        seconds = f'{(end - start).total_seconds():8.0f}' if start and end else '       -'
        started = start.strftime('%Y-%m-%d %H:%M:%S') if start else '-'
        print(f'{name[:40]:40} {state:10} {started:20} {seconds}')
    start, end = parse_time(run.get('created_at')), parse_time(run.get('finished_at'))
    if start and end:
        print(f'run {run["state"]} in {(end - start).total_seconds():.0f} s (created {start:%H:%M:%S}, finished {end:%H:%M:%S} UTC)')


def main():
    pipeline, version, run_name = os.environ['PIPELINE'], os.environ['VERSION'], os.environ['RUN_NAME']
    parameters = json.loads(os.environ.get('PARAMETERS') or '{}')
    poll = int(os.environ.get('POLL_SECONDS', '20'))

    pipeline_id, version_id = find_pipeline_version(pipeline, version)
    exp_id = experiment_id(os.environ['EXPERIMENT'])
    runs = [r for r in list_all(f'/runs?experiment_id={exp_id}', 'runs') if r.get('display_name') == run_name]
    if runs:
        run_id = runs[0]['run_id']
        print(f'run {run_name} exists ({run_id}), following it')
    else:
        run_id = call('POST', '/runs', {
            'display_name': run_name,
            'experiment_id': exp_id,
            'pipeline_version_reference': {'pipeline_id': pipeline_id, 'pipeline_version_id': version_id},
            'runtime_config': {'parameters': parameters},
        })['run_id']
        print(f'created run {run_name} ({run_id}) from {version} with {json.dumps(parameters)}')
    sys.stdout.flush()

    state = None
    while True:
        run = call('GET', f'/runs/{run_id}')
        if run.get('state') != state:
            state = run.get('state')
            print(f'{datetime.now(timezone.utc):%H:%M:%S} {state}', flush=True)
        if state in DONE:
            break
        time.sleep(poll)
    print_tasks(run)
    sys.exit(0 if state == 'SUCCEEDED' else 1)


if __name__ == '__main__':
    main()

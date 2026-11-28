"""KFP의 최근 run을 출력한다: 이름, 상태, 만든 시각, 끝난 시각.

    KFP_HOST=https://<kfp 호스트> KFP_USERNAME=<사용자> KFP_PASSWORD=<app password> \
        .venv-kfp/bin/python tools/list_runs.py [--limit 5]
"""
import argparse
import os
import sys
from pathlib import Path

import kfp

sys.path.insert(0, str(Path(__file__).parent))
from submit import AuthentikBasic  # noqa: E402

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=5)
    args = parser.parse_args()

    client = kfp.Client(host=os.environ['KFP_HOST'].rstrip('/'),
                        credentials=AuthentikBasic(os.environ['KFP_USERNAME'], os.environ['KFP_PASSWORD']))
    runs = client.list_runs(page_size=args.limit, sort_by='created_at desc').runs or []
    for r in runs:
        print(f'{r.run_id}  {r.state:<10}  {r.created_at:%m-%d %H:%M}  {r.finished_at:%m-%d %H:%M}  {r.display_name}')

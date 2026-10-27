#!/bin/sh
# requirements.in을 이미지용 lock 두 개로 푼다. 이미지는 linux/arm64, CPython 3.11 하나뿐이다.
# - requirements.lock: 2024-12-21 기준(mlflow-skinny만 서버와 같은 3.17.2를 위해 2026-10-26 기준).
#   torch는 PyPI의 aarch64 wheel이 CPU 전용이라 다른 인덱스가 필요 없다.
# - requirements-kfp.lock: KFP 경량 컴포넌트의 실행기(kfp.dsl.executor_main)만 쓰도록
#   kfp 2.9.0을 의존성 없이 넣는다. KFP가 기본으로 Pod 시작 때마다 하는
#   `pip install kfp==2.9.0 --no-deps`를 이미지 빌드 때 한 번, 해시를 확인하며 한다.
# 둘 다 tools/filter-hashes.py로 linux aarch64 cp311 파일의 해시만 남긴다(PyPI JSON API).
set -eu
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
common="--python-version 3.11 --python-platform aarch64-unknown-linux-gnu --generate-hashes --no-header --quiet"

uv pip compile requirements.in $common \
  --exclude-newer 2024-12-21T00:00:00Z --exclude-newer-package mlflow-skinny=2026-10-26T00:00:00Z \
  -o "$tmp/main.txt"
{
  echo "# requirements.in을 tools/lock.sh로 푼 결과(linux aarch64, CPython 3.11). 손으로 고치지 않는다."
  python3 tools/filter-hashes.py "$tmp/main.txt"
} > requirements.lock

echo "kfp==2.9.0" > "$tmp/kfp.in"
uv pip compile "$tmp/kfp.in" $common --no-deps --no-annotate -o "$tmp/kfp.txt"
{
  echo "# kfp 2.9.0, 의존성 없이(pip install --no-deps). tools/lock.sh가 만든다."
  python3 tools/filter-hashes.py "$tmp/kfp.txt"
} > requirements-kfp.lock

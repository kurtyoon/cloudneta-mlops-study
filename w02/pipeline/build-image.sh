#!/usr/bin/env bash
# Build the serving image tagged <git commit>-<model hash>.
# The commit alone is not enough: model/ is gitignored, so two builds of the
# same commit can carry different models.
set -euo pipefail
cd "$(dirname "$0")"
image=${IMAGE:-drl-serve}

if [ -n "$(git status --porcelain -- .)" ]; then
  echo "uncommitted changes under pipeline/; commit them first" >&2
  exit 1
fi
if [ -z "$(find model -type f ! -name .gitignore ! -name .DS_Store)" ]; then
  echo "no model files in model/" >&2
  exit 1
fi

git_sha=$(git rev-parse HEAD)
# Hash file names and contents together, NUL-separated so spaces are safe.
model_sha=$(find model -type f ! -name .gitignore ! -name .DS_Store -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
tag="${git_sha:0:8}-${model_sha:0:12}"

docker build -t "${image}:${tag}" \
  --build-arg GIT_SHA="$git_sha" --build-arg MODEL_SHA256="$model_sha" .
echo "${image}:${tag}"

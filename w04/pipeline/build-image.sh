#!/usr/bin/env bash
# 서빙 이미지를 <커밋>-<모델 해시> 태그로 빌드한다.
# model/은 git에 없어서 같은 커밋이라도 모델이 다를 수 있다.
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
# 파일 이름과 내용을 함께 해시한다.
model_sha=$(find model -type f ! -name .gitignore ! -name .DS_Store -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
tag="${git_sha:0:8}-${model_sha:0:12}"

docker build -t "${image}:${tag}" \
  --build-arg GIT_SHA="$git_sha" --build-arg MODEL_SHA256="$model_sha" .
echo "${image}:${tag}"

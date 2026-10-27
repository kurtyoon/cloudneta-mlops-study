# 파이프라인 다섯 단계가 모두 쓰는 이미지 하나(linux/arm64).
# python:3.11.18-slim-trixie를 2026-10-27에 받은 인덱스 다이제스트로 고정한다.
# 3.11은 책의 CH5와 클러스터의 feast-apply Job이 Feast 0.40.1을 돌리는 버전이다.
FROM python:3.11.18-slim-trixie@sha256:82209229d3fda30d71e603ec95bb4bea0476635ba6df3df2e32284e036654f51

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/capstone

# 모든 패키지를 해시로 확인하고 wheel로만 설치한다(tools/lock.sh).
# kfp 2.9.0은 PyPI에 sdist만 있어서, 베이스 이미지의 setuptools로 빌드한다(--no-build-isolation).
# 그래서 빌드 도구를 PyPI에서 따로 받지 않는다.
COPY requirements.lock requirements-kfp.lock ./
RUN pip install --require-hashes --only-binary :all: -r requirements.lock \
    && pip install --require-hashes --no-deps --no-build-isolation -r requirements-kfp.lock

# 토크나이저와 BERT 설정(config.json)을 고정한 revision으로 이미지에 넣는다.
# post-2의 학습(preprocessing.py, classifier.py)과 1주차 서빙 이미지가 쓴 revision이다.
# 저장할 디렉터리 이름을 bert-base-cased로 하지 않는다. WORKDIR에 그 이름이 생기면
# 다음 from_pretrained('bert-base-cased')가 Hub 대신 그 디렉터리를 읽는다.
# HF 캐시는 남기지 않는다. Pod는 Hugging Face Hub에 접속하지 않는다.
ARG TOKENIZER_REVISION=cd5ef92a9fb2f889e972770a36d4ed042daf221e
RUN HF_HOME=/tmp/hf python -c "from transformers import BertConfig, BertTokenizer; \
r = '${TOKENIZER_REVISION}'; \
t = BertTokenizer.from_pretrained('bert-base-cased', revision=r); \
c = BertConfig.from_pretrained('bert-base-cased', revision=r); \
t.save_pretrained('/opt/capstone/bert'); c.save_pretrained('/opt/capstone/bert')" \
    && rm -rf /tmp/hf

COPY capstone_kfp ./capstone_kfp
# 이미지에 git이 없다. MLflow가 run마다 git 정보를 찾다 경고를 내지 않게 한다.
ENV PYTHONPATH=/opt/capstone \
    GIT_PYTHON_REFRESH=quiet

ARG GIT_SHA=unknown
LABEL org.opencontainers.image.revision="${GIT_SHA}" \
      capstone.tokenizer.revision="${TOKENIZER_REVISION}"

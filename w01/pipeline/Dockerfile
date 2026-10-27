# python:3.12-slim as it was in June 2024 (Python 3.12.4, Debian bookworm),
# pinned by the index digest docker-library/repo-info recorded on 2024-06-27.
# The tag alone now resolves to Python 3.12.15 on Debian trixie.
FROM python:3.12-slim@sha256:da2d7af143dab7cd5b0d5a5c9545fe14e67fc24c394fcf1cf15e8ea16cbd8637

WORKDIR /app

# gcc builds psutil 5.9.8, which has no linux/arm64 wheel (amd64 needs no
# compiler). apt is still an unpinned build input.
RUN apt-get update && apt-get install -y gcc

# requirements.lock is requirements.in resolved as of 2024-06-30 with hashes
# (see its header). fastapi used to be installed unpinned after this step.
COPY requirements.lock /app
RUN pip install --require-hashes -r requirements.lock

# The tokenizer used to be fetched from the Hugging Face Hub when the server
# started. Bake it in at a fixed revision, without leaving a cache copy.
ARG TOKENIZER_REVISION=cd5ef92a9fb2f889e972770a36d4ed042daf221e
RUN HF_HOME=/tmp/hf python -c "from transformers import BertTokenizer; BertTokenizer.from_pretrained('bert-base-cased', revision='${TOKENIZER_REVISION}').save_pretrained('/app/tokenizer')" \
    && rm -rf /tmp/hf

COPY serve.py /app
COPY model /app/model

# model/ is gitignored, so the commit does not identify the model.
# build-image.sh passes the model hash here and into the image tag.
ARG GIT_SHA
ARG MODEL_SHA256
LABEL org.opencontainers.image.revision="${GIT_SHA}" \
      drl.model.sha256="${MODEL_SHA256}" \
      drl.tokenizer.revision="${TOKENIZER_REVISION}"

EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]

# 원본 커밋 당시의 python:3.12-slim 다이제스트다. 태그만 쓰면 지금은 다른 Python과 Debian을 받는다.
FROM python:3.12-slim@sha256:da2d7af143dab7cd5b0d5a5c9545fe14e67fc24c394fcf1cf15e8ea16cbd8637

WORKDIR /app

# psutil 5.9.8은 linux/arm64 wheel이 없어 gcc로 빌드한다. apt 패키지 버전은 고정하지 않는다.
RUN apt-get update && apt-get install -y gcc

COPY requirements.lock /app
RUN pip install --require-hashes -r requirements.lock

# serve.py는 Hub에 접속하지 않고 /app/tokenizer에서 토크나이저를 읽는다.
ARG TOKENIZER_REVISION=cd5ef92a9fb2f889e972770a36d4ed042daf221e
RUN HF_HOME=/tmp/hf python -c "from transformers import BertTokenizer; BertTokenizer.from_pretrained('bert-base-cased', revision='${TOKENIZER_REVISION}').save_pretrained('/app/tokenizer')" \
    && rm -rf /tmp/hf

COPY serve.py /app
COPY model /app/model

# model/은 git에 없어서 커밋만으로는 모델을 알 수 없다. 모델 해시는 build-image.sh가 넘긴다.
ARG GIT_SHA
ARG MODEL_SHA256
LABEL org.opencontainers.image.revision="${GIT_SHA}" \
      drl.model.sha256="${MODEL_SHA256}" \
      drl.tokenizer.revision="${TOKENIZER_REVISION}"

EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]

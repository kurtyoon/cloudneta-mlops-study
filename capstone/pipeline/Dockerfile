FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc

COPY requirements.txt /app
COPY serve.py /app
COPY model /app/model

RUN pip install -r requirements.txt
RUN pip install fastapi uvicorn

EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]

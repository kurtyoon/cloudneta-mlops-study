"""FastAPI 서버(../serve.py)와 같은 계약: POST /api/predict {"url": "..."} -> {"prediction": "Benign"}.
요청 URL은 드리프트 감시(drift_monitor.py)에도 넘긴다. EVIDENTLY_URL과 EVIDENTLY_SECRET이 없으면 감시는 꺼진다.
"""
import bentoml
import torch
from bentoml.exceptions import InvalidArgument
from transformers import BertForSequenceClassification, BertTokenizer

import drift_monitor

LABELS = ['Benign', 'Phishing', 'Defacement', 'Malware']


# 워커는 하나다. 드리프트 window가 프로세스 메모리에 있어서, 워커가 여럿이면 워커마다 window를 따로
# 쌓고 요청이 나뉘어 window가 늦게 찬다(스터디장 실습 노트: api_server.workers=1).
@bentoml.service(name='capstone_url_classifier', workers=1)
class CapstoneUrlClassifier:
    # bentoml build가 이 모델을 Bento에 넣는다. 서빙 중에는 Bento 안의 모델 폴더를 가리킨다.
    bento_model = bentoml.models.BentoModel('capstone-url-classifier:latest')

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.bento_model.path_of('tokenizer'))
        self.model = BertForSequenceClassification.from_pretrained(self.bento_model.path_of('model'))
        self.model.eval()
        self.max_length = int(self.bento_model.info.metadata['max_length'])
        metadata = self.bento_model.info.metadata
        # build.py가 champion run의 drift/reference.parquet를 모델 폴더에 넣었다.
        self.monitor = drift_monitor.from_env(
            self.bento_model.path_of('drift/reference.parquet'),
            metadata={'source': 'bento', 'model_version': str(metadata['mlflow_model_version']),
                      'mlflow_run_id': str(metadata['mlflow_run_id'])})

    def model_inputs(self, url):
        # 학습 step과 같은 입력: max_length에서 자르고 0으로 채운다. attention_mask는 패딩까지 1이다.
        ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(url))[: self.max_length]
        input_ids = torch.tensor([ids + [0] * (self.max_length - len(ids))], dtype=torch.long)
        return input_ids, torch.ones_like(input_ids), torch.zeros_like(input_ids)

    @bentoml.api(route='/api/predict')
    def predict(self, url: str) -> dict[str, str]:
        if not url:
            raise InvalidArgument('url is empty')
        with torch.inference_mode():
            logits = self.model(*self.model_inputs(url)).logits
        if self.monitor is not None:
            self.monitor.add(url)  # window가 차면 백그라운드 스레드에서 보고서를 보낸다. 기다리지 않는다
        return {'prediction': LABELS[int(torch.argmax(logits, dim=-1)[0])]}

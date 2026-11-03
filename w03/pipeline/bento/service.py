"""FastAPI 서버(../serve.py)와 같은 계약: POST /api/predict {"url": "..."} -> {"prediction": "Benign"}."""
import bentoml
import torch
from bentoml.exceptions import InvalidArgument
from transformers import BertForSequenceClassification, BertTokenizer

LABELS = ['Benign', 'Phishing', 'Defacement', 'Malware']


@bentoml.service(name='capstone_url_classifier')
class CapstoneUrlClassifier:
    # bentoml build가 이 모델을 Bento에 넣는다. 서빙 중에는 Bento 안의 모델 폴더를 가리킨다.
    bento_model = bentoml.models.BentoModel('capstone-url-classifier:latest')

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.bento_model.path_of('tokenizer'))
        self.model = BertForSequenceClassification.from_pretrained(self.bento_model.path_of('model'))
        self.model.eval()
        self.max_length = int(self.bento_model.info.metadata['max_length'])

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
        return {'prediction': LABELS[int(torch.argmax(logits, dim=-1)[0])]}

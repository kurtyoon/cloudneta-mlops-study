"""FastAPI 서버(../serve.py)와 같은 계약: POST /api/predict {"url": "..."} -> {"prediction": "Benign"}."""
import bentoml
import torch
from bentoml.exceptions import InvalidArgument
from bentoml.io import JSON

LABELS = ['Benign', 'Phishing', 'Defacement', 'Malware']

model_ref = bentoml.pytorch.get('capstone-url-classifier:latest')
tokenizer = model_ref.custom_objects['tokenizer']
max_length = int(model_ref.info.metadata['max_length'])
runner = model_ref.to_runner()

svc = bentoml.Service('capstone_url_classifier', runners=[runner])


def model_inputs(url):
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(url))[:max_length]
    input_ids = torch.tensor([ids + [0] * (max_length - len(ids))], dtype=torch.long)
    return input_ids, torch.ones_like(input_ids), torch.zeros_like(input_ids)


@svc.api(input=JSON(), output=JSON(), route='/api/predict')
async def predict(body):
    url = body.get('url') if isinstance(body, dict) else None
    if not isinstance(url, str) or not url:
        raise InvalidArgument('the body must be {"url": "<url>"}')
    logits = (await runner.async_run(*model_inputs(url)))[0]
    return {'prediction': LABELS[int(torch.argmax(logits, dim=-1)[0])]}

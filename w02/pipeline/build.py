import bentoml
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained("./model", local_files_only=True)
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

bentoml.transformers.save_model(
    "malicious-url-classifier",
    model,
    signatures={"classify": {"batchable": False}},
    custom_objects={"tokenizer": tokenizer},
)

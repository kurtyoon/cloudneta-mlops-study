import torch
from transformers import BertTokenizer, BertForSequenceClassification
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


class BertClassifier:
    def __init__(self, model_path):
        self.model = BertForSequenceClassification.from_pretrained(model_path, from_tf=False, local_files_only=True)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def preprocess(self, url):
        tokens = self.tokenizer.tokenize(url)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        return input_ids, attention_mask, token_type_ids

    def predict(self, url):
        input_ids, attention_mask, token_type_ids = self.preprocess(url)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        return predictions.item()

    def label_translation(self, index):
        translations = {
            0: "Benign",
            1: "Phishing",
            2: "Defacement",
            3: "Malware"
        }
        return translations[index]


app = FastAPI()

bert_classifier = BertClassifier("./model")


class URLInput(BaseModel):
    url: str


@app.post("/api/predict")
async def classify_url(input_data: URLInput):
    prediction_index = bert_classifier.predict(input_data.url)
    prediction_label = bert_classifier.label_translation(prediction_index)
    return {"prediction": prediction_label}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

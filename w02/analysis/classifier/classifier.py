import argparse
import hashlib
import json
import pandas as pd
import torch
import warnings
import ast
import time

import mlflow
import mlflow.pytorch

from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertConfig, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

warnings.filterwarnings("ignore", category=FutureWarning)

LABELS = ['benign', 'phishing', 'defacement', 'malware']  # preprocessing/util.py 순서
TOKENIZER_REVISION = 'cd5ef92a9fb2f889e972770a36d4ed042daf221e'


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class URLDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data.reset_index(drop=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'token_type_ids': item.get('token_type_ids'),
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)


class URLClassifier:
    def __init__(self, url_file, args):
        self.url_file = url_file
        self.args = args
        self.device = pick_device()
        self.config = BertConfig.from_pretrained(
            'bert-base-cased', revision=TOKENIZER_REVISION, num_labels=4, hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1, hidden_size=args.hidden_size,
            num_attention_heads=args.num_heads, num_hidden_layers=args.num_layers,
            intermediate_size=args.hidden_size * 4,
            max_position_embeddings=max(512, args.max_length)
        )
        self.model = BertForSequenceClassification(self.config)
        self.model.to(self.device)

    def load_and_prepare_data(self):
        # w02: URL data only (preprocessing.py no longer writes feature_ids.csv).
        url_data = pd.read_csv(self.url_file)

        # Parse and prepare tensors
        url_data['token_ids'] = url_data['token_ids'].apply(ast.literal_eval)
        url_data['token_ids'] = url_data['token_ids'].apply(lambda x: [item for sublist in x for item in sublist])

        max_length = self.args.max_length
        self.truncated = int((url_data['token_ids'].apply(len) > max_length).sum())
        # cut at max_length, then pad with 0 until max_length
        url_data['token_ids'] = url_data['token_ids'].apply(lambda x: x[:max_length] + [0] * (max_length - len(x[:max_length])))

        # token_ids 리스트를 텐서로 변환
        url_data['input_ids'] = url_data['token_ids'].apply(lambda ids: torch.tensor(ids, dtype=torch.long))

        # 이미 텐서로 변환된 input_ids를 사용하여 attention_mask와 token_type_ids를 생성
        # (패딩까지 1인 attention_mask는 원본 그대로 둔다. 4주차에 다룬다.)
        url_data['attention_mask'] = url_data['input_ids'].apply(lambda tensor: torch.ones_like(tensor))
        url_data['token_type_ids'] = url_data['input_ids'].apply(lambda tensor: torch.zeros_like(tensor))

        combined_data = url_data

        # Convert labels
        labels = torch.tensor(combined_data['label'].values, dtype=torch.long)

        # Split the dataset
        train_idx, test_idx = train_test_split(range(len(combined_data)), test_size=0.2, stratify=labels,
                                               random_state=self.args.seed)

        train_dataset = combined_data.iloc[train_idx]
        test_dataset = combined_data.iloc[test_idx]

        self.train_dataset = URLDataset(train_dataset, labels[train_idx])
        self.test_dataset = URLDataset(test_dataset, labels[test_idx])

    def train_model(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=self.args.lr)

        for epoch in range(self.args.epochs):
            self.model.train()
            start_time = time.time()
            total_loss, n = 0.0, 0
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                n += 1

            seconds = time.time() - start_time
            metrics = self.evaluate_model()
            mlflow.log_metrics({'train_loss': total_loss / n, 'epoch_seconds': seconds,
                                'test_accuracy': metrics['accuracy'],
                                'test_macro_f1': metrics['macro_f1']}, step=epoch + 1)
            print(f'{epoch + 1}번째 epoch 완료, 소요 시간: {seconds:.1f}초, train loss {total_loss / n:.4f}, '
                  f'accuracy {metrics["accuracy"]:.4f}, macro F1 {metrics["macro_f1"]:.4f}')
        self.final_metrics = metrics

    def evaluate_model(self):
        test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())

        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, labels=range(4), zero_division=0)
        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'macro_f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
            'per_class': {LABELS[i]: {'precision': float(precision[i]), 'recall': float(recall[i]),
                                      'f1': float(f1[i]), 'support': int(support[i])} for i in range(4)},
            'confusion_matrix': confusion_matrix(all_labels, all_preds, labels=range(4)).tolist(),
        }

    def log_results(self):
        m = self.final_metrics
        for label, v in m['per_class'].items():
            mlflow.log_metrics({f'{label}_precision': v['precision'], f'{label}_recall': v['recall'],
                                f'{label}_f1': v['f1']})
        mlflow.log_dict({'labels': LABELS, 'rows_true_cols_pred': m['confusion_matrix']},
                        'confusion_matrix.json')
        mlflow.log_dict(m['per_class'], 'per_class_metrics.json')

    def save_model(self):
        torch.save(self.model.state_dict(), 'model/model.pth')
        mlflow.pytorch.log_model(self.model, name='model',
                                 registered_model_name=self.args.register or None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url-file', default='../preprocessing/data/url_ids.csv')
    parser.add_argument('--experiment', default='capstone-url-classifier')
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--register', default='', help='모델 레지스트리에 올릴 이름 (비우면 올리지 않음)')
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name):
        classifier = URLClassifier(args.url_file, args)
        classifier.load_and_prepare_data()
        mlflow.log_params({k: v for k, v in vars(args).items() if k not in ('experiment', 'run_name', 'register')})
        mlflow.log_params({
            'device': str(classifier.device),
            'tokenizer_revision': TOKENIZER_REVISION,
            'n_parameters': sum(p.numel() for p in classifier.model.parameters()),
            'train_size': len(classifier.train_dataset),
            'test_size': len(classifier.test_dataset),
            'truncated_urls': classifier.truncated,
            'data_sha256': file_sha256(args.url_file),
        })
        classifier.train_model()
        classifier.log_results()
        classifier.save_model()
        print(json.dumps({k: classifier.final_metrics[k] for k in ('accuracy', 'macro_f1')}))

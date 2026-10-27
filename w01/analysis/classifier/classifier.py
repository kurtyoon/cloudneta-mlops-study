import pandas as pd
import torch
import warnings
import ast
import time

from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertConfig, AdamW
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)


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
    def __init__(self, url_file, feature_file):
        self.url_file = url_file
        self.feature_file = feature_file
        self.device = torch.device("mps")
        self.config = BertConfig.from_pretrained(
            'bert-base-cased', num_labels=4, hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1, hidden_size=768, num_attention_heads=12,
            num_hidden_layers=12, intermediate_size=3072
        )
        self.model = BertForSequenceClassification(self.config)
        self.model.to(self.device)

    def load_and_prepare_data(self):
        url_data = pd.read_csv(self.url_file)
        feature_data = pd.read_csv(self.feature_file)

        # Parse and prepare tensors
        url_data['token_ids'] = url_data['token_ids'].apply(ast.literal_eval)
        feature_data['token_ids'] = feature_data['token_ids'].apply(ast.literal_eval)

        url_data['token_ids'] = url_data['token_ids'].apply(lambda x: [item for sublist in x for item in sublist])
        feature_data['token_ids'] = feature_data['token_ids'].apply(
            lambda x: [item for sublist in x for item in sublist])

        max_length = 512
        # padding with 0 until max_length
        url_data['token_ids'] = url_data['token_ids'].apply(lambda x: x + [0] * (max_length - len(x)))
        feature_data['token_ids'] = feature_data['token_ids'].apply(lambda x: x + [0] * (max_length - len(x)))

        # token_ids 리스트를 텐서로 변환
        url_data['input_ids'] = url_data['token_ids'].apply(lambda ids: torch.tensor(ids, dtype=torch.long))
        feature_data['input_ids'] = feature_data['token_ids'].apply(lambda ids: torch.tensor(ids, dtype=torch.long))

        # 이미 텐서로 변환된 input_ids를 사용하여 attention_mask와 token_type_ids를 생성
        url_data['attention_mask'] = url_data['input_ids'].apply(lambda tensor: torch.ones_like(tensor))
        feature_data['attention_mask'] = feature_data['input_ids'].apply(lambda tensor: torch.ones_like(tensor))

        url_data['token_type_ids'] = url_data['input_ids'].apply(lambda tensor: torch.zeros_like(tensor))
        feature_data['token_type_ids'] = feature_data['input_ids'].apply(lambda tensor: torch.ones_like(tensor))

        # Combine data
        combined_data = pd.concat([url_data, feature_data], ignore_index=True)

        # Convert labels
        labels = torch.tensor(combined_data['label'].values, dtype=torch.long)

        # Split the dataset
        train_idx, test_idx = train_test_split(range(len(combined_data)), test_size=0.2, stratify=labels,
                                               random_state=42)

        train_dataset = combined_data.iloc[train_idx]
        test_dataset = combined_data.iloc[test_idx]

        self.train_dataset = URLDataset(train_dataset, labels[train_idx])
        self.test_dataset = URLDataset(test_dataset, labels[test_idx])

    def train_model(self):
        train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=1e-5)

        self.model.train()
        for epoch in range(10):
            start_time = time.time()
            idx = 0
            for batch in train_loader:
                batch_start_time = time.time() 
                print(f'{idx + 1}번째 batch를 진행하고 있습니다.')

                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                batch_end_time = time.time()
                print(f'배치 처리 시간: {batch_end_time - batch_start_time:.4f}초')

                idx += 1

            end_time = time.time()
            print(f'{epoch + 1}번째 epoch 완료, 소요 시간: {end_time - start_time:.4f}초')

    def evaluate_model(self):
        test_loader = DataLoader(self.test_dataset, batch_size=16, shuffle=False)
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())

    def save_model(self):
        torch.save(self.model.state_dict(), 'model/model.pth')


if __name__ == '__main__':
    classifier = URLClassifier('data/url_ids.csv',
                               'data/feature_ids.csv')
    classifier.load_and_prepare_data()
    classifier.train_model()
    classifier.evaluate_model()
    classifier.save_model()

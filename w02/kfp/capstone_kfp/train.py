"""2단계: 학습. 오프라인 스토어에서 피처를 받아 작은 BERT를 학습하고 MLflow에 남긴다.

analysis/classifier/classifier.py를 그대로 옮기고 데이터를 읽는 부분만 바꿨다.
- 입력: CSV 대신 라벨 parquet을 엔티티로 Feast get_historical_features
- 분할: train_test_split 대신 피처 단계가 적어 둔 split 칸
- 모델 설정, 학습 루프, 지표, MLflow 기록은 같다. 그래야 post-2의 run과 나란히 비교된다.
"""
import os
import time
import warnings

import mlflow
import mlflow.pytorch
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertConfig, BertForSequenceClassification

from capstone_kfp import common, config

warnings.filterwarnings("ignore", category=FutureWarning)

LABELS = config.LABELS
FEATURES = [f'{config.FEATURE_VIEW}:token_ids', f'{config.FEATURE_VIEW}:token_count']


def limit_threads():
    """torch의 스레드 수를 OMP_NUM_THREADS(파이프라인이 CPU 상한과 같은 값으로 준다)에 맞춘다.
    값이 없으면 torch는 노드의 코어 수만큼 스레드를 연다."""
    n = int(os.environ.get('OMP_NUM_THREADS', '0'))
    if n > 0:
        torch.set_num_threads(n)
    return torch.get_num_threads()


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_inputs(token_ids, max_length):
    """classifier.py와 같은 입력: max_length에서 자르고 0으로 채운다.
    (패딩까지 1인 attention_mask는 원본 그대로 둔다. 4주차에 다룬다.)
    추론 단계도 이 함수로 입력을 만든다."""
    ids = [list(x)[:max_length] for x in token_ids]
    input_ids = torch.tensor([x + [0] * (max_length - len(x)) for x in ids], dtype=torch.long)
    return {
        'input_ids': input_ids,
        'attention_mask': torch.ones_like(input_ids),
        'token_type_ids': torch.zeros_like(input_ids),
    }


class URLDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.inputs.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


class URLClassifier:
    def __init__(self, args):
        self.args = args
        self.device = pick_device()
        self.config = BertConfig.from_pretrained(
            config.BERT_DIR, num_labels=4, hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1, hidden_size=args.hidden_size,
            num_attention_heads=args.num_heads, num_hidden_layers=args.num_layers,
            intermediate_size=args.hidden_size * 4,
            max_position_embeddings=max(512, args.max_length)
        )
        self.model = BertForSequenceClassification(self.config)
        self.model.to(self.device)

    def load_and_prepare_data(self):
        # 엔티티 df: 라벨 parquet의 url_id와 event_timestamp. label, split 칸은 결과에 그대로 따라온다.
        entity_df = common.read_parquet(config.LABELS_URI)[['url_id', 'event_timestamp', 'label', 'split']]
        store = common.feature_store()
        data = store.get_historical_features(entity_df=entity_df, features=FEATURES).to_df()
        if data['token_ids'].isna().any():
            raise RuntimeError(f'오프라인 스토어에 없는 url_id {int(data["token_ids"].isna().sum())}개')
        # 조인 결과의 행 순서는 정해져 있지 않다. 같은 run이면 같은 순서가 되도록 정렬한다.
        data = data.sort_values('url_id').reset_index(drop=True)

        self.truncated = int((data['token_count'] > self.args.max_length).sum())
        train_data = data[data['split'] == 'train']
        test_data = data[data['split'] == 'test']
        self.train_dataset = URLDataset(model_inputs(train_data['token_ids'], self.args.max_length),
                                        torch.tensor(train_data['label'].values, dtype=torch.long))
        self.test_dataset = URLDataset(model_inputs(test_data['token_ids'], self.args.max_length),
                                       torch.tensor(test_data['label'].values, dtype=torch.long))

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
                  f'accuracy {metrics["accuracy"]:.4f}, macro F1 {metrics["macro_f1"]:.4f}', flush=True)
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
        return classification_metrics(all_labels, all_preds)

    def log_results(self):
        m = self.final_metrics
        for label, v in m['per_class'].items():
            mlflow.log_metrics({f'{label}_precision': v['precision'], f'{label}_recall': v['recall'],
                                f'{label}_f1': v['f1']})
        mlflow.log_dict({'labels': LABELS, 'rows_true_cols_pred': m['confusion_matrix']},
                        'confusion_matrix.json')
        mlflow.log_dict(m['per_class'], 'per_class_metrics.json')

    def save_model(self):
        # MLflow 3.17은 serialization_format을 주지 않으면 pt2(torch.export)로 저장하려 하고,
        # 그러려면 input_example이 있어야 한다. post-2(mlflow 3.0.0)의 champion v1과 같은
        # pickle로 저장해서, 추론 단계가 어느 버전이 champion이든 같은 방법으로 불러오게 한다.
        # 저장 전에 CPU로 옮긴다. mps에서 학습한 텐서를 그대로 피클하면 CPU Pod에서
        # map_location 없이는 열리지 않는다(post-2의 v1이 그렇다).
        self.model.to('cpu')
        info = mlflow.pytorch.log_model(self.model, name='model', serialization_format='pickle',
                                        registered_model_name=self.args.registered_model)
        return info.registered_model_version


def classification_metrics(y_true, y_pred):
    # classifier.py의 evaluate_model과 같은 지표
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(4), zero_division=0)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'per_class': {LABELS[i]: {'precision': float(precision[i]), 'recall': float(recall[i]),
                                  'f1': float(f1[i]), 'support': int(support[i])} for i in range(4)},
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=range(4)).tolist(),
    }


@common.hides_valkey_password
def run(args, features_sha256, labels_sha256, kfp_run_id):
    """args: num_layers, hidden_size, num_heads, max_length, epochs, batch_size, lr, seed,
    sample_size, n_online, tracking_uri, experiment, registered_model."""
    threads = limit_threads()
    mlflow.set_tracking_uri(args.tracking_uri)
    # 실험이 없으면 서버 기본 위치(mlflow-artifacts:/)로 만든다. s3:// 위치를 주면 클라이언트가
    # S3 키를 직접 써야 하므로 주지 않는다.
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f'kfp-{kfp_run_id[:8]}') as active:
        mlflow.set_tags({'kfp_run_id': kfp_run_id, 'feast_project': 'capstone',
                         'feature_view': config.FEATURE_VIEW})
        classifier = URLClassifier(args)
        classifier.load_and_prepare_data()
        mlflow.log_params({k: getattr(args, k) for k in (
            'num_layers', 'hidden_size', 'num_heads', 'max_length', 'epochs', 'batch_size', 'lr', 'seed',
            'sample_size', 'n_online')})
        mlflow.log_params({
            'device': str(classifier.device),
            'torch_threads': threads,
            'tokenizer_revision': config.TOKENIZER_REVISION,
            'n_parameters': sum(p.numel() for p in classifier.model.parameters()),
            'train_size': len(classifier.train_dataset),
            'test_size': len(classifier.test_dataset),
            'truncated_urls': classifier.truncated,
            'data_sha256': features_sha256,  # 오프라인 스토어 parquet
            'labels_sha256': labels_sha256,
        })
        classifier.train_model()
        classifier.log_results()
        version = classifier.save_model()
        m = classifier.final_metrics
        result = {'run_id': active.info.run_id, 'model_version': str(version),
                  'accuracy': m['accuracy'], 'macro_f1': m['macro_f1'], 'per_class': m['per_class']}
    print({k: result[k] for k in ('run_id', 'model_version', 'accuracy', 'macro_f1')})
    return result


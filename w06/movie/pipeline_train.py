"""책 9.3의 영화 추천 학습 파이프라인을 ml-latest-small과 이 환경에 맞춘 버전.

원본: practical-mlops/movie-recommender-project의 02_training_and_validation.py,
data_components/dataset_preparation.py(qa_data), training_and_validation_components/

흐름: QA → 메타데이터 → negative sampling → test, val 준비 → 학습(MLflow 기록,
TensorBoard 이벤트) → 검증(지표를 학습 run에 기록) → staging 승격 게이트.

원본과 달라진 점:
- 데이터는 movie-data-prep(pipeline_data_prep.py)이 만든 inference-datasets/ml-latest-small/이다.
  코드에 직접 적힌 MinIO 키 대신 Secret kfp-pipeline-s3를 환경 변수로 받는다.
- QA는 원본의 ml-25m 행 수 검사 대신 책 9.3.1 본문대로 세 분할의 컬럼, 타입,
  결측, 평점 범위를 검사한다.
- negative sampling에 시드를 둔다. 원본은 시드가 없어 실행마다 학습 데이터가 다르다.
- 단계 사이의 데이터는 pickle 대신 parquet이다. 데이터 단계(python:3.11)와 학습
  단계(ultralytics 이미지, Python 3.13)의 pandas 버전이 달라도 읽힌다.
- 학습과 검증은 torch가 들어 있는 ultralytics 이미지에서 돌고, mlflow는 단계 안에서
  설치한다. 원본은 단계마다 torch를 새로 설치한다.
- 학습 손실을 텐서가 아니라 float로 더하고 MLflow에 float로 기록한다. 원본은 계산
  그래프가 붙은 텐서를 누적하고 지표를 문자열로 넘긴다.
- 모델을 pickle 형식으로 기록한다. MLflow 3.17은 Databricks 밖에서 pt2(torch.export)가
  기본이고, pt2는 input_example이 필요하며 추론 코드가 읽는 n_items 같은 속성을 남기지 않는다.
- 학습 중 TensorBoard 이벤트를 쓴다(책 9.2). Pod의 /tmp에 쓰고, 끝나면 PVC models의
  tensorboard/<pipeline>/<MLflow run ID>/로 한 번 복사한다.
- 검증: 원본은 torch Dataset을 import하지 않아 kfp.dsl.Dataset을 상속하고,
  설치 목록에 scikit-learn 대신 scikit-metrics를 적었다. 둘 다 고쳤다. 책의 rms(배치
  평균끼리의 RMSE)는 게이트가 쓰므로 그대로 두고, 평점 단위 RMSE를 rmse_rating으로
  함께 기록한다.
- 승격: 원본은 staging이 없을 때 None의 run_id를 먼저 읽어 첫 승격에서 실패하고,
  등록 이름을 인자 대신 "recommender_production"으로 적었다. 둘 다 고쳤다. 판정을
  학습 run의 promotion 태그로 남긴다. 거부해도 단계는 성공으로 끝나는 동작은 원본대로다.

컴파일: python pipeline_train.py 를 실행하면 pipeline_train.yaml이 생긴다.
"""
import os
import sys
from typing import Dict

from kfp import compiler, dsl, kubernetes
from kfp.dsl import Dataset, Input, Output

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'idcard'))
from pipeline import YOLO_IMAGE  # noqa: E402

S3_SECRET = 'kfp-pipeline-s3'
S3_ENV = {'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY'}
MLFLOW_URI = 'http://mlflow.mlflow.svc.cluster.local:5000'


@dsl.component(base_image='python:3.11', packages_to_install=['pyarrow', 'pandas'])
def qa_data(bucket: str, dataset: str):
    import os

    from pyarrow import fs, parquet

    s3 = fs.S3FileSystem(
        endpoint_override='http://minio-service.kubeflow:9000',
        access_key=os.environ['AWS_ACCESS_KEY_ID'],
        secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        scheme='http',
        region='us-east-1',
    )
    print('Running QA')
    for split in ['train', 'val', 'test']:
        df = parquet.read_table(s3.open_input_file(f'{bucket}/{dataset}/{split}.parquet.gzip')).to_pandas()
        print(split, df.shape, dict(df.dtypes.astype(str)))
        assert {'userId', 'movieId', 'rating'} <= set(df.columns), f'Required columns not found. Found {df.columns}'
        assert df['userId'].dtype.kind == 'i' and df['movieId'].dtype.kind == 'i', 'userId and movieId must be integers'
        assert df['rating'].dtype.kind == 'f', 'rating must be float'
        assert not df[['userId', 'movieId', 'rating']].isna().any().any(), 'null values found'
        assert df['rating'].between(0.5, 5.0).all(), 'rating out of 0.5..5.0'
    print('QA passed!')


@dsl.component(base_image='python:3.11', packages_to_install=['pyarrow', 'pandas'])
def get_dataset_metadata(bucket: str, dataset: str) -> Dict[str, int]:
    import os

    from pyarrow import fs, parquet

    s3 = fs.S3FileSystem(
        endpoint_override='http://minio-service.kubeflow:9000',
        access_key=os.environ['AWS_ACCESS_KEY_ID'],
        secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        scheme='http',
        region='us-east-1',
    )
    data_map = {'n_users': 0, 'n_items': 0}
    for split in ['test', 'train', 'val']:
        df = parquet.read_table(s3.open_input_file(f'{bucket}/{dataset}/{split}.parquet.gzip')).to_pandas()
        data_map['n_users'] = max(data_map['n_users'], int(df.userId.max()))
        data_map['n_items'] = max(data_map['n_items'], int(df.movieId.max()))
    print(data_map)
    return data_map


@dsl.component(base_image='python:3.11', packages_to_install=['pyarrow', 'pandas', 'numpy'])
def negative_sampling(num_ng_test: int, bucket: str, dataset: str, split: str, random_state: int,
                      negative_sampled_dataset: Output[Dataset]):
    import os

    import numpy as np
    import pandas as pd
    from pyarrow import fs, parquet

    s3 = fs.S3FileSystem(
        endpoint_override='http://minio-service.kubeflow:9000',
        access_key=os.environ['AWS_ACCESS_KEY_ID'],
        secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        scheme='http',
        region='us-east-1',
    )
    ratings = parquet.read_table(s3.open_input_file(f'{bucket}/{dataset}/{split}.parquet.gzip')).to_pandas()
    rng = np.random.default_rng(random_state)
    item_pool = set(ratings['movieId'].unique())
    interact_status = (
        ratings.groupby('userId')['movieId']
        .apply(set)
        .reset_index()
        .rename(columns={'movieId': 'interacted_items'}))
    interact_status['negative_samples'] = interact_status['interacted_items'].apply(
        lambda x: rng.choice(sorted(item_pool - x), num_ng_test))
    interact_status['rating'] = 0.0
    interact_status['timestamp'] = 1051631039
    interact_status = (interact_status.drop(columns=['interacted_items'])
                       .explode('negative_samples')
                       .rename(columns={'negative_samples': 'movieId'}))
    interact_status['movieId'] = interact_status['movieId'].astype('int64')
    ret = pd.concat([ratings, interact_status], ignore_index=True)
    print(f'before {len(ratings)} after {len(ret)} negatives {len(interact_status)}')
    ret.to_parquet(negative_sampled_dataset.path, index=False)


@dsl.component(base_image='python:3.11', packages_to_install=['pyarrow', 'pandas'])
def get_test_valid_dataset(bucket: str, dataset: str, testing_dataset: Output[Dataset], validation_dataset: Output[Dataset]):
    import os

    from pyarrow import fs, parquet

    s3 = fs.S3FileSystem(
        endpoint_override='http://minio-service.kubeflow:9000',
        access_key=os.environ['AWS_ACCESS_KEY_ID'],
        secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        scheme='http',
        region='us-east-1',
    )
    for split, out in [('test', testing_dataset), ('val', validation_dataset)]:
        df = parquet.read_table(s3.open_input_file(f'{bucket}/{dataset}/{split}.parquet.gzip')).to_pandas()
        df.to_parquet(out.path, index=False)
        print(split, df.shape)


@dsl.component(base_image=YOLO_IMAGE, packages_to_install=['kfp==2.17.0', '--no-deps'])
def train_model(mlflow_experiment_name: str, mlflow_run_id: str, mlflow_tags: dict, mlflow_uri: str,
                hot_reload_model_run_id: str, training_data: Input[Dataset], training_data_metadata: Dict[str, int],
                testing_data: Input[Dataset],
                model_embedding_factors: int, model_learning_rate: float, model_hidden_dims: int, model_dropout_rate: float,
                optimizer_step_size: float, optimizer_gamma: float,
                training_epochs: int,
                train_batch_size: int, test_batch_size: int, shuffle_training_data: bool, shuffle_testing_data: bool,
                tensorboard_dir: str) -> str:
    input_params = {}
    for k, v in locals().items():
        if k == 'input_params':
            continue
        input_params[k] = v

    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-warn-script-location',
                           'mlflow==3.17.2', 'torchinfo==1.8.0', 'tensorboard==2.20.0'])

    import os
    import shutil
    import time

    import mlflow
    import pandas as pd
    import torch
    from mlflow.models import infer_signature
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset as TorchDataset
    from torch.utils.tensorboard import SummaryWriter
    from torchinfo import summary

    class datasetReader(TorchDataset):
        def __init__(self, df, dataset_name):
            self.df = df
            self.name = dataset_name
            print(f"{self.name} : {self.df.shape[0]}")

        def __len__(self):
            return self.df.shape[0]

        def __getitem__(self, idx):
            sd = self.df.iloc[idx]
            user = sd['userId']
            item = sd['movieId']
            rating = sd['rating']
            return torch.tensor(user - 1).long(), torch.tensor(item - 1).long(), torch.tensor(rating).float()

    class MatrixFactorization(torch.nn.Module):
        def __init__(self, n_users, n_items, n_factors, hidden_dim, dropout_rate):
            super().__init__()
            self.n_items = n_items
            self.user_factors = torch.nn.Embedding(n_users + 1, n_factors, sparse=False)
            self.item_factors = torch.nn.Embedding(n_items + 1, n_factors, sparse=False)
            self.linear = torch.nn.Linear(in_features=n_factors, out_features=hidden_dim)
            self.linear2 = torch.nn.Linear(in_features=hidden_dim, out_features=1)
            self.dropout = torch.nn.Dropout(p=dropout_rate)
            self.relu = torch.nn.ReLU()

        def forward(self, user, item):
            user_embedding = self.user_factors(user)
            item_embedding = self.item_factors(item)
            embeddding_vector = torch.mul(user_embedding, item_embedding)
            x = self.relu(self.linear(embeddding_vector))
            x = self.dropout(x)
            rating = self.linear2(x)
            return rating

    train_dataset = datasetReader(pd.read_parquet(training_data.path), dataset_name='train')
    test_dataset = datasetReader(pd.read_parquet(testing_data.path), dataset_name='test')

    n_users = training_data_metadata['n_users']
    n_items = training_data_metadata['n_items']

    if hot_reload_model_run_id == 'none':
        hot_reload_model_run_id = None

    mlflow.set_tracking_uri(uri=mlflow_uri)
    if hot_reload_model_run_id:
        model = mlflow.pytorch.load_model(f"runs:/{hot_reload_model_run_id}/model")
    else:
        model = MatrixFactorization(n_users, n_items, n_factors=model_embedding_factors, hidden_dim=model_hidden_dims,
                                    dropout_rate=model_dropout_rate)

    optimizer = torch.optim.SGD(model.parameters(), lr=model_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optimizer_step_size, gamma=optimizer_gamma)
    loss_func = torch.nn.L1Loss()
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle_training_data)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=shuffle_testing_data)

    mlflow.set_experiment(mlflow_experiment_name)
    if mlflow_run_id == "":
        mlflow_run_id = None

    with mlflow.start_run(run_id=mlflow_run_id) as run:
        current_run_id = run.info.run_id
        local_tb = os.path.join('/tmp/tensorboard', current_run_id)
        writer = SummaryWriter(log_dir=local_tb)
        for k, v in input_params.items():
            if 'mlflow_' not in k:
                mlflow.log_param(k, v)
        mlflow.log_param("loss_function", loss_func.__class__.__name__)
        mlflow.log_param("optimizer", "SGD")
        mlflow.log_params({'n_user': n_users, 'n_items': n_items})
        for k, v in mlflow_tags.items():
            mlflow.set_tag(k, v)

        with open("model_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")

        model_signature = None
        t_start = time.monotonic()
        for train_iter in range(training_epochs):
            print(train_iter)
            epoch_start = time.monotonic()
            model.train()
            t_loss = 0.0
            t_count = 0
            for row, col, rating in train_dataloader:
                prediction = model(row, col)
                if model_signature is None:
                    model_signature = infer_signature(
                        {'user': row.cpu().detach().numpy(), 'movie': col.cpu().detach().numpy()},
                        prediction.cpu().detach().numpy())
                loss = loss_func(prediction, rating.unsqueeze(1))
                t_loss += loss.item()
                t_count += 1
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            mlflow.log_metric("avg_training_loss", t_loss / t_count, step=train_iter)
            writer.add_scalar('loss/train', t_loss / t_count, train_iter)
            writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], train_iter)
            scheduler.step()
            model.eval()
            te_loss = 0.0
            te_count = 0
            print('Evaluating')
            with torch.no_grad():
                for row, col, rating in test_dataloader:
                    prediction = model(row, col)
                    loss = loss_func(prediction, rating.unsqueeze(1))
                    te_loss += loss.item()
                    te_count += 1
            mlflow.log_metric("avg_testing_loss", te_loss / te_count, step=train_iter)
            writer.add_scalar('loss/test', te_loss / te_count, train_iter)
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.detach(), train_iter)
            print(f"Test loss: {te_loss / te_count}")
            print(f"Train loss: {t_loss / t_count}")
            print(f"TIMING epoch {train_iter} {time.monotonic() - epoch_start:.1f}s")

        print(f"TIMING train total {time.monotonic() - t_start:.1f}s")
        writer.close()
        # MLflow 3.17 saves as pt2 (torch.export) by default outside Databricks; that needs an
        # input_example and keeps only the traced graph, not attributes such as n_items that
        # 9.17 inference reads. pickle keeps the module as the book loads it.
        mlflow.pytorch.log_model(model, name="model", signature=model_signature, serialization_format="pickle")

    target = os.path.join('/models', tensorboard_dir, current_run_id)
    shutil.copytree(local_tb, target)
    print(f'tensorboard events copied to {target}')
    return current_run_id


@dsl.component(base_image=YOLO_IMAGE, packages_to_install=['kfp==2.17.0', '--no-deps'])
def validate_model(model_run_id: str, top_k: int, threshold: int, val_batch_size: int, mlflow_uri: str,
                   validation_dataset: Input[Dataset]):
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-warn-script-location',
                           'mlflow==3.17.2'])

    # https://pureai.substack.com/p/recommender-systems-with-pytorch
    from collections import defaultdict

    import mlflow
    import mlflow.pytorch
    import pandas as pd
    import torch
    from sklearn.metrics import root_mean_squared_error
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset as TorchDataset

    mlflow.set_tracking_uri(uri=mlflow_uri)
    recommendation_model = mlflow.pytorch.load_model(f"runs:/{model_run_id}/model")

    class datasetReader(TorchDataset):
        def __init__(self, df, dataset_name):
            super().__init__()
            self.df = df
            self.name = dataset_name
            print(f"{self.name} : {self.df.shape[0]}")

        def __len__(self):
            return self.df.shape[0]

        def __getitem__(self, idx):
            sd = self.df.iloc[idx]
            return (torch.tensor(sd['userId'] - 1).long(), torch.tensor(sd['movieId'] - 1).long(),
                    torch.tensor(sd['rating']).float())

    def calculate_precision_recall(user_ratings, k, threshold):
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum(true_r >= threshold for _, true_r in user_ratings)
        n_rec_k = sum(est >= threshold for est, _ in user_ratings[:k])
        n_rel_and_rec_k = sum((true_r >= threshold) and (est >= threshold) for est, true_r in user_ratings[:k])
        precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
        return precision, recall

    user_ratings_comparison = defaultdict(list)
    val_data = datasetReader(pd.read_parquet(validation_dataset.path), dataset_name='val')
    val_dataloader = DataLoader(val_data, batch_size=val_batch_size, shuffle=True)

    y_pred = []
    y_true = []
    all_pred = []
    all_true = []
    recommendation_model.eval()
    with torch.no_grad():
        for users, movies, ratings in val_dataloader:
            output = recommendation_model(users, movies)
            y_pred.append(output.sum().item() / len(users))
            y_true.append(ratings.sum().item() / len(users))
            all_pred.extend(output[:, 0].tolist())
            all_true.extend(ratings.tolist())
            for user, pred, true in zip(users, output, ratings):
                user_ratings_comparison[user.item()].append((pred[0].item(), true.item()))

    user_precisions = dict()
    user_based_recalls = dict()
    k = top_k
    for user_id, user_ratings in user_ratings_comparison.items():
        precision, recall = calculate_precision_recall(user_ratings, k, threshold)
        user_precisions[user_id] = precision
        user_based_recalls[user_id] = recall

    average_precision = sum(prec for prec in user_precisions.values()) / len(user_precisions)
    average_recall = sum(rec for rec in user_based_recalls.values()) / len(user_based_recalls)
    rms = root_mean_squared_error(y_true, y_pred)
    rmse_rating = root_mean_squared_error(all_true, all_pred)

    print(f"precision_{k}: {average_precision:.4f}")
    print(f"recall_{k}: {average_recall:.4f}")
    print(f"rms: {rms:.4f}")
    print(f"rmse_rating: {rmse_rating:.4f}")
    mlflow.log_metric(f"precision_{k}", average_precision, run_id=model_run_id)
    mlflow.log_metric(f"recall_{k}", average_recall, run_id=model_run_id)
    mlflow.log_metric("rms", rms, run_id=model_run_id)
    mlflow.log_metric("rmse_rating", rmse_rating, run_id=model_run_id)


@dsl.component(base_image='python:3.11', packages_to_install=['mlflow==3.17.2'])
def promote_model_to_staging(model_run_id: str, registered_model_name: str, rms_threshold: float,
                             precision_threshold: float, top_k: int, recall_threshold: float, mlflow_uri: str):
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    mlflow.set_tracking_uri(uri=mlflow_uri)
    client = MlflowClient()

    def decide(text):
        print(text)
        client.set_tag(model_run_id, 'promotion', text)

    current_staging = None
    try:
        current_staging = client.get_model_version_by_alias(registered_model_name, "staging")
    except MlflowException:
        # RestException from a tracking server, MlflowException from a local store.
        print("No staging model found. Auto upgrade current run to staging.")

    if current_staging is not None:
        if current_staging.run_id == model_run_id:
            decide("already staging")
            return
        staging_metrics = client.get_run(current_staging.run_id).data.to_dictionary()['metrics']
        new_metrics = client.get_run(model_run_id).data.to_dictionary()['metrics']
        rms_diff = new_metrics['rms'] - staging_metrics['rms']
        prec_diff = new_metrics[f'precision_{top_k}'] - staging_metrics[f'precision_{top_k}']
        rec_diff = new_metrics[f'recall_{top_k}'] - staging_metrics[f'recall_{top_k}']
        print(f'against staging v{current_staging.version}: rms {rms_diff:+.4f}, precision {prec_diff:+.4f}, recall {rec_diff:+.4f}')
        if rms_diff > rms_threshold:
            decide(f"rejected: rms {rms_diff:+.4f} > {rms_threshold}")
            return
        if prec_diff < precision_threshold:
            decide(f"rejected: precision {prec_diff:+.4f} < {precision_threshold}")
            return
        if rec_diff < recall_threshold:
            decide(f"rejected: recall {rec_diff:+.4f} < {recall_threshold}")
            return

    result = mlflow.register_model(f"runs:/{model_run_id}/model", registered_model_name)
    client.set_registered_model_alias(registered_model_name, "staging", result.version)
    decide(f"promoted: {registered_model_name} v{result.version} staging")


@dsl.pipeline(name='movie-train', description='MovieLens ml-latest-small 행렬 분해 학습, MLflow 기록, 검증, staging 승격 (책 9.3)')
def training_pipeline(
        bucket: str = 'inference-datasets',
        dataset: str = 'ml-latest-small',
        number_of_negative_samples: int = 10,
        random_state: int = 42,
        training_batch_size: int = 64,
        training_learning_rate: float = 0.001,
        model_embedding_factors: int = 20,
        model_hidden_dims: int = 256,
        training_epochs: int = 30,
        optimizer_step_size: float = 10.0,
        optimizer_gamma: float = 0.1,
        model_dropout_rate: float = 0.2,
        testing_batch_size: int = 64,
        shuffle_training_data: bool = True,
        shuffle_testing_data: bool = True,
        hot_reload_model_id: str = 'none',
        validation_top_k: int = 50,
        validation_threshold: int = 3,
        validation_batch_size: int = 32,
        model_promote_rms_threshold: float = 0.0001,
        model_promote_precision_threshold: float = -0.3,
        model_promote_recall_threshold: float = -0.2,
        mlflow_experiment_name: str = 'recommender',
        mlflow_registered_model_name: str = 'recommender_production',
        mlflow_uri: str = MLFLOW_URI,
        tensorboard_dir: str = 'tensorboard/movie-train'):

    qa_op = qa_data(bucket=bucket, dataset=dataset).set_display_name('qa-training-data')
    dataset_metadata = get_dataset_metadata(bucket=bucket, dataset=dataset).set_display_name('metadata').after(qa_op)
    negative_sampled_data = negative_sampling(
        num_ng_test=number_of_negative_samples, bucket=bucket, dataset=dataset, split='train',
        random_state=random_state).set_display_name('negative-sampling').after(dataset_metadata)
    aux_data = get_test_valid_dataset(bucket=bucket, dataset=dataset).set_display_name('test-valid').after(negative_sampled_data)
    for op in [qa_op, dataset_metadata, negative_sampled_data, aux_data]:
        kubernetes.use_secret_as_env(op, secret_name=S3_SECRET, secret_key_to_env=S3_ENV)

    training = train_model(
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_run_id="",
        mlflow_tags={'pipeline': 'movie-train', 'dataset': dataset},
        mlflow_uri=mlflow_uri,
        hot_reload_model_run_id=hot_reload_model_id,
        training_data=negative_sampled_data.outputs['negative_sampled_dataset'],
        training_data_metadata=dataset_metadata.output,
        testing_data=aux_data.outputs['testing_dataset'],
        model_embedding_factors=model_embedding_factors,
        model_learning_rate=training_learning_rate,
        model_hidden_dims=model_hidden_dims,
        model_dropout_rate=model_dropout_rate,
        optimizer_step_size=optimizer_step_size,
        optimizer_gamma=optimizer_gamma,
        training_epochs=training_epochs,
        train_batch_size=training_batch_size,
        test_batch_size=testing_batch_size,
        shuffle_training_data=shuffle_training_data,
        shuffle_testing_data=shuffle_testing_data,
        tensorboard_dir=tensorboard_dir).set_display_name('train').after(negative_sampled_data)
    training.set_cpu_request('1').set_cpu_limit('2').set_memory_request('1Gi').set_memory_limit('4Gi')
    training.set_env_variable('OMP_NUM_THREADS', '2')
    kubernetes.mount_pvc(training, pvc_name='models', mount_path='/models')

    val = validate_model(
        model_run_id=training.output,
        top_k=validation_top_k,
        threshold=validation_threshold,
        val_batch_size=validation_batch_size,
        mlflow_uri=mlflow_uri,
        validation_dataset=aux_data.outputs['validation_dataset']).set_display_name('validate').after(training)
    val.set_cpu_request('500m').set_cpu_limit('1').set_memory_request('1Gi').set_memory_limit('3Gi')
    val.set_env_variable('OMP_NUM_THREADS', '1')

    promote = promote_model_to_staging(
        model_run_id=training.output,
        registered_model_name=mlflow_registered_model_name,
        top_k=validation_top_k,
        rms_threshold=model_promote_rms_threshold,
        precision_threshold=model_promote_precision_threshold,
        recall_threshold=model_promote_recall_threshold,
        mlflow_uri=mlflow_uri).set_display_name('promote-staging').after(val)

    for op in [qa_op, dataset_metadata, negative_sampled_data, aux_data, training, val, promote]:
        op.set_caching_options(False)


if __name__ == '__main__':
    compiler.Compiler().compile(pipeline_func=training_pipeline, package_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline_train.yaml'))

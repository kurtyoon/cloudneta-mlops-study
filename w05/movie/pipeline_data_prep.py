"""저자의 영화 추천 데이터 준비 파이프라인을 ml-latest-small과 이 환경의 S3 권한에 맞춘 버전.

원본: practical-mlops/movie-recommender-project의 01_data_preparation_pipeline.py, data_components/dataset_preparation.py

원본과 달라진 점:
- ml-25m 대신 ml-latest-small을 받는다. 주소와 zip 내부 경로, QA의 기대 행 수가 바뀐다.
- 코드에 직접 적힌 접근 키를 지우고 Secret kfp-pipeline-s3를 환경 변수로 받는다.
- 이 키에는 버킷 생성 권한이 없어 create_bucket을 지웠다. 기존 버킷 inference-datasets의 ml-latest-small/ 아래에 쓴다.
- 쓰이지 않고 실제 컬럼명과도 다른 user_id, item_id 참조를 지웠다. 컬럼은 userId, movieId다.
- 다운로드의 verify=False를 지우고 상태 코드를 확인한다.
- split 결과를 index=False로 쓴다. 뒤 장 코드는 userId, movieId만 쓴다.
"""
from kfp import compiler, dsl, kubernetes
from kfp.dsl import Artifact, Input, Output

S3_SECRET = 'kfp-pipeline-s3'
S3_ENV = {'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY'}


@dsl.component(base_image='python:3.11', packages_to_install=['requests'])
def download_data(output_path_one: Output[Artifact]):
    import requests

    url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(output_path_one.path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                file.write(chunk)


@dsl.component(base_image='python:3.11')
def unzip_data(input_path: Input[Artifact], ratings_output_path: Output[Artifact], movies_output_path: Output[Artifact]):
    import zipfile

    with zipfile.ZipFile(input_path.path, 'r') as z:
        with open(ratings_output_path.path, 'wb') as f:
            f.write(z.read('ml-latest-small/ratings.csv'))
        with open(movies_output_path.path, 'wb') as f:
            f.write(z.read('ml-latest-small/movies.csv'))


@dsl.component(base_image='python:3.11', packages_to_install=['pandas', 'fastparquet'])
def csv_to_parquet(inputFile: Input[Artifact], output_path: Output[Artifact]):
    import pandas as pd

    df = pd.read_csv(inputFile.path, index_col=False)
    df.to_parquet(output_path.path, compression='gzip')


@dsl.component(base_image='python:3.11', packages_to_install=['scikit-learn', 'pandas', 'fastparquet'])
def split_dataset(input_parquet: Input[Artifact], dataset_path: Output[Artifact], random_state: int = 42):
    import os

    import pandas as pd
    from sklearn.model_selection import train_test_split

    train_ratio, validation_ratio, test_ratio = 0.75, 0.15, 0.10
    ratings_df = pd.read_parquet(input_parquet.path)

    train, rest = train_test_split(ratings_df, test_size=1 - train_ratio, random_state=random_state)
    val, test = train_test_split(
        rest, test_size=test_ratio / (test_ratio + validation_ratio), random_state=random_state
    )

    os.mkdir(dataset_path.path)
    for name, frame in [('train', train), ('val', val), ('test', test)]:
        frame.to_parquet(os.path.join(dataset_path.path, f'{name}.parquet.gzip'), compression='gzip', index=False)
        print(f'{name}: {len(frame)}')


# boto3 1.36부터 기본 체크섬 방식이 바뀌었다. URL 모델에서 SeaweedFS와 동작을 확인한 버전으로 고정한다.
@dsl.component(base_image='python:3.11', packages_to_install=['boto3==1.35.36'])
def put_to_s3(inputFile: Input[Artifact], bucket: str, dataset: str, upload_file_name: str = ''):
    import os

    import boto3

    client = boto3.client('s3', endpoint_url='http://minio-service.kubeflow:9000')
    if os.path.isdir(inputFile.path):
        for file in os.listdir(inputFile.path):
            client.upload_file(os.path.join(inputFile.path, file), bucket, f'{dataset}/{file}')
            print(f'uploaded {bucket}/{dataset}/{file}')
    else:
        file = upload_file_name or os.path.basename(inputFile.path)
        client.upload_file(inputFile.path, bucket, f'{dataset}/{file}')
        print(f'uploaded {bucket}/{dataset}/{file}')


@dsl.component(base_image='python:3.11', packages_to_install=['pyarrow', 'pandas'])
def qa_data(bucket: str, dataset: str, total_rows: int = 100836):
    import os

    from pyarrow import fs, parquet

    s3 = fs.S3FileSystem(
        endpoint_override='http://minio-service.kubeflow:9000',
        access_key=os.environ['AWS_ACCESS_KEY_ID'],
        secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        scheme='http',
        region='us-east-1',
    )
    df = parquet.read_table(s3.open_input_file(f'{bucket}/{dataset}/train.parquet.gzip')).to_pandas()
    print(f'train shape {df.shape}, columns {list(df.columns)}')
    assert df.shape[1] == 4
    assert df.shape[0] >= 0.75 * total_rows
    assert {'userId', 'movieId'} <= set(df.columns)
    print('QA passed')


@dsl.pipeline(name='movie-data-prep', description='MovieLens ml-latest-small 데이터 준비')
def pipeline(bucket: str = 'inference-datasets', dataset: str = 'ml-latest-small', random_init: int = 42):
    download_op = download_data().set_display_name('download')
    unzip_op = unzip_data(input_path=download_op.outputs['output_path_one']).set_display_name('unzip')
    ratings_op = csv_to_parquet(inputFile=unzip_op.outputs['ratings_output_path']).set_display_name('ratings-parquet')
    movies_op = csv_to_parquet(inputFile=unzip_op.outputs['movies_output_path']).set_display_name('movies-parquet')
    split_op = split_dataset(input_parquet=ratings_op.output, random_state=random_init).set_display_name('split')

    put_movies = put_to_s3(
        inputFile=movies_op.output, bucket=bucket, dataset=dataset, upload_file_name='movies.parquet.gzip'
    ).set_display_name('put-movies')
    put_split = put_to_s3(inputFile=split_op.output, bucket=bucket, dataset=dataset).set_display_name('put-split')
    qa_op = qa_data(bucket=bucket, dataset=dataset).after(put_split).set_display_name('qa')

    for task in [put_movies, put_split, qa_op]:
        kubernetes.use_secret_as_env(task, secret_name=S3_SECRET, secret_key_to_env=S3_ENV)
    for task in [download_op, unzip_op, ratings_op, movies_op, split_op, put_movies, put_split, qa_op]:
        task.set_caching_options(False)


if __name__ == '__main__':
    compiler.Compiler().compile(pipeline_func=pipeline, package_path='pipeline.yaml')

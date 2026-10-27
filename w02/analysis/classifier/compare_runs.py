import argparse

import mlflow

LABELS = ['benign', 'phishing', 'defacement', 'malware']  # classifier.py와 같은 순서


if __name__ == '__main__':
    # MLflow에 남은 run들을 파라미터와 지표로 나란히 놓는다.
    # 접속 정보는 MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD 환경 변수로 받는다.
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='capstone-url-classifier')
    args = parser.parse_args()

    runs = mlflow.search_runs(experiment_names=[args.experiment], order_by=['start_time ASC'])
    runs = runs[runs['status'] == 'FINISHED']

    columns = {'tags.mlflow.runName': 'run', 'params.lr': 'lr', 'params.max_length': 'max_len',
               'metrics.test_accuracy': 'accuracy', 'metrics.test_macro_f1': 'macro_f1'}
    for label in LABELS:
        columns[f'metrics.{label}_recall'] = f'{label}_recall'
    table = runs[list(columns)].rename(columns=columns)
    print(table.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

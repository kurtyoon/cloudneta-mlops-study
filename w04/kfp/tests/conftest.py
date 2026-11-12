"""단위 테스트는 패키지와 토크나이저가 든 컴포넌트 이미지 안에서 pytest만 더 깔아 돌린다.

    docker run --rm -v "$PWD:/src" -w /src -e PYTHONPATH=/src capstone-kfp:dev \
        sh -c 'pip install --quiet pytest==8.3.4 && python -m pytest -q tests'
"""
import pandas as pd
import pytest


class CharTokenizer:
    """BertTokenizer 대신 쓰는 토크나이저: 글자 하나가 토큰 하나, id는 코드 포인트."""

    def tokenize(self, text):
        return list(text)

    def convert_tokens_to_ids(self, tokens):
        return [ord(t) for t in tokens]


@pytest.fixture
def tokenizer():
    return CharTokenizer()


@pytest.fixture
def raw():
    """클래스 비율이 원본과 비슷한 400행."""
    counts = {'benign': 264, 'phishing': 60, 'defacement': 56, 'malware': 20}
    rows = [(f'http://{t}-{i}.example/path/{i}', t) for t, n in counts.items() for i in range(n)]
    return pd.DataFrame(rows, columns=['url', 'type'])

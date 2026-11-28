"""tests/data/url_stats_sample.csv를 만든다: features.py와 같은 방법으로 뽑은 표본의 split, type, url_stats.
tests/test_url_drift.py가 train 행을 reference로, test 행을 요청으로 쓴다. 원본 CSV는 저장소에 없다:

    docker run --rm -v "$PWD:/src" -w /src -e PYTHONPATH=/src -v <원본 CSV 디렉터리>:/data:ro capstone-kfp:dev \
        python tests/data/make_url_stats_sample.py /data/malicious_phish.csv
"""
import sys
from pathlib import Path

import pandas as pd

from capstone_kfp import config, features, url_drift

SAMPLE_SIZE = 6000
SEED = 42
OUT = Path(__file__).with_name('url_stats_sample.csv')


def main(raw_csv):
    deduped, _ = features.dedupe(pd.read_csv(raw_csv))
    sample = features.stratified_sample(deduped, SAMPLE_SIZE, SEED).reset_index(drop=True)
    split = features.assign_split(sample['type'].map(config.LABELS.index).to_numpy(), SEED)
    out = pd.concat([pd.DataFrame({'split': split, 'type': sample['type']}), url_drift.url_stats(sample['url'])],
                    axis=1)
    out.to_csv(OUT, index=False)
    print(OUT, len(out), out.groupby(['split', 'type']).size().to_dict())


if __name__ == '__main__':
    main(sys.argv[1])

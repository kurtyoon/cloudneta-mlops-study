"""캡스톤 URL 분류기에 정상 구간과 드리프트 구간의 요청을 차례로 보낸다(6주차 모니터링 실습).

- 정상 구간: malicious_phish.csv에서 클래스 비율을 지켜 뽑은 URL. 학습 데이터와 같은 분포라 window가
  드리프트로 판정되지 않아야 한다(11-05 실험: 학습 split과 test split의 window 100개 중 0개).
- 드리프트 구간: 한 클래스(기본 defacement)의 URL만. 공격 묶음이 한꺼번에 들어오는 상황이다. 11-05 실험에서
  phishing만, defacement만으로 된 window는 20개 중 20개가 드리프트로 잡혔다.

predictor는 serving namespace 안에서만 열려 있다. 노드를 거치는 SSH 터널로 부른다(NetworkPolicy가 노드를
받아들인다):

    ssh -N -L 3010:<capstone-url-classifier-predictor의 ClusterIP>:80 <노드>
    python3 send_urls.py --base http://127.0.0.1:3010 --normal 300 --drift 200

표준 라이브러리만 쓴다. 요청은 순서대로 하나씩 보내고(--rate로 초당 수 제한), 구간마다 예측 라벨 분포를 찍는다.
"""
import argparse
import collections
import csv
import json
import random
import time
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'analysis/preprocessing/data/malicious_phish.csv'


def load(path):
    by_type = collections.defaultdict(list)
    with open(path, newline='', encoding='utf-8') as file:
        for row in csv.DictReader(file):
            if row['url']:
                by_type[row['type']].append(row['url'])
    return by_type


def stratified(by_type, n, rng):
    total = sum(len(v) for v in by_type.values())
    # 클래스마다 내림한 몫을 주고, 모자란 수는 나머지가 큰 클래스부터 하나씩 더한다(합이 정확히 n).
    shares = {k: n * len(v) / total for k, v in by_type.items()}
    take = {k: int(s) for k, s in shares.items()}
    for k in sorted(shares, key=lambda k: shares[k] - take[k], reverse=True)[:n - sum(take.values())]:
        take[k] += 1
    urls = [u for k, values in by_type.items() for u in rng.sample(values, take[k])]
    rng.shuffle(urls)
    return urls


def send(base, urls, rate, label):
    counts, errors = collections.Counter(), 0
    gap = 1 / rate if rate > 0 else 0
    t0 = time.monotonic()
    for i, url in enumerate(urls, 1):
        req = urllib.request.Request(f'{base}/api/predict', data=json.dumps({'url': url}).encode(),
                                     headers={'content-type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                counts[json.load(response)['prediction']] += 1
        except OSError as error:
            errors += 1
            print(f'{label} {i}: {error}', flush=True)
        if gap:
            time.sleep(max(0.0, t0 + i * gap - time.monotonic()))
    took = time.monotonic() - t0
    print(f'{label}: {len(urls)}건 {took:.0f}초, 오류 {errors}, 예측 {dict(counts)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', default='http://127.0.0.1:3010')
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--normal', type=int, default=300, help='정상 구간 요청 수(window 100의 배수가 알기 쉽다)')
    parser.add_argument('--drift', type=int, default=200, help='드리프트 구간 요청 수')
    parser.add_argument('--drift-type', default='defacement', help='드리프트 구간에 보낼 클래스(benign, phishing, defacement, malware)')
    parser.add_argument('--rate', type=float, default=5, help='초당 요청 수 상한')
    parser.add_argument('--seed', type=int, default=20261125)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    by_type = load(args.data)
    print('클래스별 URL 수:', {k: len(v) for k, v in by_type.items()}, flush=True)
    if args.normal:
        send(args.base, stratified(by_type, args.normal, rng), args.rate, '정상 구간')
    if args.drift:
        send(args.base, rng.sample(by_type[args.drift_type], args.drift), args.rate, f'드리프트 구간({args.drift_type})')


if __name__ == '__main__':
    main()

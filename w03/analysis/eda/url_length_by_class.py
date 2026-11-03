import argparse

import matplotlib
import pandas as pd
import seaborn.objects as so

matplotlib.use('Agg')

ORDER = ['benign', 'defacement', 'phishing', 'malware']
BINS = [0, 25, 50, 75, 100, 150, 10_000]
BIN_LABELS = ['1-25', '26-50', '51-75', '76-100', '101-150', '151+']


def load(path):
    df = pd.read_csv(path)
    df['length'] = pd.cut(df['url'].str.len(), bins=BINS, labels=BIN_LABELS)
    df['type'] = pd.Categorical(df['type'], categories=ORDER)
    return df


def stacked(df):
    return (
        so.Plot(df, x='length', color='type')
        .add(so.Bar(), so.Count(), so.Stack())
        .scale(color=so.Nominal(order=ORDER))
        .label(x='URL length (characters)', y='count', color='type',
               title='Class counts per URL length (stack)')
    )


def filled(df):
    # 비율은 pandas로 계산한다. seaborn 0.13.2의 so.Hist(common_norm=['x'])는
    # 'x'를 그룹 변수로 받지 않는다는 경고를 낸다.
    shares = (df.groupby(['length', 'type'], observed=False).size()
              .groupby(level='length', observed=False).transform(lambda n: n / n.sum())
              .rename('share').reset_index())
    return (
        so.Plot(shares, x='length', y='share', color='type')
        .add(so.Bar(), so.Stack())
        .scale(color=so.Nominal(order=ORDER))
        .label(x='URL length (characters)', y='share within length', color='type',
               title='Class share per URL length (fill)')
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../preprocessing/data/malicious_phish.csv')
    parser.add_argument('--out', default='.')
    args = parser.parse_args()

    df = load(args.data)
    print(pd.crosstab(df['length'], df['type'], margins=True))
    print(pd.crosstab(df['length'], df['type'], normalize='index').round(3))
    stacked(df).layout(size=(7, 4)).save(f'{args.out}/url-length-stack.png', dpi=150, bbox_inches='tight')
    filled(df).layout(size=(7, 4)).save(f'{args.out}/url-length-fill.png', dpi=150, bbox_inches='tight')

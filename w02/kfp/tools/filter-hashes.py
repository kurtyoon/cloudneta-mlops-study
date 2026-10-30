#!/usr/bin/env python3
"""`uv pip compile --generate-hashes` 결과에서 linux aarch64 CPython 3.11 이미지가 설치할 파일의 해시만 남긴다.

파일 이름과 sha256은 PyPI JSON API에서 받는다.

사용법: tools/filter-hashes.py <해시가 든 lock> > requirements.lock
"""
import json
import re
import sys
import urllib.request


def usable_wheel(filename):
    if not filename.endswith('.whl'):
        return False
    pytag, abitag, plat = filename[:-4].split('-')[-3:]
    pys = pytag.split('.')
    python_ok = ('cp311' in pys or any(t in ('py3', 'py311') for t in pys)
                 or (abitag == 'abi3' and any(t.startswith('cp3') and int(t[3:] or 0) <= 11 for t in pys)))
    plat_ok = any(t == 'any' or (t.startswith(('manylinux', 'linux')) and t.endswith('aarch64'))
                  for t in plat.split('.'))
    return python_ok and plat_ok


def files(name, version):
    with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json', timeout=30) as r:
        data = json.load(r)
    return {u['filename']: u['digests']['sha256'] for u in data['urls']}


text = open(sys.argv[1]).read()
# 패키지 하나가 한 덩어리: "name==version [; marker] \" 다음 줄부터 해시와 주석.
blocks = re.split(r'\n(?=\S)', text.strip())
out, missing = [], []
for block in blocks:
    head = block.splitlines()[0].rstrip(' \\')
    m = re.match(r'([A-Za-z0-9_.\-\[\]]+)==([^ ;]+)', head)
    if not m:
        out.append(block)
        continue
    name, version = re.sub(r'\[.*\]', '', m.group(1)), m.group(2)
    listed = set(re.findall(r'sha256:([0-9a-f]{64})', block))
    available = {f: h for f, h in files(name, version).items() if h in listed}
    keep = sorted(h for f, h in available.items() if usable_wheel(f))
    if not keep:
        # wheel이 없으면 sdist 해시를 남긴다.
        keep = sorted(h for f, h in available.items() if f.endswith('.tar.gz'))
    if not keep:
        missing.append(f'{name}=={version}')
        continue
    via = [line for line in block.splitlines()[1:] if line.lstrip().startswith('#')]
    out.append(' \\\n'.join([head] + [f'    --hash=sha256:{h}' for h in keep]) + ''.join('\n' + v for v in via))
if missing:
    sys.exit('쓸 수 있는 파일이 없다: ' + ', '.join(missing))
print('\n'.join(out))

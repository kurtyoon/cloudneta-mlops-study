#!/usr/bin/env python3
"""Writes a compiled KFP pipeline as a Pipeline and a PipelineVersion object.

    uv run --with pyyaml render.py <compiled.yaml> <pipeline-name> [--display-name TEXT] [--source TEXT] [--no-cache]

writes pipelines/<pipeline-name>.yaml next to this script. The compiled file
is what `kfp.compiler.Compiler().compile()` writes: the pipeline spec, and a
second document with the platform spec when the pipeline uses kfp-kubernetes
(PVCs, Secrets as env). The PipelineVersion is named <pipeline-name>-<hash>,
the first 10 hex digits of the sha256 of both specs, because KFP refuses any
change to a version's spec (its validating webhook): a new compile with a
different spec becomes a new version, and the same spec keeps its name, so
re-rendering an unchanged pipeline changes nothing. The file holds only the
current version; Argo CD prunes the previous one, and runs made from it keep
their own copy of the spec in MySQL.

--no-cache turns KFP's execution cache off for every task (cachingOptions in
the spec), for pipelines that are run again with the same inputs to measure
them; the SDK this cluster's labs compile with (kfp 2.9.0) has no switch for
it at compile time.
"""
import argparse
import hashlib
import json
import os
import re
import sys

import yaml

parser = argparse.ArgumentParser()
parser.add_argument('compiled')
parser.add_argument('name')
parser.add_argument('--display-name')
parser.add_argument('--source', help='where the pipeline code lives, recorded in the file header')
parser.add_argument('--no-cache', action='store_true', help='turn the execution cache off for every task')
args = parser.parse_args()

if not re.fullmatch(r'[a-z0-9]([a-z0-9-]{0,50}[a-z0-9])?', args.name):
    sys.exit(f'pipeline name must be lowercase letters, digits and hyphens: {args.name}')

with open(args.compiled) as f:
    docs = [d for d in yaml.safe_load_all(f) if d]
pipeline_spec = docs[0]
platform_spec = docs[1] if len(docs) > 1 else None
if 'pipelineInfo' not in pipeline_spec:
    sys.exit(f'{args.compiled}: the first document is not a pipeline spec')

if args.no_cache:
    def tasks(dag):
        for task in dag.get('tasks', {}).values():
            yield task
    count = 0
    for holder in [pipeline_spec.get('root', {})] + list(pipeline_spec.get('components', {}).values()):
        for task in tasks(holder.get('dag', {})):
            task.setdefault('cachingOptions', {})['enableCache'] = False
            count += 1
    print(f'cache off for {count} tasks')

digest = hashlib.sha256(json.dumps([pipeline_spec, platform_spec], sort_keys=True).encode()).hexdigest()[:10]
version = f'{args.name}-{digest}'
display = args.display_name or pipeline_spec['pipelineInfo'].get('displayName') or pipeline_spec['pipelineInfo']['name']

pipeline = {
    'apiVersion': 'pipelines.kubeflow.org/v2beta1',
    'kind': 'Pipeline',
    'metadata': {'name': args.name, 'annotations': {'argocd.argoproj.io/sync-wave': '-1'}},
    'spec': {'displayName': display, 'description': pipeline_spec['pipelineInfo'].get('description', '')},
}
version_obj = {
    'apiVersion': 'pipelines.kubeflow.org/v2beta1',
    'kind': 'PipelineVersion',
    'metadata': {'name': version},
    'spec': {
        'pipelineName': args.name,
        'displayName': version,
        'pipelineSpec': pipeline_spec,
    },
}
if platform_spec:
    version_obj['spec']['platformSpec'] = platform_spec

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipelines', f'{args.name}.yaml')
with open(out, 'w') as f:
    f.write(f'# Rendered by render.py from {os.path.basename(args.compiled)}')
    f.write(f' ({args.source}).\n' if args.source else '.\n')
    if args.no_cache:
        f.write('# Execution cache off for every task (--no-cache).\n')
    f.write('# Do not edit: compile the pipeline again and re-render.\n')
    yaml.safe_dump_all([pipeline, version_obj], f, sort_keys=False, allow_unicode=True, width=1000)
print(f'wrote {out}: version {version}')

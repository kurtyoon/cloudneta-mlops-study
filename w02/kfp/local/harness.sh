#!/usr/bin/env bash
# 클러스터에 올리기 전에, 같은 이미지로 파이프라인 다섯 단계를 로컬 docker에서 돌린다.
# S3(SeaweedFS), Valkey, MLflow를 일회용 값으로 띄우고, 네트워크 별칭으로 클러스터 안 이름을 준다.
# 그래서 코드와 컴파일된 기본값을 바꾸지 않고 그대로 돈다. 키와 비밀번호는 up이 새로 만들어
# local/.work/(git 무시)에 둔다.
#
#   local/harness.sh up <malicious_phish.csv>   서버 셋, 버킷, 원본 CSV 업로드
#   local/harness.sh run <pipeline.yaml> [이름=값 ...]   컴파일된 파이프라인 실행(run_compiled.py)
#   local/harness.sh admin <python 코드>   관리자 키로 boto3 코드 실행(확인용, 변수 s3)
#   local/harness.sh valkey <명령 ...>   valkey-cli
#   local/harness.sh down   컨테이너, 네트워크, 볼륨, local/.work 삭제
#
# run은 PyYAML이 있는 파이썬이 필요하다(컴파일용 kfp venv면 된다): PYTHON=.venv-kfp/bin/python
set -euo pipefail
cd "$(dirname "$0")"
work=$PWD/.work
net=capstone-kfp-local
image=${IMAGE:-capstone-kfp:dev}
py=${PYTHON:-python3}

key() { od -An -N"$1" -tx1 /dev/urandom | tr -d ' \n'; }
admin() { # boto3로 관리자 작업. 변수 s3가 클라이언트다.
  docker run --rm --network "$net" --env-file "$work/admin.env" -v "$work:/work" "$image" python -c "
import boto3
s3 = boto3.client('s3', endpoint_url='http://seaweedfs-all-in-one.storage.svc.cluster.local:9000', region_name='us-east-1')
$1"
}

case ${1:-} in
up)
  csv=${2:?원본 CSV 경로}
  mkdir -p "$work"
  chmod 700 "$work"
  # 클러스터의 s3-identities.json과 같은 권한, 일회용 키. kfp-pipeline은 capstone/ 아래와
  # inference-datasets에만 쓸 수 있다.
  for id in admin mlflow kfp; do
    printf '%s %s\n' "$(key 10)" "$(key 20)" > "$work/key-$id"
  done
  read -r ak as < "$work/key-admin"; read -r mk ms < "$work/key-mlflow"; read -r kk ks < "$work/key-kfp"
  cat > "$work/s3.json" <<EOF
{"identities": [
 {"name": "admin", "credentials": [{"accessKey": "$ak", "secretKey": "$as"}], "actions": ["Admin", "Read", "Write", "List", "Tagging"]},
 {"name": "mlflow", "credentials": [{"accessKey": "$mk", "secretKey": "$ms"}], "actions": ["Read:mlflow", "Write:mlflow/*", "List:mlflow"]},
 {"name": "kfp-pipeline", "credentials": [{"accessKey": "$kk", "secretKey": "$ks"}], "actions": [
   "Read:feature-registry", "Write:feature-registry/capstone/*", "List:feature-registry",
   "Read:feature-data-sets", "Write:feature-data-sets/capstone/*", "List:feature-data-sets",
   "Read:inference-datasets", "Write:inference-datasets/*", "List:inference-datasets"]}
]}
EOF
  printf 'AWS_ACCESS_KEY_ID=%s\nAWS_SECRET_ACCESS_KEY=%s\n' "$ak" "$as" > "$work/admin.env"
  vp=$(key 16)
  printf 'requirepass %s\nmaxmemory 96mb\nmaxmemory-policy noeviction\nsave ""\n' "$vp" > "$work/valkey.conf"
  printf 'REDISCLI_AUTH=%s\n' "$vp" > "$work/valkey.env"
  cat > "$work/secrets.json" <<EOF
{"kfp-pipeline-s3": {"AWS_ACCESS_KEY_ID": "$kk", "AWS_SECRET_ACCESS_KEY": "$ks"},
 "feast-valkey": {"VALKEY_PASSWORD": "$vp"}}
EOF
  chmod 600 "$work"/*
  chmod 644 "$work/s3.json" "$work/valkey.conf"  # 컨테이너 안의 서버 사용자가 읽는다

  docker network create "$net" >/dev/null
  docker run -d --name capstone-s3 --network "$net" \
    --network-alias seaweedfs-all-in-one.storage.svc.cluster.local -v capstone-s3-data:/data \
    -v "$work/s3.json:/etc/sw/s3.json:ro" chrislusf/seaweedfs:4.48 \
    server -dir=/data -ip.bind=0.0.0.0 -volume.max=0 -master.volumeSizeLimitMB=100 \
    -filer -s3 -s3.port=9000 -s3.config=/etc/sw/s3.json >/dev/null
  docker run -d --name capstone-valkey --network "$net" --network-alias valkey.feast.svc.cluster.local \
    --memory 256m -v "$work/valkey.conf:/etc/valkey/valkey.conf:ro" --env-file "$work/valkey.env" \
    valkey/valkey:9.1.2-trixie valkey-server /etc/valkey/valkey.conf >/dev/null
  # MLflow는 클러스터처럼 아티팩트를 프록시해 S3 버킷 mlflow에 둔다. 호스트의 127.0.0.1:35000도 열어
  # 노트북 쪽 클라이언트(post-2 환경)가 붙을 수 있게 한다.
  docker run -d --name capstone-mlflow --network "$net" --network-alias mlflow.mlflow.svc.cluster.local \
    -p 127.0.0.1:35000:5000 -v capstone-mlflow-data:/mlflow \
    -e AWS_ACCESS_KEY_ID="$mk" -e AWS_SECRET_ACCESS_KEY="$ms" -e AWS_DEFAULT_REGION=us-east-1 \
    -e MLFLOW_S3_ENDPOINT_URL=http://seaweedfs-all-in-one.storage.svc.cluster.local:9000 \
    -e MLFLOW_SERVER_ALLOWED_HOSTS=mlflow.mlflow.svc.cluster.local:5000,localhost:35000,127.0.0.1:35000 \
    -e MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false \
    ghcr.io/mlflow/mlflow:v3.17.2-full mlflow server --host=0.0.0.0 --port=5000 --workers=2 \
    --backend-store-uri sqlite:////mlflow/mlflow.db --serve-artifacts --artifacts-destination=s3://mlflow >/dev/null

  for _ in $(seq 1 60); do
    admin 's3.list_buckets()' >/dev/null 2>&1 && break
    sleep 2
  done
  cp "$csv" "$work/malicious_phish.csv"
  admin "
for b in ('feature-data-sets', 'feature-registry', 'inference-datasets', 'mlflow'):
    s3.create_bucket(Bucket=b)
s3.upload_file('/work/malicious_phish.csv', 'feature-data-sets', 'capstone/raw/malicious_phish.csv')
print('buckets', [b['Name'] for b in s3.list_buckets()['Buckets']])"
  rm "$work/malicious_phish.csv"
  for _ in $(seq 1 60); do
    curl -fsS http://127.0.0.1:35000/health >/dev/null 2>&1 && break
    sleep 2
  done
  echo "mlflow $(curl -fsS http://127.0.0.1:35000/health), valkey $(docker exec --env-file "$work/valkey.env" capstone-valkey valkey-cli ping)"
  ;;
run)
  yaml=${2:?컴파일된 pipeline.yaml}
  shift 2
  params=()
  for p in "$@"; do params+=(--param "$p"); done
  "$py" run_compiled.py "$yaml" --network "$net" --secrets "$work/secrets.json" --work "$work/runs" "${params[@]}"
  ;;
admin)
  admin "${2:?python 코드}"
  ;;
valkey)
  shift
  docker exec --env-file "$work/valkey.env" capstone-valkey valkey-cli "$@"
  ;;
down)
  docker rm -f capstone-s3 capstone-valkey capstone-mlflow >/dev/null 2>&1 || true
  docker network rm "$net" >/dev/null 2>&1 || true
  docker volume rm capstone-s3-data capstone-mlflow-data >/dev/null 2>&1 || true
  rm -rf "$work"
  ;;
*)
  sed -n 2,14p "$0"
  exit 1
  ;;
esac

# 1주차: 서빙 이미지 고정

`../baseline/`을 복사해 `pipeline/`만 고친 상태입니다. `analysis/`는 원본과 같습니다. 모델 설명과 원본 진단은 [baseline/README.md](../baseline/README.md)에 있습니다.

## 글

- [다시 만들 수 없는 모델](https://blog.somnitor.dev/posts/81/reading-capstone-through-ml-lifecycle): 원본 진단, 코드 변경 없음
- [같은 태그, 다른 이미지](https://blog.somnitor.dev/posts/83/same-tag-different-image): 아래 변경

## 원본에서 바꾼 파일

- `pipeline/Dockerfile`: 베이스 이미지를 2024-06-27 기록의 다이제스트로 고정, 해시 검사(`--require-hashes`)로 설치, 토크나이저를 고정 리비전으로 이미지에 넣음, 커밋과 모델 해시를 라벨로 기록
- `pipeline/requirements.in`: 원본 `requirements.txt`에 따로 설치하던 fastapi, uvicorn을 더함
- `pipeline/requirements.lock` (새 파일): `requirements.in`을 2024-06-30 기준으로 풀어 해시까지 적은 잠금 파일
- `pipeline/build-image.sh` (새 파일): `<커밋>-<모델 해시>` 태그로 이미지를 빌드
- `pipeline/serve.py`: 이미지에 넣은 토크나이저를 읽음
- `pipeline/bentofile.yaml`: 잠금 파일을 포함

전체 diff는 [커밋 38dcf32](https://github.com/kurtyoon/cloudneta-mlops-study/commit/38dcf32)에서 볼 수 있습니다.

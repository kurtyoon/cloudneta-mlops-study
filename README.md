# cloudneta-mlops-study

MLOps 스터디(2026-10 ~ 11)를 하며 2024년 캡스톤의 악성 URL 분류 모델 코드를 주차마다 고쳐 가는 저장소입니다. 글은 [blog.somnitor.dev](https://blog.somnitor.dev)에 올립니다.

| 경로 | 내용 |
|---|---|
| `baseline/` | 캡스톤 원본([CSID-DGU/2024-1-CECD1-EJO-2](https://github.com/CSID-DGU/2024-1-CECD1-EJO-2) `09b1ebe`)에서 모델 관련 코드만 그대로 옮긴 것. 고치지 않습니다 |
| `capstone/` | `baseline/`을 복사해 주차마다 고쳐 가는 사본 |

원본의 Spring 백엔드 코드는 올리지 않았습니다. 학습 데이터와 학습된 모델 파일도 원본 저장소에 없어서 여기에도 없습니다.

## 주차별 변경

| 주차 | 글 | 바꾼 곳 |
|---|---|---|
| 1주차 | [다시 만들 수 없는 모델](https://blog.somnitor.dev/posts/81/reading-capstone-through-ml-lifecycle) | 없음 (원본 진단) |
| 1주차 | 같은 태그, 다른 이미지 | `capstone/pipeline/`: 서빙 이미지의 베이스 이미지·의존성·토크나이저 고정, 모델 해시를 담은 이미지 태그 |

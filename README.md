# cloudneta-mlops-study

2026년 10월부터 11월까지 MLOps 스터디를 하며 2024년 캡스톤의 악성 URL 분류 모델 코드를 주차마다 고쳐 가는 저장소입니다. 글은 [blog.somnitor.dev](https://blog.somnitor.dev)에 올립니다.

## 구성

- `baseline/`: 캡스톤 원본 [CSID-DGU/2024-1-CECD1-EJO-2](https://github.com/CSID-DGU/2024-1-CECD1-EJO-2)의 `09b1ebe`에서 모델 관련 코드만 그대로 옮긴 것입니다. 이 코드는 고치지 않습니다. 모델 설명과 원본 진단은 [baseline/README.md](baseline/README.md)에 있습니다.
- `w01/`, `w02/`, ...: 그 주가 끝났을 때의 코드 전체입니다. 주차마다 지난 주차 폴더를 복사해 고치고, 지난 주차 폴더의 코드는 다시 고치지 않습니다. 폴더마다 README에 그 주의 글과 바꾼 파일을 적었습니다.

원본의 Spring 백엔드 코드는 올리지 않았습니다. 학습 데이터와 학습된 모델 파일도 원본 저장소에 없어서 여기에도 없습니다.

## 주차

- [1주차](w01/): 서빙 이미지 고정
- [2주차](w02/): 학습을 MLflow에 기록, Feast와 Kubeflow Pipelines

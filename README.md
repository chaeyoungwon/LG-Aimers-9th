# LG Aimers 9th — Pitch Control Prediction

LG Aimers 9기 Phase 2 야구 투구 단위 제구 성공 확률 예측 프로젝트입니다.

## 기록

- 검증된 최고 리더보드 BSS: **957.6295435994	**
- 현재 최고 구성: 기존 최적 LightGBM 앙상블 90% + 정규화 선수 임베딩 MLP 10%
- 이전 최고 BSS: **939.9186320496**
- 모델: calibrated LightGBM ensemble + player embedding MLP

## 제출 이력

| 차수 | 주요 변경 | 리더보드 BSS |
| ---: | --- | ---: |
| 1차 | 기본 LightGBM 앙상블 | 937.6158960266 |
| 2차 | 단조 제약 LightGBM 비중 확대 | 934.3279102770 |
| 3차 | 기본 앙상블 85% + 단조 앙상블 15% | 939.0447236482 |
| 4차 | 기본 앙상블 60% + 단조 앙상블 40% | **939.9186320496** |
| 5차 | 최적 LightGBM 앙상블 90% + 선수 임베딩 MLP 10% | **957.6295435994	** |

2차에서는 단조 제약 비중이 지나치게 커 점수가 하락했지만, 예측 오차의 다양성이 확인되었습니다. 이후 확률 공간에서 보수적으로 블렌딩하여 3차와 4차 제출에서 최고점을 갱신했습니다. 5차에서는 수치형 표준화와 투수·타자 ID 임베딩을 적용한 MLP를 10% 혼합해 **957.6295435994	점**으로 최고점을 다시 갱신했습니다.

## 폴더 구조

```text
.
├── src/                         # 학습 코드
│   ├── train_base.py            # 기본 3종 LightGBM 학습
│   ├── train_monotone.py        # 단조 모델 학습 및 최적 블렌드 설정
│   ├── train_nn_candidate.py    # 정규화 선수 임베딩 MLP 학습
│   └── nn_embed.py              # MLP 전처리·임베딩·직렬화 유틸
├── candidates/
│   └── nn10/                    # 리더보드 957.6295435994	점 NN 10% 제출 구성
├── submission/                  # 평가 서버에 들어가는 파일의 원본
│   ├── model/                   # 학습 모델 및 앙상블 메타데이터
│   ├── script.py                # 평가 서버 실행 진입점
│   └── requirements.txt
├── scripts/
│   └── build_submission.py      # submission/ → dist/submit.zip
├── artifacts/
│   ├── submit_40pct_lb939.918632.zip  # 점수가 확인된 제출물
│   ├── submit_optimal_39.094pct_expected939.919868.zip
│   └── submit_nn10_candidate.zip      # 리더보드 957.6295435994	점 제출물
└── data/                        # 로컬 전용, Git에서 제외
```

## 파일 역할

`submission/`은 제출 ZIP의 압축을 풀었을 때 보이는 구조입니다. 개발 중에는 이 폴더의 코드와 모델을 수정합니다.

`dist/submit.zip`은 업로드용 빌드 결과입니다. Git에 커밋하지 않고 필요할 때 생성합니다.

`artifacts/submit_40pct_lb939.918632.zip`은 실제 리더보드에서 939.9186320496점을 기록한 보존용 스냅샷입니다. `submit_optimal_39.094pct_expected939.919868.zip`은 네 번의 실제 제출 점수로 복원한 최적 혼합비 후보이며, `submit_nn10_candidate.zip`은 **957.6295435994	점**을 기록한 현재 최고 제출물입니다.

## 학습

대회 데이터를 저장소 루트의 `data/`에 배치한 뒤 실행합니다.

```bash
python -m src.train_base
python -m src.train_monotone
```

첫 번째 명령은 기본 모델 3종을 학습하고, 두 번째 명령은 단조 제약 모델을 학습한 뒤 60.9056:39.0944 앙상블 메타데이터를 구성합니다.

## 제출 ZIP 생성

```bash
python scripts/build_submission.py
```

생성 결과:

```text
dist/submit.zip
├── model/
├── script.py
└── requirements.txt
```

## 모델 구성

- 일반 LightGBM GBDT (`num_leaves=31`, `63`)
- 최근 시즌에 높은 표본 가중치를 적용한 LightGBM
- 제구 성공·스트라이크 비율에 양의 단조 제약 적용
- 가운데·반대·볼 비율에 음의 단조 제약 적용
- 각 앙상블에 고정 확률 캘리브레이션 적용
- 수치형 중앙값 대치·표준화와 결측 플래그를 적용한 MLP
- 투수·타자 ID 임베딩 및 신규 선수 대응 ID dropout 적용

현재 최고 제출의 최종 확률은 최적 LightGBM 앙상블 90%와 선수 임베딩 MLP 10%의 가중 평균입니다.

## 규정 준수

추론은 각 테스트 행의 값과 학습 시 고정된 모델·매핑만 사용합니다. 테스트 데이터의 다른 행을 이용한 집계, rolling, lag, 빈도, 순위 또는 분포 보정은 사용하지 않습니다.

대회 데이터는 배포 규정에 따라 저장소에 포함하지 않습니다.

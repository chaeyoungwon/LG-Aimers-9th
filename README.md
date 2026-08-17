# LG Aimers 9th — Pitch Control Prediction

LG Aimers 9기 Phase 2 야구 투구 단위 제구 성공 확률 예측 프로젝트입니다.

## 기록

- 검증된 최고 리더보드 BSS: **1007.50439**
- 현재 최고 구성: R10 이종 앙상블 복원본 + 정규시즌 투수·타자 season-to-date form 보정
- 이전 최고 BSS: **995.300917194**
- 모델: calibrated LightGBM ensemble + player embedding MLP

## 제출 이력

| 차수 | 주요 변경 | 리더보드 BSS |
| ---: | --- | ---: |
| 1차 | 기본 LightGBM 앙상블 | 937.6158960266 |
| 2차 | 단조 제약 LightGBM 비중 확대 | 934.3279102770 |
| 3차 | 기본 앙상블 85% + 단조 앙상블 15% | 939.0447236482 |
| 4차 | 기본 앙상블 60% + 단조 앙상블 40% | **939.9186320496** |
| 5차 | 최적 LightGBM 앙상블 90% + 선수 임베딩 MLP 10% | **957.6295435994** |
| 6차 | 5차 NN10 + 행 독립적 투수 season-to-date form 보정 (`M=75`, `alpha=0.28`) | **995.300917194** |
| 7차 | 6차 + 카운트×투·타 좌우 OOF residual-cell 보정 | 983.5132453353 |
| 8차 | R10 이종 앙상블 복원본 + 기존 투수 form + 신규 타자 form (`M=50`, `alpha=0.112506`) | **1007.50439** |

2차에서는 단조 제약 비중이 지나치게 커 점수가 하락했지만, 예측 오차의 다양성이 확인되었습니다. 이후 확률 공간에서 보수적으로 블렌딩하여 3차와 4차 제출에서 최고점을 갱신했습니다. 5차에서는 수치형 표준화와 투수·타자 ID 임베딩을 적용한 MLP를 10% 혼합해 **957.6295435994점**을 기록했습니다. 6차에서는 공식 제공 as-of 누적값과 train-only 투수 상수표로 정규시즌 당해 시즌 폼을 행별 복원해 **995.300917194점**으로 최고점을 갱신했습니다. 7차 residual-cell 보정은 로컬 OOF에서는 개선됐지만 리더보드에서 **983.5132453353점(-11.7877)**으로 하락해 기각했습니다. 8차에서는 R10 이종 앙상블 복원본의 기존 투수 form 위에, 같은 행 독립 원리로 계산한 타자 season-to-date form을 추가해 **1007.50439점**으로 최고 기록을 갱신했습니다.

## 폴더 구조

```text
.
├── src/                         # 학습 코드
│   ├── train_base.py            # 기본 3종 LightGBM 학습
│   ├── train_monotone.py        # 단조 모델 학습 및 최적 블렌드 설정
│   ├── train_nn_candidate.py    # 정규화 선수 임베딩 MLP 학습
│   └── nn_embed.py              # MLP 전처리·임베딩·직렬화 유틸
├── candidates/
│   ├── nn10/                    # 리더보드 957.6295435994점 NN 10% 제출 구성
│   └── nn10_form_a28/           # 리더보드 995.300917194점 season-form 구성
├── submission/                  # 평가 서버에 들어가는 파일의 원본
│   ├── model/                   # 학습 모델 및 앙상블 메타데이터
│   ├── script.py                # 평가 서버 실행 진입점
│   └── requirements.txt
├── scripts/
│   └── build_submission.py      # submission/ → dist/submit.zip
├── artifacts/
│   ├── submit_40pct_lb939.918632.zip  # 점수가 확인된 제출물
│   ├── submit_optimal_39.094pct_expected939.919868.zip
│   ├── submit_nn10_candidate.zip      # 리더보드 957.6295435994점 제출물
│   ├── submit_nn10_a28.zip            # 리더보드 995.300917194점 제출물
│   ├── submit_nn10_a28_cells.zip      # 983.5132453353점, 기각 보존본
│   └── submit_r10_pitcher.zip          # 리더보드 1007.50439점, 현재 최고
└── data/                        # 로컬 전용, Git에서 제외
```

## 파일 역할

`submission/`은 제출 ZIP의 압축을 풀었을 때 보이는 구조입니다. 개발 중에는 이 폴더의 코드와 모델을 수정합니다.

`dist/submit.zip`은 업로드용 빌드 결과입니다. Git에 커밋하지 않고 필요할 때 생성합니다.

`artifacts/submit_40pct_lb939.918632.zip`은 실제 리더보드에서 939.9186320496점을 기록한 보존용 스냅샷입니다. `submit_optimal_39.094pct_expected939.919868.zip`은 네 번의 실제 제출 점수로 복원한 최적 혼합비 후보이고, `submit_nn10_candidate.zip`은 **957.6295435994점**을 기록했습니다. 현재 최고 제출물은 `submit_r10_pitcher.zip`이며 실제 리더보드 점수는 **1007.50439**입니다. 이 파일의 SHA-256은 `c7f0c4d42c6a0487dd218bd725a5715987e98c37cf62a8e4198b56692564aeb2`입니다.

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

현재 최고 제출은 R10 이종 앙상블 복원본을 기반으로 합니다. 마지막 단계에서 기존 투수 season-form(`M=75`, `alpha=0.28`)을 적용한 뒤, train에서 본 타자이면서 당해 시즌 표본이 존재하는 `game_type=R` 행에 타자 season-form(`M=50`, `alpha=0.11250648296393913`)을 한 번 더 적용합니다. 두 보정 모두 각 평가 행과 train-only 고정 상수표만 사용하는 행 독립 방식입니다.

## 규정 준수

추론은 각 테스트 행의 값과 학습 시 고정된 모델·매핑만 사용합니다. 테스트 데이터의 다른 행을 이용한 집계, rolling, lag, 빈도, 순위 또는 분포 보정은 사용하지 않습니다.

대회 데이터는 배포 규정에 따라 저장소에 포함하지 않습니다.

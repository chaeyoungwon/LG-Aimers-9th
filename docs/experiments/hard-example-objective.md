# Hard-Example / Uncertainty-Aware objective 검증

실행일: 2026-08-22. 동일한 공식 47개 predictor와 기존 LightGBM team recipe를
유지한 채 학습 sample weight만 바꿨다. 결론은 **NO SUBMISSION CANDIDATE**다.

## A. Inner OOF

각 시즌 OOF는 그보다 과거 시즌만 학습한 CatBoost 30% + LightGBM team 70%다.
2020 OOF는 2019만 학습해 추가 생성했고 기존 2021~2023 rolling OOF를 재사용했다.
과거가 없는 2019행은 중립 weight 1.0이다.

| Outer validation | train inner OOF coverage | source seasons |
|---:|---:|---|
| 2022 | 67.41% | 2020, 2021 |
| 2023 | 75.68% | 2020, 2021, 2022 |
| 2024 | 80.57% | 2020, 2021, 2022, 2023 |

Outer validation target/prediction/distribution은 weight 함수에 전달하지 않았다. hard
weight는 inner OOF squared error q50/q75/q90에 따라 `1/1.5/2/3`, uncertainty는
`1 + λ·4p(1-p)`, λ=.5/1.0이다. 미래 label 변경 시 기존 train weight가 비트 동일한
단위 테스트를 추가했다.

## B. Candidate result

seed 42·43 평균 Brier다. Specialist는 A/B가 전부 실패해 사전 중단 규칙대로 실행하지
않았다.

| Fold | Champion | Hard-weight | Uncertainty .5 | Uncertainty 1.0 | Specialist |
|---:|---:|---:|---:|---:|---|
| 2022 | .243132 | .264014 | .243197 | .243221 | NOT RUN |
| 2023 | .252347 | .261563 | .252614 | .252607 | NOT RUN |
| 2024 | .247655 | .267692 | .247683 | .247685 | NOT RUN |

모든 후보가 모든 폴드에서 악화했다. hard weighting은 어려운 행의 반대 방향을 과도하게
학습해 champion과 prediction correlation까지 음수가 됐다.

## C. Hard-error subgroup

표는 candidate 단독 Brier gain이며 양수가 개선이다. 사후 oracle인 champion error
상위 구간에서도 uncertainty .5는 전부 악화했다.

| Fold | top 1% | top 5% | top 10% | top 20% |
|---:|---:|---:|---:|---:|
| 2022 | -.007458 | -.004378 | -.002784 | -.001890 |
| 2023 | -.001824 | -.003353 | -.003273 | -.002744 |
| 2024 | -.003012 | -.002910 | -.002775 | -.002330 |

Hard 후보는 이 oracle 구간에서 양수지만 전체 calibration과 Brier가 극단적으로
붕괴하므로 배포 가능한 개선이 아니다. 고정 `p<.4/y=1`, `p>.6/y=0` sign 구간도
산출물에 기록했다.

## D. Residual complementarity

| Fold | 후보 | prediction corr | residual corr |
|---:|---|---:|---:|
| 2022 | uncertainty .5 | .996458 | .999895 |
| 2023 | uncertainty .5 | .995999 | .999914 |
| 2024 | uncertainty .5 | .991387 | .999926 |

residual correlation `>.995` 빠른 중단 조건을 명확히 충족한다. Hard 후보의 residual
corr은 `.961~.965`지만 낮은 상관은 유용한 신호가 아니라 예측 반전과 calibration
붕괴에서 왔다.

## E. Calibration

| Fold | target mean | champion mean | hard mean / ECE | unc .5 mean / ECE | champion ECE |
|---:|---:|---:|---:|---:|---:|
| 2022 | .528920 | .525074 | .480250 / .114837 | .522482 / .008204 | .004860 |
| 2023 | .499957 | .510607 | .499450 / .091920 | .506822 / .033662 | .029381 |
| 2024 | .486105 | .489717 | .491913 / .116535 | .486275 / .001009 | .004091 |

Hard weighting은 확률 모델로 사용할 수 없는 수준이다. Uncertainty는 평균/ECE를 일부
움직였지만 Brier·logloss를 개선하지 못했다. validation calibration은 fit하지 않았다.

## F. Blend

두 seed 평균, BSS 단위다. 대표 uncertainty .5:

| Weight | 2022 | 2023 | 2024 | Mean Δ |
|---:|---:|---:|---:|---:|
| 2.5% | -.09 | -2.10 | +.19 | -.67 |
| 5% | -.20 | -4.24 | +.35 | -1.36 |
| 10% | -.52 | -8.59 | +.60 | -2.84 |
| 15% | -.96 | -13.05 | +.75 | -4.42 |

Hard 5%는 `-39.12 / +173.58 / -10.44`, mean `+41.34`지만 2023 하나에만
집중되고 std `115.42`, 2SE `133.27`이라 실패다.

## G. Statistics

Uncertainty .5의 5% blend는 mean `-1.36`, std `2.50`, 2SE `2.89 BSS`다.
Uncertainty 1.0은 mean `-1.41`, std `2.33`, 2SE `2.69 BSS`다. seed 42·43 모두
세 후보가 2022·2024를 악화했다. 사전 등록된 fast-stop에 따라 seed 44를 생략했다.

## H. Rule audit

- future leakage: **PASS** — outer validation은 weight 생성에 사용하지 않음
- inner OOF leakage: **PASS** — 각 inner season보다 과거만 학습
- test row independence: **PASS (structural)** — inference는 frozen LightGBM과 현재
  행만 사용하며 test 집계/빈도/order/threshold/calibration 없음
- production ZIP test: **NOT RUN** — 성능 gate 실패로 ZIP을 만들지 않음

## I. 최종 결정

**NO SUBMISSION CANDIDATE**

Signed residual specialist, seed 44, 재보정, weight/grid 확장은 실행하지 않는다. 현재
`artifacts/sub_tree_reblend_w0p300.zip`과 SHA-256은 변경하지 않았다. 재현 코드는
`scripts/validate_hard_example_objective.py`, 단위 테스트는
`tests/test_hard_example_objective.py`, 상세 CSV/OOF는
`artifacts/backtest/hard_example/`에 있다.

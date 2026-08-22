# 공식 제공 데이터 전체 Feature Inventory 및 신규 정보축 검증

실행일: 2026-08-22

## 결론

공식 main table의 test-visible predictor 47개는 21차 Cat30/team70 champion이 모두
직접 사용한다. 완전히 미사용인 SAFE raw predictor는 없다. 파생 표현 후보 중 기존
실험과 겹치지 않는 `pitcher usage/role state` 하나만 실제 검증했지만 최근 두 시즌과
두 번째 seed에서 악화했고 residual correlation도 `0.999764`였다.

**FAIL — NO SUBMISSION CANDIDATE.** 현재 최고 ZIP은 변경하지 않았다.

## A. 공식 데이터 inventory

공식 배포 경로에서 확인한 파일은 6개다.

| 파일 | 형태 | 컬럼/멤버 | 역할 |
|---|---:|---:|---|
| `data/train.csv` | 1,475,092 rows | 49 columns | main train + target |
| `data/test.csv` | local sample 5 rows | 48 columns | evaluation schema |
| `data/sample_submission.csv` | 5 rows | 2 columns | output contract |
| `data/trackman_history.csv` | 1,793,078 rows | 30 columns | official historical TrackMan |
| `data_description.md` | document | - | schema/rule description |
| `baseline_submit.zip` | archive | 4 members | baseline reference |

CSV별 column entry는 129개, 고유 column 이름은 69개다. Main train/test 차이는
`control_success` 하나뿐이며 `row_id`를 제외한 test-visible predictor는 47개다.
Machine-readable 전체 목록은 `artifacts/feature_inventory.csv`에 저장했다.

요약:

- champion 직접 사용: main predictor **47/47**
- 파생 사용: champion 생성 피처 23개(`count_state`, hand/gap/log/state 등), 원천 raw는
  위 47개와 중복
- 기각 이력: TrackMan 30개, context/pressure/reliability/identity 파생 축 전반
- 미사용 SAFE raw predictor: **0**
- TRAIN-ONLY predictive source: `control_success` + current-pitch TrackMan 30개
- raw schema UNSAFE: **0**; 다만 main table에 game/date/order가 없어 workload와
  same-game familiarity 파생은 UNSAFE

## Champion 실제 feature data flow

21차 ZIP `meta.json`과 `script.py`를 역추적했다.

### CatBoost 30%

- 최종 입력 65개
- 공식 raw/context/as-of 45개(ID 제외)
- row-local derived: `count_state`, `same_hand`, `hand_combo`, gap/dominance
- train-only lookup derived: season-state pitcher/batter 13개
- categorical은 고정 train vocabulary, 수치는 현재 행과 고정 artifact만 사용

### LightGBM team 70%

- 최종 입력 70개
- 공식 predictor 47개 전부(ID 포함)
- row-local derived: count/hand/late-inning/support/recent gap
- season-state pitcher/batter 13개
- 4개 frozen recipe → 두 group calibration → 고정 60:40 group blend

따라서 공식 컬럼 분류는 다음과 같다.

- A 직접 사용: 47개 main predictor
- B 파생 형태로 사용: 위 47개를 원천으로 하는 23개 champion derived feature
- C 기존 실험 후 기각: TrackMan, pressure/context, reliability, target encoding/GNN,
  state derivative, count/hand interaction 등
- D 실질적 미사용: **raw column 없음**. 안전한 구조 표현 또는 inference 불가능한
  workload/familiarity만 남음

## 규칙 안전성

- Main test 47 predictor: `SAFE` — 공식 current-row/as-of 값
- 고정 임계값 role/context 표현: `SAFE WITH TRANSFORMATION`
- current-pitch TrackMan, target: `TRAIN-ONLY`
- game workload, within-game pitch index, same-game familiarity: `UNSAFE` — main test에
  game ID/date/order가 없음
- 외부 run-expectancy/weather lookup: `UNSAFE`

Test groupby, frequency, rolling, shift, cumsum, test-wide normalization/calibration은 어느
후보에도 허용하지 않았다.

## B. 미사용 정보축 Top 5

Cost는 5가 가장 비싸다.

| Rank | 정보축 | Novelty | Coverage | Safety | Residual independence 기대 | Temporal 기대 | Cost | 판단 |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Pitcher usage/role state | 3 | 5 | 5 | 3 | 3 | 2 | **실험 선정** |
| 2 | Batter opportunity/reliability state | 2 | 5 | 5 | 2 | 3 | 2 | season-state와 중복 큼 |
| 3 | Tactical/run-expectancy state | 2 | 5 | 5 | 2 | 2 | 2 | pressure family 기각 이력 |
| 4 | Team × role context | 3 | 5 | 5 | 2 | 2 | 3 | 1위 축에 포함 |
| 5 | Workload/fatigue/familiarity | 5 | 0 | 0 | 4 | 2 | 5 | game/date/order 부재로 UNSAFE |

TrackMan은 미사용 raw source지만 direct/profile/LUPI/distillation 및 실제 LB 실패까지
완료된 폐쇄 축이라 Top 5 신규 실험에서 제외했다.

## C. 실제 실험 후보

선정 축은 `pitcher usage/role state`다. 기존 state×count 실험을 반복하지 않고 다음
현재 행 기반 구조만 추가했다.

- `role_inning_phase`: early/middle/late
- `role_experience`: official `asof_pitcher_n`의 고정 workload bucket
- `role_pitch_mix`: official as-of fastball/breaking/offspeed 우세군
- `role_recent_stability`: prev1 vs prev5 성공률 차이의 고정 bucket
- `role_team_context`: pitcher team × game type × inning phase

게임별 투구 수, 이전 행, row_id 순서, test 빈도는 사용하지 않았다. 배포 team 4모델
recipe·round·monotone·calibration은 동일하게 두고 위 5개 categorical feature만 추가했다.

## D. Rolling OOF

두 seed 평균 candidate와 고정 21차 champion 비교:

| Fold | Champion BSS | Candidate BSS | Δ |
|---:|---:|---:|---:|
| 2022 | 2420.73 | 2395.08 | -25.65 |
| 2023 | -938.83 | -1263.76 | **-324.93** |
| 2024 | 861.55 | 811.38 | **-50.17** |

동일 recipe control 대비 feature 자체의 seed별 Δ BSS:

| Seed | 2022 | 2023 | 2024 |
|---:|---:|---:|---:|
| 42 | +5.25 | -250.98 | -44.10 |
| 43 | -8.15 | -288.10 | -43.24 |

두 seed 모두 2023/2024 악화, 2022 방향 반전이므로 빠른 중단 조건에 따라 seed 44는
생략했다.

## E. Residual complementarity

- prediction correlation = `0.987687`
- residual correlation = **`0.999764`**
- champion error top 10% candidate Brier gain = `-0.00297646`
- champion error top 20% candidate Brier gain = `-0.00181261`

후보는 champion이 크게 틀린 행에서도 더 나빴고 residual 독립성 기준 `0.995`를 크게
넘었다.

## F. Small blend

두 seed 평균 candidate를 21차 champion에 섞은 Δ BSS다.

| Candidate weight | 2022 | 2023 | 2024 | Mean Δ |
|---:|---:|---:|---:|---:|
| 2.5% | +0.51 | -6.76 | -0.14 | -2.13 |
| 5% | +0.96 | -13.59 | -0.34 | -4.32 |
| 10% | +1.68 | -27.45 | -0.91 | -8.89 |
| 15% | +2.17 | -41.60 | -1.70 | -13.71 |

## Subgroup

5% blend는 2023 low-history pitcher `-43.82`, low-history batter `-47.60`, team 13
`-59.90 BSS`로 악화했다. Count, inning, L/R, team별로 일부 양수 셀이 있으나 2023
대부분이 음수이며 2024에서도 팀별 부호가 엇갈린다. 특정 역할 subset에서만 안정적인
개선이 있다는 근거가 없다. 상세는 `artifacts/backtest/role_state/subgroup_results.csv`.

## G. 통계 판정

5% blend fold Δ 기준:

- mean = `-4.32 BSS`
- std = `8.05 BSS`
- SE = `4.65 BSS`
- 2SE = `9.29 BSS`
- seed stability = FAIL
- recent season non-collapse = FAIL
- residual complementarity = FAIL
- 최종 판정 = **FAIL**

## H. 최종 결정

**NO SUBMISSION CANDIDATE**

검증 코드는 `scripts/validate_role_state.py`, 행 지역성 테스트는
`tests/test_role_state.py`다. 현재 최고 `artifacts/sub_tree_reblend_w0p300.zip`은 수정하지
않았다.

# Official-data-only TrackMan pitcher crosswalk

실행일: 2026-08-21

## 결론

**B. RELIABLE CROSSWALK BUT WEAK MODEL SIGNAL**

공식 `train.csv`와 `trackman_history.csv`의 anonymized pitcher namespace는 직접 겹치지 않지만, pitch-mix/time-series fingerprint로 HIGH-confidence subset을 통계적으로 연결할 수 있었다. 다만 이 결과는 외부 실명 정답으로 identity를 확인했다는 뜻이 아니다. 공식 데이터 내부 future-season holdout과 cutoff stability가 지지하는 제한적 crosswalk이며, LOW mapping은 feature 생성에 사용하지 않았다.

TrackMan feature를 사용한 ET25 small blend는 세 fold에서 양수였지만 strict strong 기준인 pooled `>= 2e-5`를 넘지 못했다. production ZIP은 생성하지 않았고 champion을 유지한다.

- Champion: `artifacts/sub_et25_w020.zip`
- LB BSS: `1017.0233029621`
- SHA-256 before/after: `316bf6dab11fb78f06cc74089ea38bdb3b957355f4510db5fcc8a485dc26cecf`

## 데이터·금지사항 준수

- Identity inference에 사용한 데이터: 공식 `train.csv`, `trackman_history.csv`뿐
- 공식 설명서로 schema와 historical-use contract만 확인
- 선수 이름, KBO/Statiz/roster, 외부 기록, fuzzy name matching, 수동 label 사용 없음
- `test.csv` 읽지 않음
- validation season fingerprint와 TrackMan row를 해당 fold mapping/profile fit에 사용하지 않음

## Fingerprint 정의

Main의 `asof_pitcher_pitchmix_n`과 세 rate는 시즌이 바뀌어도 단조 증가하는 누적 이력이다. 따라서 pitcher-season에서 최소 누적 group count와 terminal 누적 group count를 빼 season contribution을 복원했다. 이 차분은 fastball/breaking/offspeed/other 합이 모든 유효 profile에서 정확히 1이 되는지 검사했다.

TrackMan은 `pitcher_trackman_id × season`별 pitch count와 동일한 네 pitch group 비율로 집계했다. 손잡이 namespace는 외부 의미를 넣지 않고 season별 prevalence 오차가 최소인 permutation을 선택했다.

- 선택: main `1 → Left`, `2 → Right`
- 선택 permutation squared error: `0.000282`
- 반대 permutation squared error: `2.768607`

Pair distance는 multi-season pitch-mix L1을 주항으로 두고 relative log-count pattern, season presence, coverage ratio, overlap penalty를 작은 고정 가중치로 더했다. NN과 one-to-one Hungarian을 모두 평가했고, 모든 투수를 강제로 연결하지 않았다.

## Confidence와 coverage

Feature/model에는 선택한 Hungarian의 HIGH tier만 사용했다.

| Cutoff | Hungarian HIGH | MEDIUM | LOW | UNMATCHED |
|---:|---:|---:|---:|---:|
| ≤2021 | 90 | 33 | 290 | 128 |
| ≤2022 | 91 | 31 | 340 | 162 |
| ≤2023 | 108 | 27 | 354 | 199 |
| ≤2024 | 111 | 25 | 416 | 212 |

| Validation | Rows | HIGH matched rows | HIGH row coverage | HIGH pitcher coverage |
|---:|---:|---:|---:|---:|
| 2022 | 247,472 | 65,215 | 26.35% | 18.46% |
| 2023 | 245,525 | 70,236 | 28.61% | 17.02% |
| 2024 | 253,507 | 76,062 | 30.00% | 19.95% |
| pooled | 746,504 | 211,513 | 28.33% | — |

## Crosswalk self-validation

`≤2021→2022`, `≤2022→2023`, `≤2023→2024`에서 mapping에 쓰지 않은 다음 시즌 pitch-mix를 평가했다.

| Hungarian tier | Future pairs | Median future L1 | Top-1 | Top-5 | Top-10% rank | Median rank percentile |
|---|---:|---:|---:|---:|---:|---:|
| HIGH | 210 | 0.04660 | 29.05% | 62.38% | 83.33% | 1.24% |
| MEDIUM | 51 | 0.11518 | 13.73% | 39.22% | 70.59% | 3.87% |
| LOW | 226 | 0.18616 | 10.18% | 23.89% | 47.35% | 11.88% |

HIGH가 MEDIUM/LOW보다 future mismatch와 rank에서 명확히 분리됐다. HIGH 공통 투수의 cutoff간 mapping stability는 다음 세 비교에서 모두 100%였다.

- `≤2021 vs ≤2022`: 63/63
- `≤2022 vs ≤2023`: 80/80
- `≤2023 vs ≤2024`: 95/95

NN도 gate를 통과했지만 identity의 one-to-one 제약을 반영하고 future 결과가 동등한 Hungarian을 선택했다.

## Physical feature

각 fold는 TrackMan `<= validation season - 1`만 사용했다. Career/recent mean·std, pitch-type별 speed/spin/movement, velocity gaps, movement separation, release/movement/velocity variability, recent-career delta를 만들었다. Count bucket은 `three_ball`, `two_strike`, `neutral`로 나누고 표본 수 기반 고정 shrinkage 50을 적용했다. 모든 normalization은 cutoff 이전 TrackMan profile만 사용했다.

Residual feature gate를 통과한 7개 중 대부분은 mechanics가 아니라 usage/history volume이었다.

| Feature | Corr 2023 | Corr 2024 | Corr pooled |
|---|---:|---:|---:|
| count_two_strike_n | 0.04114 | 0.01595 | 0.02029 |
| fastball_n | 0.04041 | 0.01351 | 0.01920 |
| tm_history_n | 0.03816 | 0.01451 | 0.01871 |
| count_neutral_n | 0.03750 | 0.01433 | 0.01843 |
| offspeed_n | 0.03069 | 0.01726 | 0.01660 |
| fastball_offspeed_velocity_gap | 0.02982 | 0.01117 | 0.01588 |
| count_three_ball_n | 0.03207 | 0.01066 | 0.01509 |

## ET25 temporal OOF

두 고정 모델 모두 HIGH-matched training rows에서만 학습하고 HIGH-matched validation rows에만 적용했다. Unmatched validation prediction은 current champion과 정확히 동일하다.

- `accepted52_plus_tm`: accepted raw 52 + TrackMan history
- `tm_plus_context`: TrackMan history + 기존 row-local game/count context
- Recipe: 25 trees, `min_samples_leaf=8`, `max_features=sqrt`, seed 42

Raw hard replacement는 두 모델 모두 matched subset에서 악화됐다.

| Learner | Season | Matched gain | Overall routed gain | Corr champion | Corr ET25 | Champion top-10% error에서 model win |
|---|---:|---:|---:|---:|---:|---:|
| accepted52+TM | 2022 | -2.248e-3 | -5.925e-4 | 0.7372 | 0.6296 | 59.58% |
| accepted52+TM | 2023 | -1.890e-3 | -5.408e-4 | 0.7246 | 0.6191 | 60.52% |
| accepted52+TM | 2024 | -2.373e-3 | -7.119e-4 | 0.5635 | 0.5176 | 45.60% |
| accepted52+TM | pooled | -2.174e-3 | -6.160e-4 | 0.6820 | 0.5993 | 55.81% |
| TM+context | 2022 | -2.548e-3 | -6.715e-4 | 0.7164 | 0.5986 | 59.25% |
| TM+context | 2023 | -1.575e-3 | -4.505e-4 | 0.7480 | 0.6349 | 56.86% |
| TM+context | 2024 | -2.953e-3 | -8.861e-4 | 0.5138 | 0.4535 | 48.11% |
| TM+context | pooled | -2.371e-3 | -6.717e-4 | 0.6790 | 0.5793 | 55.44% |

## Fixed small blend

Blend는 HIGH-matched row에서만 적용하고 나머지는 champion identity를 유지했다. 양수는 개선이다.

| Learner | Weight | 2022 | 2023 | 2024 | Pooled |
|---|---:|---:|---:|---:|---:|
| accepted52+TM | .01 | +1.926e-6 | +3.444e-6 | +3.028e-6 | +2.800e-6 |
| accepted52+TM | .02 | +3.694e-6 | +6.709e-6 | +5.851e-6 | +5.418e-6 |
| accepted52+TM | .05 | +8.045e-6 | +1.543e-5 | +1.309e-5 | +1.219e-5 |
| accepted52+TM | .10 | +1.212e-5 | +2.639e-5 | +2.105e-5 | **+1.985e-5** |
| TM+context | .01 | +1.663e-6 | +4.169e-6 | +1.458e-6 | +2.418e-6 |
| TM+context | .02 | +3.157e-6 | +8.164e-6 | +2.707e-6 | +4.651e-6 |
| TM+context | .05 | +6.624e-6 | +1.909e-5 | +5.204e-6 | +1.024e-5 |
| TM+context | .10 | +9.016e-6 | +3.381e-5 | +5.197e-6 | +1.587e-5 |

`accepted52_plus_tm, w=.10`이 OOF-selected best다. 세 fold가 모두 양수이고 matched pooled gain은 `+7.006e-5`지만, overall pooled gain `+1.984983e-5`는 strict A threshold `>=2e-5`보다 `1.502e-7` 작다. 근소한 차이를 반올림해 A로 승격하지 않는다.

## Production 결정

- Verdict: B
- Production ZIP: 생성하지 않음
- LB weight selection: 사용하지 않음
- Batter mapping: 진행하지 않음
- Champion: 변경 없음

## 실행

```bash
.venv/bin/python scripts/validate_tm_crosswalk.py
```

검증된 crosswalk와 model cache를 재사용한 결과 재생성:

```bash
.venv/bin/python scripts/validate_tm_crosswalk.py --reuse-crosswalk
```

## 결과물

- `artifacts/tm_crosswalk/main_fingerprints.csv`
- `artifacts/tm_crosswalk/tm_fingerprints.csv`
- `artifacts/tm_crosswalk/mapping_candidates.csv`
- `artifacts/tm_crosswalk/mapping_stability.csv`
- `artifacts/tm_crosswalk/future_validation.csv`
- `artifacts/tm_crosswalk/coverage.csv`
- `artifacts/tm_crosswalk/feature_results.csv`
- `artifacts/tm_crosswalk/oof_results.csv`
- `artifacts/tm_crosswalk/complementarity.csv`
- `artifacts/tm_crosswalk/blend_results.csv`
- `artifacts/tm_crosswalk/summary.json`
- `scripts/validate_tm_crosswalk.py`

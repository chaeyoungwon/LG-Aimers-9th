# Champion residual discovery

작성일: 2026-08-20

## Champion

- LB BSS: `1016.4442212358`
- prediction under study: reconstructed temporal OOF `champion_final`
- immutable ZIP: `artifacts/submit_r10_pitcher_w418194.zip`
- SHA-256 before/after:
  `ce10c428c9b6bd88ebf713fcc05b91ce22fc555573f3d11f8deb283637484c47`

Regime-aware ensemble 연구는 종료했다. 이 실험에서는 weight를 다시 최적화하지 않고,
2021~2024 OOF cache를 재사용해 champion 이후 residual structure만 진단했다. Test row와
TrackMan은 읽지 않았다.

## Feature inventory

| category | row-local/train-derived features | handling |
| --- | --- | --- |
| Pitch context | month/day, inning, top/bottom, balls/strikes, outs, score/run state, runners/base state, win expectancy, leverage, game type | 현재 행에서 직접 계산 |
| Matchup | pitcher/batter hand, same-hand | 현재 행에서 직접 계산 |
| Pitcher prior/reliability | official `asof_pitcher_*`, pitch-mix count/rates, cutoff-derived `pitcher_seen` | 공식 as-of 또는 과거 cutoff lookup |
| Batter prior/reliability | official `asof_batter_n`, success/middle rate | 공식 as-of |
| Model disagreement | five-member std/range, HGB-Cat/NN gaps, ours-team gap | 같은 행의 OOF prediction만 사용 |
| Form/cold-start | pitcher/batter/total form adjustment, cold flag, cold expert gap | 기존 cutoff-safe endpoint artifact에서 복원 |

`row_id`는 정렬 검증에만 사용했다. Raw pitcher/batter/team ID는 residual diagnostic model에
넣지 않았고, validation season 자체도 drift memorization을 피하기 위해 feature에서 제외했다.

## Validation protocol

| validation | residual-model training OOF |
| ---: | --- |
| 2022 | 2021 |
| 2023 | 2022 |
| 2024 | 2022 + 2023 |

Quantile 경계, 결측 대체값, 표준화 값, category vocabulary는 각 행의 validation season보다
이른 위 training OOF에서만 계산했다. Validation label은 residual과 metric 계산에만 사용했다.

## Most stable residual patterns

95% 차이 기준을 세 fold에서 모두 만족하고, fold 간 효과 크기 최대/최소 비율이 5 이하인
경우만 robust pattern으로 분류했다.

| signal | 2022 | 2023 | 2024 | stable | interpretation |
| --- | ---: | ---: | ---: | :---: | --- |
| total form adjustment high-low residual | +0.007141 | +0.020771 | +0.009707 | yes | positive form 방향의 residual이 남음; Brier gradient는 불안정 |
| pitcher form adjustment high-low residual | +0.010305 | +0.020221 | +0.008700 | yes | form 방향과 동일하지만 기존 alpha LB 실패 축이므로 즉시 재튜닝하지 않음 |
| full count (`3-2`) residual vs complement | -0.009270 | -0.012252 | -0.015575 | yes | champion이 full-count에서 상대적으로 과대예측 |
| unseen pitcher residual vs seen | -0.005543 | -0.022911 | -0.005902 | yes | cold endpoint 이후에도 unseen에서 상대적 과대예측; Brier 우열은 fold별 반전 |
| high-low `balls_before` Brier | -0.001347 | -0.000879 | -0.000825 | yes | 많은 ball count가 반복적으로 더 쉬운 구간; signed gradient는 전 fold 유의하지 않음 |
| count state 2 Brier vs complement | +0.002470 | +0.001190 | +0.001231 | yes | 반복적으로 더 어려운 cell이나 residual 방향은 반전 |
| count state 7 Brier vs complement | -0.001944 | -0.001707 | -0.000695 | yes | 반복적으로 더 쉬운 cell이나 residual 방향은 반전 |

구조 신호는 존재하지만, 바로 additive correction으로 옮길 근거는 부족하다. Form은 이미
LB에서 alpha 재최적화가 실패했고, count residual cell correction도 과거 LB에서 크게
실패했다. 이번 결과는 그 축을 재제출하라는 의미가 아니라, 다음 round가 있다면 한 신호만
격리한 nested candidate로 재검증할 근거다.

## Model disagreement findings

Disagreement가 큰 row가 반복적으로 더 어려운지에 대한 답은 **아니다**.

| disagreement high-low Brier | 2022 | 2023 | 2024 | stable |
| --- | ---: | ---: | ---: | :---: |
| member std | -0.018554 | +0.011294 | -0.002698 | no |
| member range | -0.020320 | +0.012028 | -0.002571 | no |
| abs(ours-team) | -0.013962 | +0.026770 | -0.004041 | no |
| abs(HGB-Cat) | -0.005299 | +0.004378 | -0.000996 | no |

2023에서만 high-disagreement row의 calibration과 error가 크게 악화했다. 2022와 2024는
반대로 high-disagreement 구간의 Brier가 더 낮다. 따라서 disagreement를 gating/reliability
signal로 사용하는 후보는 만들지 않았다.

## Calibration findings

| subset | 2022 gap | 2023 gap | 2024 gap | finding |
| --- | ---: | ---: | ---: | --- |
| overall | +0.001535 | -0.010530 | -0.003270 | 방향 불일치; 2023 overprediction이 큼 |
| R | +0.001987 | +0.012800 | -0.000925 | 방향 불일치 |
| F | -0.001683 | -0.210208 | -0.020736 | 2023 구조적 shift; 이전 fold로 예측 불가 |
| cold-start | +0.000240 | +0.011737 | -0.003605 | 방향 불일치 |
| unseen pitcher | -0.002820 | -0.029664 | -0.007828 | 세 fold 모두 overprediction, Brier 우열은 불안정 |
| low pitcher reliability | +0.001574 | -0.046391 | -0.013070 | 2022와 이후 방향 불일치 |
| high disagreement | +0.000668 | -0.045788 | -0.004748 | 2023에 편중 |

가장 큰 calibration 문제는 2023 F/low-reliability/high-disagreement에 집중되어 있고,
2022에서는 같은 방향이 아니다. 이전 시즌 residual로 학습한 model이 이 shift를 예측하지
못한 핵심 이유다.

## Residual predictability

각 cell은 `corr / R²`이다. 모든 Huber fit은 iteration 300 이내에서 수렴했다.

| model | 2022 | 2023 | 2024 | pooled correlation | verdict |
| --- | ---: | ---: | ---: | ---: | :---: |
| Ridge | -0.0028 / -0.0122 | -0.0521 / -0.0061 | +0.0086 / -0.0055 | -0.0152 | reject |
| Huber | -0.0033 / -0.1188 | -0.0811 / -0.0103 | +0.0084 / -0.0062 | -0.0249 | reject |
| depth-2 tree | -0.0025 / -0.0120 | -0.0050 / -0.0014 | -0.0063 / -0.0010 | -0.0046 | reject |

R²가 모든 fold/model에서 음수다. 즉 residual mean보다도 일반화가 나쁘며, champion 이후
전체 residual을 현재 feature set으로 예측할 수 있다는 근거가 없다.

## Candidate correction

사전 정의한 세 쌍 `(gamma, cap) = (0.10, .005), (0.25, .01), (0.50, .02)`만
평가했다. 아래는 각 model에서 가장 덜 나쁜 설정이며 모두 `gamma=.10, cap=.005`다.
양수는 champion 대비 Brier improvement다.

| candidate | 2022 | 2023 | 2024 | pooled | verdict |
| --- | ---: | ---: | ---: | ---: | :---: |
| Ridge | +1.107e-6 | -1.068e-4 | +5.357e-6 | -3.295e-5 | reject |
| Huber | -9.737e-6 | -1.743e-4 | +4.612e-6 | -5.898e-5 | reject |
| depth-2 tree | -2.194e-6 | -1.209e-5 | -1.119e-5 | -8.504e-6 | reject |

강한 gamma/cap은 모두 더 악화했다. Passing correction candidate는 0개이며 package나
submission을 만들지 않는다.

## TrackMan decision

TrackMan은 다시 열지 않았다. Unseen/reliability calibration 차이는 관찰됐지만 Brier 난이도와
residual predictability가 fold별로 안정적이지 않아, pitch-characteristic lookup을 추가할
근거가 아직 없다.

## Artifacts

Minimum requested outputs:

- `artifacts/residual_discovery/residual_univariate.csv`
- `artifacts/residual_discovery/residual_categories.csv`
- `artifacts/residual_discovery/disagreement_diagnostics.csv`
- `artifacts/residual_discovery/calibration_diagnostics.csv`
- `artifacts/residual_discovery/residual_model_results.csv`

Additional stability/candidate summaries and `stable_residual_features.md` are saved in the same
directory. Large generated artifacts are gitignored; `scripts/run_residual_discovery.py` reproduces
them.

## Final decision

**A. 안정적인 residual signal 발견 — 다음 round에서 해당 signal 전용 candidate를 개발한다.**

단, residual model 자체는 안정적 개선이 아니며 현재 round에서는 production을 수정하지
않는다. 다음 연구가 진행된다면 form/cold/count를 한꺼번에 섞지 말고, 위 robust signal 중
하나만 사전 등록한 bounded nested candidate로 검증해야 한다. 현재 champion은 그대로
유지한다.

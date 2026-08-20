# Leakage-safe hierarchical target-statistics experiment

작성일: 2026-08-20

## Final verdict

**C. no stable historical interaction signal**

Hierarchical empirical-Bayes posterior는 단순 global prior보다 분명한 feature-only signal을
보였다. 그러나 current `w=.020` champion residual을 season-stable하게 보완하지 못했고,
LightGBM historical expert도 2022/2023/2024 모두 champion보다 나빴다. Stage B gate에서
종료했으며 blend, CatBoost, stacked/profile model, mixture-of-experts와 production ZIP은 만들지
않았다.

## Immutable champion

- LB BSS: `1017.0233029621`
- Artifact: `artifacts/sub_et25_w020.zip`
- SHA-256 before/after:
  `316bf6dab11fb78f06cc74089ea38bdb3b957355f4510db5fcc8a485dc26cecf`
- ET25 weight/tree count: `.0200 / 25`

실험은 champion과 기존 OOF cache를 읽기 전용으로 사용했다.

## Existing feature audit

Champion HGB/CatBoost와 ID-free ET25는 19개의 official `asof_pitcher_*` 및
`asof_batter_*` history feature를 이미 사용한다. NN/team member는 같은 history feature와 raw
pitcher/batter ID도 사용한다. Pitcher season form, batter season form, cold-start LightGBM도
이미 존재한다.

반면 다음 cutoff-static target posterior는 기존 feature contract에 없었다.

- pitcher/batter posterior
- pitcher × batter hand/count/game type
- batter × pitcher hand/count
- pitcher × batter hand × count
- pitcher × base state/inning bucket
- batter × pitcher hand × count
- shrunk pitcher × batter pair

따라서 official simple as-of feature를 복제하지 않고 interaction hierarchy만 추가했다.

## Leakage-safe construction

각 prediction season `S`의 lookup은 `season < S` target만 집계했다.

| Prediction rows | Career history | Recent history |
| --- | --- | --- |
| 2022 validation | `<=2021` | `2021` |
| 2023 validation | `<=2022` | `2022` |
| 2024 validation | `<=2023` | `2023` |
| 2025 test audit | `<=2024` | `2024` |

Historical expert의 training row도 동일하게 season `S` target feature를 만들 때 `<S`만
사용했다. Validation/test row끼리 groupby, rolling, lag, frequency 또는 target update는 없다.

각 hierarchy에는 career/recent 각각 posterior, count, reliability와 parent/career delta를
제공했다. Raw group rate는 사용하지 않았다.

```text
posterior = (successes + M * parent_rate) / (n + M)
reliability = n / (n + M)
```

## Coverage

2024 validation 기준 주요 coverage다.

| Hierarchy | Groups | Median group n | Coverage |
| --- | ---: | ---: | ---: |
| pitcher | 711 | 807.0 | 80.14% |
| batter | 742 | 483.5 | 90.70% |
| pitcher × game type | 1,129 | 233.0 | 79.51% |
| pitcher × batter hand × count | 15,916 | 27.0 | 79.86% |
| batter × pitcher hand × count | 15,387 | 18.0 | 90.41% |
| pitcher × batter | 82,297 | 9.0 | 51.16% |

Exact pitcher–batter pair는 희소해 raw rate를 사용하지 않고 pitcher parent와 `M` prior로
shrink한 진단 feature로만 포함했다.

## Stage A — posterior signal

사전 고정 grid는 `M={25,75,200}`이었다. Career hierarchy 전체의 pooled mean gain vs global을
기준으로 `M=200`을 선택했다.

| M | Mean gain vs global | Median gain | Best gain |
| ---: | ---: | ---: | ---: |
| 25 | `-7.516e-4` | `-7.357e-4` | `-1.023e-5` |
| 75 | `-2.038e-4` | `-8.137e-5` | `+3.695e-4` |
| 200 | `+2.067e-4` | `+4.005e-4` | `+7.047e-4` |

가장 강한 단독 signal은 `M=200`, `pitcher × game_type`, previous-season posterior였다.

| Fold | Posterior Brier | Gain vs global | Gain vs champion | Gap/residual corr |
| ---: | ---: | ---: | ---: | ---: |
| 2022 | `0.246113` | `+3.253e-3` | `-2.998e-3` | `+0.00528` |
| 2023 | `0.252592` | `-1.025e-3` | `+4.35e-4` | `+0.06864` |
| 2024 | `0.250966` | `+9.09e-4` | `-2.902e-3` | `-0.00701` |
| Pooled | `0.249892` | `+1.050e-3` | `-1.836e-3` | `+0.02002` |

Global 대비 pooled signal은 Stage A threshold `+1e-4`를 크게 넘었지만, champion 대비 방향은
2023에만 양수였고 2024 residual correlation도 음수였다. 즉 historical representation에는
정보가 있으나 이미 강한 champion을 안정적으로 교정하는 signal은 아니다.

## Stage B — LightGBM historical expert

Raw player ID를 제외하고 104개 hierarchical posterior/count/reliability feature와 27개
row-local context feature를 사용했다. 고정 LightGBM recipe 하나만 학습했고 tuning은 하지
않았다.

| Fold | Expert Brier | Champion Brier | Gain vs champion |
| ---: | ---: | ---: | ---: |
| 2022 | `0.243785` | `0.243115` | `-6.697e-4` |
| 2023 | `0.254968` | `0.253027` | `-1.941e-3` |
| 2024 | `0.248746` | `0.248065` | `-6.815e-4` |
| Pooled | `0.249148` | `0.248056` | `-1.092e-3` |

Complementarity diagnostics:

| Fold | Corr(expert, champion) | Gap/residual corr | Top-10% champion-error wins |
| ---: | ---: | ---: | ---: |
| 2022 | `0.9194` | `+0.01074` | `80.05%` |
| 2023 | `0.9046` | `-0.02163` | `48.34%` |
| 2024 | `0.6464` | `+0.00455` | `61.35%` |
| Pooled | `0.8904` | `-0.00301` | `67.63%` |

Prediction diversity와 high-error-row 승률은 있었지만 pooled residual direction이 음수이고
2023 complementarity가 실패했다. 단독 Brier도 세 fold 모두 악화했다. 따라서 stable-signal
gate를 통과하지 못했다.

## Stage gates

| Stage | Result | Reason |
| --- | --- | --- |
| A — hierarchy signal | pass | Best gain vs global `+1.050e-3` |
| B — historical expert | fail | All-fold standalone loss; pooled gap/residual corr `-0.00301` |
| C — fixed blend | not run | Stage B failure |
| D — profile/stack/MoE | not run | Stage B/C failure |

따라서 `w={.02,.05,.10,.15}` blend를 평가하지 않았다. CatBoost target-encoding 중복 ablation,
stacked logistic/ridge, profile MLP와 reliability gating도 실행하지 않았다. 이 축에서 작은 양수만
골라내기 위한 추가 model/weight 탐색은 하지 않는다.

## Leakage and row-independence audits

- 2024 validation target 2,000개를 전부 flip: feature max diff `0.0`
- Validation row 하나 제거 후 다른 validation feature: max diff `0.0`
- Test single/full/shuffle/subset/reverse historical expert prediction: max diff `0.0`
- Test lookup source: train `<=2024` only
- Current champion SHA before/after: identical

## Production decision

Strong-signal 기준인 stable pooled blend gain `>=2e-5`에 도달하지 못했고 stable blend 자체가
없다. Production ZIP과 leaderboard candidate는 생성하지 않았다. 기존
`artifacts/sub_et25_w020.zip`이 immutable champion으로 유지된다.

## Artifacts

- `artifacts/hier_target/hierarchy_coverage.csv`
- `artifacts/hier_target/posterior_brier.csv`
- `artifacts/hier_target/residual_signal.csv`
- `artifacts/hier_target/expert_results.csv`
- `artifacts/hier_target/complementarity.csv`
- `artifacts/hier_target/blend_results.csv`
- `artifacts/hier_target/profile_results.csv`
- `artifacts/hier_target/summary.json`

Reproduction:

```bash
cd /Users/wooh/Documents/dev/LG-Aimers-9th
.venv/bin/python scripts/validate_hier_target.py
.venv/bin/python -m unittest tests.test_hier_target
```

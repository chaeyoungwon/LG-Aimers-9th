# TrackMan signal ablation

Date: 2026-08-21

## Verdict

**B. TRACKMAN SIGNAL REAL BUT BASELINE IS BEST**

Accepted Hungarian HIGH mapping과 현재 champion을 동결한 채 TrackMan signal을 분해했다. `ALL PHYSICAL`은 pooled gain이 기존 TrackMan baseline보다 컸지만 주요 subset safety gate를 크게 위반했다. 작고 해석 가능한 family 및 confidence/history adaptive weighting은 기존 baseline을 넘지 못했다. 따라서 production ZIP은 생성하지 않았고 `artifacts/sub_et25_w020.zip`을 유지한다.

## Frozen references

- Champion: `artifacts/sub_et25_w020.zip`
- Champion SHA-256: `316bf6dab11fb78f06cc74089ea38bdb3b957355f4510db5fcc8a485dc26cecf`
- Mapping: Hungarian, HIGH only
- Temporal folds: train through 2021/2022/2023, validate 2022/2023/2024
- Model: accepted raw52 + ET25 recipe; family와 blend weight 외 변경 없음
- Fixed weights: `.05`, `.10`, `.15`, `.20`
- Adaptive weights: train-reference ECDF로 고정된 `.05`–`.15`

기존 `ALL TRACKMAN w=.10` gain을 오차 `1.7e-21` 이하로 재현했다.

| Fold | Reconstructed gain |
|---|---:|
| 2022 | 1.2124564658e-5 |
| 2023 | 2.6392871962e-5 |
| 2024 | 2.1054167049e-5 |
| pooled | 1.9849831856e-5 |

## Seven prior residual features

통과한 7개 중 6개는 cutoff 누적 usage count이고 물리량은 `fastball_offspeed_velocity_gap` 하나뿐이었다.

| Feature | Class | Residual corr 2022 | 2023 | 2024 |
|---|---|---:|---:|---:|
| `count_two_strike_n` | usage/history | .002666 | .041143 | .015948 |
| `fastball_n` | usage/history | .003723 | .040411 | .013507 |
| `tm_history_n` | usage/history | .002599 | .038161 | .014506 |
| `count_neutral_n` | usage/history | .002465 | .037496 | .014334 |
| `offspeed_n` | usage/history | -.000757 | .030690 | .017264 |
| `fastball_offspeed_velocity_gap` | physical/separation | .005724 | .029822 | .011170 |
| `count_three_ball_n` | usage/history | .003216 | .032067 | .010661 |

모든 feature는 matched 211,513 validation rows와 215 season별 pitcher 관측에서 missingness 0, pitcher coverage 1.0이었다. 정확한 정의와 coverage는 `feature_inventory.csv`에 기록했다.

## Family results

아래는 각 family에서 사전 정의 weight 중 pooled gain이 가장 컸던 결과다. Gain은 전체 validation rows에서 champion 대비 값이다.

| Expert | w | 2022 | 2023 | 2024 | pooled |
|---|---:|---:|---:|---:|---:|
| ALL PHYSICAL | .15 | 0.998e-6 | 54.954e-6 | 22.245e-6 | 25.959e-6 |
| ALL TRACKMAN | .15 | 12.239e-6 | 32.884e-6 | 23.894e-6 | 22.987e-6 |
| usage + FO gap | .15 | 4.078e-6 | 21.351e-6 | 12.241e-6 | 12.531e-6 |
| history count only | .15 | 8.816e-6 | 22.683e-6 | 3.495e-6 | 11.570e-6 |
| release/mechanics | .10 | 5.770e-6 | 10.100e-6 | 16.714e-6 | 10.911e-6 |
| movement | .10 | 7.748e-6 | 16.731e-6 | 5.837e-6 | 10.054e-6 |
| pitch separation | .10 | 2.113e-6 | 10.220e-6 | 15.831e-6 | 9.438e-6 |
| usage/history | .10 | 0.496e-6 | 16.854e-6 | 2.431e-6 | 6.533e-6 |
| velocity | .10 | -3.456e-6 | 10.314e-6 | 8.930e-6 | 5.279e-6 |

`ALL PHYSICAL w=.15`는 pooled에서 baseline보다 `+6.109e-6`였지만 2022 full-count `-1.256e-4`, 2024 F `-1.053e-4`, 2022 high-C `-1.991e-5`로 safety floor `-1e-5`를 위반했다. `ALL TRACKMAN w=.15`도 pooled는 baseline보다 `+3.137e-6`였으나 2024 F `-1.078e-4`, 2022 full-count `-8.561e-5`, high-C `-3.613e-5`였다. 따라서 둘 다 reject했다.

Family removal delta는 `blend_results.csv`의 `family_removal_delta` 열에 동일 weight의 ALL TRACKMAN 대비로 저장했다. 물리 feature만 남기면 pooled 평균은 개선될 수 있지만 그 효과가 fold/subset에 고르게 분포하지 않았다. 반대로 usage-only 및 `usage + FO gap`은 더 안전한 방향이지만 baseline signal의 절반가량만 유지했다.

## Usage, coverage, confidence

대부분의 cumulative TrackMan usage는 `log1p(asof_pitcher_n)` 및 `log1p(asof_pitcher_pitchmix_n)`과 높은 상관을 보였다(`tm_history_n` .903–.952, count buckets 약 .875–.952). 통제 후 residual association은 2023/2024에는 양수였지만 2022에는 대부분 0 또는 음수였다. 예외적으로 recent count는 main reliability와 상관이 .196–.374로 낮았지만 단독 pooled gain은 `8.102e-6`에 그쳤다. 즉 usage volume 일부는 새 정보이나 안정적인 최소 expert로 충분하지 않았다.

Coverage-only `raw52 + tm_matched` hard model은 2022/2023/2024 각각 `-2.102e-3`, `-1.698e-3`, `-2.176e-3`였다. 따라서 기존 `+1.985e-5`를 단순한 matched indicator 효과로 설명할 수 없다.

Raw ALL TRACKMAN expert 자체는 모든 HIGH confidence tercile에서 champion보다 나빴고, small blend가 이를 완화하는 구조였다. Confidence-linear와 history-linear pooled gain은 각각 `1.9148e-5`, `1.8793e-5`로 constant `.10`보다 낮았다. Advantage quantile도 season 간 단조 관계가 재현되지 않아 threshold나 learned gating을 정당화하지 못한다.

## Leakage and independence

- 각 fold의 TrackMan profile은 validation 직전 cutoff까지만 사용했다.
- 실제 validation 표본 257행/season에서 single, full, shuffle, subset, reverse feature max diff가 모두 `0.0`이었다.
- 같은 표본의 validation target을 뒤집어도 feature max diff는 모든 season에서 `0.0`이었다.
- test data는 읽지 않았다.
- 외부 identity 정보는 사용하지 않았다.

## Decision

세 fold 양수이면서 pooled baseline을 넘는 조합은 있었지만 major subset 및 confidence safety를 함께 통과한 후보는 0개였다. 따라서 새 ZIP을 만들지 않고 TrackMan signal purification 축을 B로 종료한다.

Reproduce:

```bash
.venv/bin/python scripts/validate_tm_signal.py
.venv/bin/python -m unittest tests.test_tm_signal tests.test_tm_crosswalk
```

Artifacts:

- `artifacts/tm_signal_ablation/feature_inventory.csv`
- `artifacts/tm_signal_ablation/family_oof.csv`
- `artifacts/tm_signal_ablation/usage_ablation.csv`
- `artifacts/tm_signal_ablation/coverage_diagnostics.csv`
- `artifacts/tm_signal_ablation/confidence_diagnostics.csv`
- `artifacts/tm_signal_ablation/advantage_quantiles.csv`
- `artifacts/tm_signal_ablation/blend_results.csv`
- `artifacts/tm_signal_ablation/summary.json`

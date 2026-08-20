# Pitcher Control State/Profile representation experiment

실험일: 2026-08-21

## 결론

**C. no stable control-profile representation signal**

요청한 A~G family를 각각 raw HGB-52 feature contract에 독립적으로 추가하고,
동일한 25-tree ExtraTrees recipe로 temporal OOF를 수행했다. 일곱 family 모두
`UNSTABLE`이었다. 따라서 사전 정의된 stage gate에 따라 complementarity, blend,
full-train production build는 실행하지 않았다.

현재 immutable champion은 변경되지 않았다.

- artifact: `artifacts/sub_et25_w020.zip`
- LB BSS: `1017.0233029621`
- SHA-256: `316bf6dab11fb78f06cc74089ea38bdb3b957355f4510db5fcc8a485dc26cecf`

## 비교 계약

두 reference를 구분했다.

1. **Family representation gain**은 accepted raw HGB-52 ET25와 비교한다. 동일한
   learner에서 family 추가 자체의 효과를 분리하기 위해서다.
2. **Complementarity와 blend**는 현재 immutable champion과 비교한다. 현재
   champion OOF는 `0.98 * original champion + 0.02 * accepted raw ET25`로
   재구성했고, 기존 w=.020 결과와 2022/2023/2024/pooled 모두 절대 오차
   `1e-15` 이하로 일치했다.

`feature_gain_vs_raw_et25`의 부호는
`Brier(raw ET25) - Brier(domain ET25)`이며 양수가 개선이다. 단독 domain ET25는
champion을 이기는 것이 목적이 아니므로, champion 대비 Brier도 별도 열로 기록했다.

## 중복 감사

현재 champion의 HGB/Cat/NN 및 team LightGBM feature contract를 직접 확인했다.
다음 요청 feature는 exact duplicate라 추가하지 않았다.

| 요청 representation | 기존 feature |
|---|---|
| `control_margin` | `success_minus_middle` |
| `strike_ball_balance` | `strike_minus_ball` |
| `success_dev_5` | `prev_vs_career`, `pitcher_rate_gap_career_5` |
| `success_1v5` | `pitcher_rate_gap_1_5` |
| `log_pitcher_n` | `log1p_asof_pitcher_n` |
| `log_pitchmix_n` | `log1p_asof_pitcher_pitchmix_n` |

`prev1-career`, `prev3-career`, `prev1-prev3`, `prev3-prev5`와 모든 middle-rate
차이는 기존 exact feature가 없어 평가했다. 상세 감사 결과는
`artifacts/control_profile/feature_audit.csv`에 있다.

## Family 정의

각 family는 한꺼번에 결합하지 않고 독립적으로 평가했다.

- A — long-term control: `middle-ball`, `success-reverse`
- B — multi-timescale form: 중복을 제외한 success gap, 모든 middle gap, sign
  direction, `dev3 × log1p(pitcher_n)`
- C — deterioration type: `success_dev3`와 `middle_dev3` 방향 조합 및 full-count
  interaction
- D — recent consistency: 1/3/5 monotonic direction, range, joint instability,
  three-ball interaction
- E — pitcher-batter relative: success/middle profile gap
- F — pitch mix: primary share, normalized entropy/HHI, coverage, 두 rate gap 및
  제한된 control/count interaction
- G — control state × leverage: `log1p(li)`, fixed close/late-close, cutoff-train
  LI 75% quantile, form/instability/type interaction

방향 feature는 threshold tuning 없이 0 기준 sign을 사용했다. `close_game`은
`abs(score_diff_pitcher_team) <= 1`, `late_close`는 inning 7 이상으로 고정했다.
유일한 학습 기반 feature threshold인 high-leverage LI 75% quantile은 각 fold의
cutoff-training row에서만 적합했다.

결측 recent/profile 값은 그대로 비유한 값으로 보존한 뒤 cutoff-training median으로
처리하며, availability indicator를 함께 제공했다. 범주형 unknown은 train-only
mapping의 `-1`로 처리했다. cold-start에서 무한대가 발생하지 않는다.

## 모델과 fold

모든 family에 동일한 모델 하나만 사용했다.

```text
ExtraTreesClassifier(
    n_estimators=25,
    max_depth=None,
    min_samples_leaf=8,
    max_features="sqrt",
    bootstrap=False,
    n_jobs=-1,
    random_state=42,
)
```

- season < 2022 → validation 2022
- season < 2023 → validation 2023
- season < 2024 → validation 2024

새 seed, tree count, calibration, hyperparameter search를 수행하지 않았다.

## Temporal OOF 결과

아래 수치는 raw HGB-52 ET25 대비 family 추가 gain이다.

| Family | 2022 | 2023 | 2024 | Pooled | Grade |
|---|---:|---:|---:|---:|---|
| A long-term control | +1.101637e-4 | -6.170120e-5 | -1.199018e-4 | -2.449111e-5 | UNSTABLE |
| B multi-timescale form | -2.032160e-4 | +2.130728e-4 | -5.923905e-4 | -1.984594e-4 | UNSTABLE |
| C deterioration type | +1.693072e-4 | -1.921794e-5 | -1.182865e-4 | +9.636708e-6 | UNSTABLE |
| D recent consistency | +6.092106e-6 | +2.319770e-4 | -2.394585e-4 | -3.001500e-6 | UNSTABLE |
| E pitcher-batter relative | -8.772241e-6 | +8.454005e-5 | -2.144122e-4 | -4.791560e-5 | UNSTABLE |
| F pitch mix | -1.689115e-4 | -2.590944e-4 | -3.494192e-4 | -2.598717e-4 | UNSTABLE |
| G control state × leverage | -1.866913e-4 | +1.895819e-4 | -1.214223e-4 | -4.077028e-5 | UNSTABLE |

등급은 사전 정의 그대로 적용했다.

- STRONG: 2023 > 0, 2024 > 0, pooled > `1e-5`
- WEAK_STABLE: pooled > 0, 3 fold 중 2개 이상 양수, worst fold > `-2e-5`
- 그 외: UNSTABLE

C는 pooled가 `+9.637e-6`이지만 2023과 2024가 모두 음수이고 worst fold가
`-1.183e-4`라 WEAK_STABLE이 아니다. D는 두 fold가 양수지만 pooled가 음수이고
2024 손실이 `-2.395e-4`다. F는 세 fold 모두 음수였다. 무엇보다 일곱 family가
모두 2024에서 음수여서 최근 season으로 일반화되는 representation을 찾지 못했다.

## Complementarity 및 blend gate

STRONG 또는 WEAK_STABLE family가 하나도 없었다. 따라서 다음 분석은 수행하지
않았으며 CSV는 schema만 가진 빈 파일이다.

- champion correlation
- champion top-10% error win rate
- residual correlation
- champion + domain ET25 weights `0.005/0.010/0.020/0.050`

이는 weak한 단독 모델을 임의 blend로 구제하거나 weight를 탐색하지 않기 위한
stage gate의 의도된 동작이다.

## Leakage 및 재현성 감사

- feature source: current row와 공식 `asof_*`만 사용
- validation target flip changed cells: `0`
- single/full/shuffle row-local feature parity: 통과
- 2024 state-fit max season: `2023`
- test 데이터 read: `false`
- raw IDs 및 target의 model feature 유입: 없음
- current champion SHA-256 전후 동일

## 산출물

- `scripts/validate_control_profile.py`
- `artifacts/control_profile/feature_audit.csv`
- `artifacts/control_profile/family_oof.csv`
- `artifacts/control_profile/complementarity.csv`
- `artifacts/control_profile/blend_results.csv`
- `artifacts/control_profile/summary.json`

실행:

```bash
.venv/bin/python scripts/validate_control_profile.py
```

fold prediction cache는 `artifacts/control_profile/cache/`에 저장된다. cache를
무시하고 같은 frozen recipe를 다시 학습하려면 `--force`를 사용한다.

## 최종 판단

현재 raw 52 feature에는 요청한 장기/최근 rate의 원재료가 이미 모두 들어 있으며,
차이·방향·상태·interaction을 명시화해도 ET25의 최신 season decision boundary는
개선되지 않았다. 이 결과에 따라 control-profile representation 축은 production으로
넘기지 않고, 현재 `sub_et25_w020.zip`을 유지한다.

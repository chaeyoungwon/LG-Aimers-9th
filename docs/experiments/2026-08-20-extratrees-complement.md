# ExtraTrees complementary learner validation

작성일: 2026-08-20

## Final verdict

**A. Stable complementary signal found**

ID-free ExtraTrees의 raw 확률은 champion 단독 성능보다 낮았지만, champion과 다른 error를
만들었고 `w=0.01` small blend가 2022/2023/2024와 pooled에서 모두 양의 Brier gain을 냈다.
사전 정의 subset loss 한도와 tiny-segment concentration 조건도 통과했다. 이는 다음 bounded
round에서 production-equivalent package를 검토할 근거이며, 이번 round에서는 champion,
production ZIP 또는 leaderboard 제출물을 만들거나 변경하지 않았다.

## Frozen reference

- champion LB BSS: `1016.4442212358`
- artifact: `artifacts/submit_r10_pitcher_w418194.zip`
- SHA-256 before/after:
  `ce10c428c9b6bd88ebf713fcc05b91ce22fc555573f3d11f8deb283637484c47`
- test rows read: no
- leaderboard probe: no

## Hypothesis and protocol

ExtraTrees의 bagging과 random split threshold가 boosting 중심 champion과 다른 inductive bias를
만들어, 단독 Brier가 더 낮지 않더라도 row-level error complementarity를 제공하는지 검증했다.
기존 global/regime weight, residual, form, cold-start, calibration 및 TrackMan 축은 다시 열지
않았다.

Temporal fold는 기존 reconstructed champion OOF와 정확히 맞췄다.

| Train cutoff | Validation | Train rows | Validation rows |
| ---: | ---: | ---: | ---: |
| ≤2021 | 2022 | 728,588 | 247,472 |
| ≤2022 | 2023 | 976,060 | 245,525 |
| ≤2023 | 2024 | 1,221,585 | 253,507 |

각 fold의 category map과 numeric missing median은 해당 cutoff train에서만 계산했다. Unknown과
missing category는 `-1`로 encoding했다. Validation label은 ExtraTrees fitting, feature
encoding, imputation 또는 어떤 training statistic에도 사용하지 않고 metric과 사전 정의
gate 계산에만 사용했다.

## Feature and model contract

Champion HGB/CatBoost의 leakage-safe 52열 row-local feature contract를 순서까지 그대로
재사용했다. `row_id`, `control_success`, `pitcher_id`, `batter_id`는 제외했다. Raw player ID
포함/제외 두 representation을 비교하지 않았고 새 feature도 만들지 않았다.

최대 fold가 122만 행이고 host가 16 GiB이므로 사용자 지시에서 허용한 resource fallback인
300 trees를 시작 전에 선택했다. 사용한 모델은 하나뿐이며 grid search나 seed ensemble은
없다.

```python
ExtraTreesClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=8,
    max_features="sqrt",
    bootstrap=False,
    n_jobs=-1,
    random_state=42,
)
```

## Standalone probability quality

| Validation | ExtraTrees Brier | Champion Brier | pred mean | pred std | min | max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 0.243760788 | 0.243116332 | 0.531596 | 0.066405 | 0.328962 | 0.816101 |
| 2023 | 0.253063393 | 0.253041573 | 0.521531 | 0.063496 | 0.332773 | 0.796560 |
| 2024 | 0.248667790 | 0.248069141 | 0.501614 | 0.038953 | 0.322811 | 0.678486 |

ExtraTrees는 champion을 단독으로 이기지 못했다. 이 실험의 채택 근거는 단독 score가 아니라
아래 correlation/error complementarity와 bounded blend다.

## Prediction correlation

| Validation | corr(extra, champion) | corr(extra, ours stage) | corr(extra, team stage) |
| ---: | ---: | ---: | ---: |
| 2022 | 0.948155 | 0.955007 | 0.951582 |
| 2023 | 0.941174 | 0.951690 | 0.947582 |
| 2024 | 0.752723 | 0.795196 | 0.775939 |
| pooled | 0.918843 | 0.930540 | 0.901445 |

모든 fold의 champion correlation이 사전 high-correlation cutoff `0.98`보다 낮았다. 특히
2024에서 prediction diversity가 커졌다.

## Error complementarity

`ExtraTrees wins`는 같은 validation row에서 ExtraTrees squared error가 champion보다 작은
비율이다. Top-error subset은 champion error로 validation에서만 진단했으며 test-time gate로
사용하지 않는다.

| Validation | All rows | Champion error top 10% | Champion error top 20% |
| ---: | ---: | ---: | ---: |
| 2022 | 46.45% | 87.38% | 81.74% |
| 2023 | 48.67% | 74.74% | 78.36% |
| 2024 | 49.85% | 57.93% | 59.51% |
| pooled | 48.33% | 78.27% | 74.50% |

세 fold 모두 overall win-rate `>20%`, champion top-10%/top-20% error win-rate `>50%`를
만족해 raw complementarity gate를 통과했다.

## Preregistered small blend

`p_new = (1-w) * champion + w * ExtraTrees`만 사용했다. 별도 calibration이나 weight
optimization은 하지 않았다. 양수는 champion 대비 Brier improvement다.

| w | 2022 | 2023 | 2024 | pooled | Safety verdict |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 0.01 | +1.130e-6 | +7.195e-6 | +2.010e-6 | +3.424e-6 | **pass** |
| 0.02 | +2.107e-6 | +1.424e-5 | +3.859e-6 | +6.693e-6 | reject: 2024 F `-1.362e-5` |
| 0.03 | +2.932e-6 | +2.114e-5 | +5.546e-6 | +9.807e-6 | reject: 2024 F `-2.099e-5` |
| 0.05 | +4.121e-6 | +3.448e-5 | +8.435e-6 | +1.557e-5 | reject: 2024 F `-3.688e-5` |
| 0.08 | +4.757e-6 | +5.337e-5 | +1.156e-5 | +2.306e-5 | reject: 2024 F `-6.356e-5` |

전체 fold gain은 weight가 커질수록 증가했지만, 사전 정의한 major-subset loss floor
`-1e-5` 때문에 `w>=.02`는 모두 기각했다. 따라서 결과를 보고 weight를 더 탐색하지 않고
등록 후보 중 가장 작은 `w=.01`만 통과로 표시했다.

## Subset diagnostics for w=0.01

Positive/negative form은 pitcher form adjustment 부호이며 exact zero row는 두 subset에서
제외했다. Low/high pitcher history 경계는 validation 분포가 아니라 각 cutoff train의
`asof_pitcher_n` median(`1148 / 1375 / 1618`)이다.

| Subset | 2022 | 2023 | 2024 | pooled |
| --- | ---: | ---: | ---: | ---: |
| R | +9.079e-7 | +4.905e-6 | +3.169e-6 | +3.004e-6 |
| F | +2.715e-6 | +2.680e-5 | -6.619e-6 | +6.645e-6 |
| seen pitcher | +2.371e-7 | +7.318e-6 | +3.074e-6 | +3.609e-6 |
| unseen pitcher | +4.405e-6 | +6.576e-6 | -1.599e-6 | +2.695e-6 |
| full count | -1.623e-6 | -2.040e-7 | -1.859e-6 | -1.225e-6 |
| non-full count | +1.268e-6 | +7.572e-6 | +2.199e-6 | +3.656e-6 |
| positive form | +6.336e-7 | +2.664e-6 | +5.563e-6 | +2.976e-6 |
| negative form | +9.738e-7 | +6.891e-6 | +2.691e-6 | +3.738e-6 |
| low pitcher history | +1.866e-6 | +6.143e-6 | -1.715e-6 | +1.664e-6 |
| high pitcher history | +8.528e-7 | +7.531e-6 | +3.813e-6 | +4.112e-6 |

모든 `n>=5,000` subset이 loss floor `-1e-5` 이상이다. 20% 이하의 작은 사전 정의
segment가 전체 positive gain의 80% 이상을 단독 설명하는 경우도 0개였다. Full-count의
미세한 음수는 세 fold에 반복되므로 다음 package round에서도 별도 안전성 대상으로 유지해야
하지만 현재 rejection threshold를 넘지는 않는다.

## Conditional RandomForest comparison

ExtraTrees가 안정 조건을 통과했기 때문에만 같은 52열, 300 trees, leaf 8, sqrt, seed 42의
`RandomForestClassifier(bootstrap=True)` 한 모델을 비교했다. Grid/seed 실험은 없었다.

| Validation | RF Brier | Champion Brier | corr(RF, champion) | RF small-blend direction |
| ---: | ---: | ---: | ---: | --- |
| 2022 | 0.244565265 | 0.243116332 | 0.892096 | `w=.01`: -1.863e-7 |
| 2023 | 0.252871839 | 0.253041573 | 0.875714 | `w=.01`: +1.559e-5 |
| 2024 | 0.249620589 | 0.248069141 | 0.625469 | `w=.01`: -3.182e-6 |

RF도 high-error complementarity gate는 통과했지만 모든 blend weight에서 2024 gain이
음수였다. `w=.01`의 unseen-pitcher 2024 gain도 `-1.359e-5`로 subset floor를 위반했다.
Passing RF weight는 0개이며 판정은 **B. Weak complementary signal, not deployment-worthy**다.
RF 신호는 2023에 집중되어 종료한다. ExtraTrees fit time 합계는 약 `368s`, RF는 약
`1,252s`로 RF가 약 3.4배 더 비쌌다.

ExtraTrees가 이미 안정 신호를 보였으므로 XGBoost는 후순위 조건에 해당하지 않아 실행하지
않았다.

## Safety and artifacts

- Validation season label은 해당 fold ExtraTrees/RF 학습이나 preprocessing에 사용하지 않았다.
- Raw player ID, test row, TrackMan, external data를 사용하지 않았다.
- Champion 후처리, calibration, residual, regime, form, cold-start를 변경하지 않았다.
- Champion checksum은 실험 전후 동일하다.
- Production/submission package와 leaderboard 제출물을 만들지 않았다.

생성 artifact:

- `artifacts/extratrees_complement/member_results.csv`
- `artifacts/extratrees_complement/correlation.csv`
- `artifacts/extratrees_complement/complementarity.csv`
- `artifacts/extratrees_complement/blend_results.csv`
- `artifacts/extratrees_complement/subset_results.csv`
- `artifacts/extratrees_complement/summary.json`

조건부 RF 결과는 같은 디렉터리의 `random_forest_*.csv`에 분리했다. Fold prediction cache는
재실행 시 같은 단일 모델을 다시 학습하지 않기 위한 research-only artifact다.

## Reproduction

```bash
cd /Users/wooh/Documents/dev/LG-Aimers-9th
.venv/bin/python scripts/validate_extratrees_complement.py \
  --n-estimators 300 \
  --n-jobs -1
.venv/bin/python -m unittest tests.test_extratrees_complement
shasum -a 256 artifacts/submit_r10_pitcher_w418194.zip
```

다음 round를 연다면 오직 current champion + frozen ExtraTrees 1%의 production-equivalent
parity를 검증한다. 이 문서는 새로운 weight search, calibration 또는 leaderboard probing을
허가하지 않는다.

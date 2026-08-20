# ExtraTrees ordered-prefix size ablation

작성일: 2026-08-20

## Final verdict

**A. COMPRESSED CANDIDATE READY FOR ONE LB SUBMISSION**

Accepted 300-tree ExtraTrees의 ordered estimator prefix만 평가했다. 25 trees가 세 temporal fold와
pooled에서 모두 양의 Brier gain을 유지하고, pooled gain은 300-tree reference의 `103.63%`,
worst eligible subset gain은 `-2.096e-6`로 safety floor `-1e-5`를 통과했다. 통과 후보 중 가장
작은 25 trees를 선택했으며 다른 tree count의 production ZIP은 만들지 않았다.

최종 25-tree candidate는 ZIP `121,125,119 bytes`, 253,507행 peak RSS
`2,250,194,944 bytes`, runtime `10.384s`로 내부 목표를 모두 통과했다.

## Frozen references

- Champion: `artifacts/submit_r10_pitcher_w418194.zip`
- Champion SHA-256 before/after:
  `ce10c428c9b6bd88ebf713fcc05b91ce22fc555573f3d11f8deb283637484c47`
- Accepted 300-tree candidate: `artifacts/submit_r10_pitcher_w418194_extra01.zip`
- 300-tree candidate SHA-256 before/after:
  `42658ee044671925cf8c08f1d26c127f92b6ff587bfc038cf59915e3862965b9`
- Blend: `0.99 * champion + 0.01 * ExtraTrees prefix`
- Tree counts: `25 / 50 / 100 / 200 / 300`
- Seed: `42`

Champion과 기존 300-tree candidate는 읽기 전용으로 사용했으며 둘의 checksum은 작업 전후
동일하다. 500-tree model, 별도 seed, weight 탐색, calibration, RF, XGBoost 또는 leaderboard
probe는 실행하지 않았다.

## Method

OOF estimator 객체는 이전 실험에서 저장하지 않고 prediction cache만 남겼다. 따라서 각
temporal fold에서 accepted recipe의 300-tree forest를 정확히 한 번 재구성했다. 각 후보는
동일한 fitted model에서 다음 순서 고정 prefix로 계산했다.

```python
prefix.estimators_ = model.estimators_[:tree_count]
prefix.n_estimators = tree_count
```

Tree count별 별도 fit은 없다. Feature는 accepted ID-free HGB/CatBoost 52열이고 fold별 category
map과 numeric median은 cutoff train에서만 계산했다. Blend weight는 모든 후보에서 정확히
1%다.

| Train cutoff | Validation | 300-tree fit time | Accepted prediction max diff |
| ---: | ---: | ---: | ---: |
| ≤2021 | 2022 | `79.0s` | `3.331e-16` |
| ≤2022 | 2023 | `119.2s` | `2.776e-16` |
| ≤2023 | 2024 | `161.8s` | `3.331e-16` |

모든 reconstructed 300-tree prediction이 기존 accepted OOF cache와 `1e-12` 이내에서
일치했다.

## Temporal OOF gains

양수는 frozen champion 대비 Brier 개선이다. Pooled retention은 300-tree pooled gain
`+3.423916159345e-6` 대비 비율이다.

| Trees | 2022 | 2023 | 2024 | Pooled | Pooled retention | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **25** | `+8.623e-7` | `+7.400e-6` | `+2.440e-6` | `+3.548e-6` | `103.63%` | **pass / selected** |
| 50 | `+9.524e-7` | `+7.439e-6` | `+2.041e-6` | `+3.455e-6` | `100.92%` | pass |
| 100 | `+9.809e-7` | `+7.112e-6` | `+2.447e-6` | `+3.495e-6` | `102.08%` | pass |
| 200 | `+8.890e-7` | `+7.609e-6` | `+2.140e-6` | `+3.524e-6` | `102.92%` | pass |
| 300 | `+1.130e-6` | `+7.195e-6` | `+2.010e-6` | `+3.424e-6` | `100.00%` | reference |

다섯 후보가 모두 사전 gate를 통과했다. Gain 최대 후보를 선택하지 않고, 사용자 지시대로 가장
작은 25 trees를 선택했다. 25-tree pooled gain이 300보다 소폭 큰 것은 prefix noise 범위의
진단 결과일 뿐 추가 tree-count나 weight 탐색의 근거로 사용하지 않는다.

## Prediction convergence and complementarity

아래는 세 fold를 합친 pooled 진단이다. Top-10%는 champion squared error가 가장 큰 validation
row에서 ExtraTrees가 champion보다 작은 squared error를 보인 비율이다.

| Trees | corr vs 300 | MAE vs 300 | Max diff vs 300 | corr vs champion | Top-10% win rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | `0.839322` | `0.030677` | `0.172941` | `0.773684` | `66.04%` |
| 50 | `0.914823` | `0.020739` | `0.124829` | `0.841323` | `70.47%` |
| 100 | `0.963378` | `0.013101` | `0.085420` | `0.886534` | `74.13%` |
| 200 | `0.990318` | `0.006569` | `0.040659` | `0.910188` | `77.35%` |
| 300 | `1.000000` | `0.000000` | `0.000000` | `0.918843` | `78.27%` |

25-tree prefix는 300-tree 예측과의 수렴도는 가장 낮지만 champion correlation도 더 낮고,
champion high-error row win rate가 `66.04%`여서 complementary structure가 남아 있다. 25-tree
fold별 top-10% win rate는 2022 `69.94%`, 2023 `66.15%`, 2024 `52.83%`다.

## Subset safety for 25 trees

모든 표의 row 수는 5,000 이상이므로 safety gate 대상이다.

| Subset | 2022 | 2023 | 2024 | Pooled |
| --- | ---: | ---: | ---: | ---: |
| R | `+8.003e-7` | `+3.613e-6` | `+2.784e-6` | `+2.408e-6` |
| F | `+1.304e-6` | `+3.981e-5` | `-1.201e-7` | `+1.229e-5` |
| seen pitcher | `-5.034e-8` | `+7.588e-6` | `+3.776e-6` | `+3.839e-6` |
| unseen pitcher | `+4.208e-6` | `+6.446e-6` | `-2.096e-6` | `+2.402e-6` |
| full count | `+3.959e-7` | `+2.190e-6` | `+3.569e-6` | `+2.051e-6` |
| non-full count | `+8.856e-7` | `+7.665e-6` | `+2.385e-6` | `+3.623e-6` |
| positive form | `+2.365e-8` | `+2.603e-6` | `+5.809e-6` | `+2.832e-6` |
| negative form | `+8.226e-7` | `+4.886e-6` | `+2.543e-6` | `+2.883e-6` |

Worst eligible subset은 2024 unseen pitcher의 `-2.096e-6`이며 rejection floor
`-1e-5`보다 충분히 높다.

## Model and package size curve

Model size는 기존 full-train 300 model의 ordered prefix를 joblib zlib level 3으로 실제
직렬화해 측정했다. ZIP estimate는 동일 champion packaging overhead를 더한 값이다. 선택된
25-tree ZIP만 실제 생성했다.

| Trees | Actual model size | ZIP estimate | Actual candidate ZIP |
| ---: | ---: | ---: | ---: |
| **25** | `117,775,849 B` | `121,125,048 B` | `121,125,119 B` |
| 50 | `234,206,869 B` | `237,556,068 B` | — |
| 100 | `467,106,130 B` | `470,455,329 B` | — |
| 200 | `932,961,866 B` | `936,311,065 B` | — |
| 300 | `1,398,488,651 B` | `1,401,837,850 B` | 기존 실측 `1,400,529,259 B` |

## Production parity and audit

선택된 full-train model은 기존 accepted full 300-tree joblib의
`estimators_[:25]`만 직렬화했다. 재학습하지 않았다.

- Serialized reload prediction max diff: `5.551e-17`
- Production implementation vs OOF experiment max diff:
  - 2022: `3.331e-16`
  - 2023: `3.331e-16`
  - 2024: `3.331e-16`
- Candidate 내부 champion component vs 원본 champion max diff: `0.0`
- 최종 candidate single/full/shuffle/subset/reverse max diff: `0.0`
- ExtraTrees 내부 batch-order max diff: `1.110e-16` (`atol=1e-12`)
- 격리 실행 3회 output: byte-identical, prediction max diff `0.0`
- NaN/Inf: 없음, probability bounds: 통과
- Requirements: 기존 champion과 byte-identical
- Package member/dependency/absolute-path audit: 통과

## Production resource comparison

253,507행 반복 test frame의 end-to-end 격리 실행 결과다. 공식 competition 제한은
repository에서 찾지 못했으므로 사전에 지정한 내부 목표를 적용했다.

| Metric | 300-tree candidate | 25-tree candidate | Reduction | Internal target | Pass |
| --- | ---: | ---: | ---: | ---: | :---: |
| ZIP | `1,400,529,259 B` | `121,125,119 B` | `91.35%` | `<500,000,000 B` | yes |
| Peak RSS | `8,754,511,872 B` | `2,250,194,944 B` | `74.30%` | `<4 GiB` | yes |
| Runtime | `20.421s` | `10.384s` | `49.15%` | `<20s` | yes |

기술 parity와 세 deployment target이 모두 통과해 verdict A다. Leaderboard 제출 자체는 이번
검증에서 수행하지 않았다.

## Artifacts

- `artifacts/extratrees_size_ablation/oof_results.csv`
- `artifacts/extratrees_size_ablation/prediction_convergence.csv`
- `artifacts/extratrees_size_ablation/subset_results.csv`
- `artifacts/extratrees_size_ablation/size_estimates.csv`
- `artifacts/extratrees_size_ablation/production_audit.json`
- `artifacts/extratrees_size_ablation/summary.json`

Reproduction:

```bash
cd /Users/wooh/Documents/dev/LG-Aimers-9th
.venv/bin/python scripts/validate_extratrees_size_ablation.py
.venv/bin/python -m unittest tests.test_extratrees_size_ablation
```

통과한 새 production candidate는 25-tree ZIP 하나뿐이다. 50/100/200 trees의 production ZIP은
생성하지 않았고 새 500-tree reference도 만들지 않았다.

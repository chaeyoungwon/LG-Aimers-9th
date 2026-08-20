# ET25 upper-weight safety validation

작성일: 2026-08-20

> 최신 상태: `w=.0200`이 LB BSS `1017.0233029621`을 기록해 현재 immutable champion으로
> 승격됐다. 이 문서의 `.015`와 `.0175` champion 표기는 당시 선택 과정을 보존한 historical
> record다. 이미 계산된 `.0225`의 production-equivalent 후속 검증은 별도 문서에 기록했다.

## Final verdict

**A. NEW UPPER WEIGHT READY FOR ONE LB SUBMISSION**

Frozen ET25 model과 original pre-ET champion endpoint를 유지하고
`0.0150 / 0.0175 / 0.0200 / 0.0225 / 0.0250` 다섯 weight만 temporal OOF로 평가했다.
`w=.0175`는 original champion 대비 세 fold가 모두 양수이고, current `.015` 대비 delta도 세
fold와 pooled에서 모두 양수다. 주요 subset safety와 production audit도 통과했다.

더 큰 weight의 pooled gain이 높더라도 선택하지 않았다. 결과 확인 전에 고정한 원칙대로
다음을 모두 만족하는 가장 작은 upper weight인 `.0175`를 선택했다.

- original champion 대비 2022/2023/2024/pooled 모두 양수
- `.015` 대비 2022/2023/2024/pooled 모두 음수가 아님
- `.015` 대비 pooled delta 최소 `+5e-7`
- 주요 subset original gain `>=-1e-5`
- 주요 subset delta vs `.015` `>=-5e-6`

## Immutable references

- Current w=.015 LB BSS: `1016.9655337572`
- Current w=.015 SHA-256 before/after:
  `11b7d514dadb47df505461176961cd6e436676bf99ddfa4e10f510fa65516b18`
- Original pre-ET champion LB BSS: `1016.4442212358`
- Original champion SHA-256 before/after:
  `ce10c428c9b6bd88ebf713fcc05b91ce22fc555573f3d11f8deb283637484c47`
- ET25 model member SHA-256:
  `6d2c3fdb3db427bea3c8474f19006746a51ec9bbbe1c8d307e80e73e17de54df`

알려진 `.010`과 `.015` LB 결과는 history로만 기록했다. LB 선형 외삽, quadratic fit 또는 LB
optimum 추정은 하지 않았다. Tree count, tree subset, seed, feature, preprocessing,
calibration 및 model object는 변경하지 않았다.

정확한 prediction 구조는 항상 다음과 같다.

```text
(1 - w) * original 1016.444 champion endpoint + w * frozen ET25
```

Current `.015` endpoint를 다시 blend base로 사용하지 않았다.

## Weight-response curve

양수는 original champion 대비 Brier improvement다.

| Weight | 2022 | 2023 | 2024 | Pooled | Worst major subset |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `.0150` | `+1.126e-6` | `+1.094e-5` | `+3.490e-6` | `+5.155e-6` | `-3.308e-6` |
| **`.0175`** | **`+1.216e-6`** | **`+1.266e-5`** | **`+3.973e-6`** | **`+5.917e-6`** | **`-3.956e-6`** |
| `.0200` | `+1.279e-6` | `+1.436e-5` | `+4.427e-6` | `+6.651e-6` | `-4.631e-6` |
| `.0225` | `+1.313e-6` | `+1.603e-5` | `+4.853e-6` | `+7.357e-6` | `-5.334e-6` |
| `.0250` | `+1.320e-6` | `+1.768e-5` | `+5.251e-6` | `+8.035e-6` | `-6.064e-6` |

Grid 안에서는 pooled curve가 계속 상승해 peak는 관찰되지 않았다. 하지만 이번 목표는 최대
OOF weight가 아니라 안전한 첫 upper step 검증이다. `.025`를 선택하거나 grid를 `.025` 위로
확장하지 않고 최소 안정 후보 `.0175`에서 멈췄다.

## Incremental gain versus w=.015

| Weight | 2022 delta | 2023 delta | 2024 delta | Pooled delta |
| ---: | ---: | ---: | ---: | ---: |
| `.0150` | `0` | `0` | `0` | `0` |
| **`.0175`** | **`+9.021e-8`** | **`+1.727e-6`** | **`+4.827e-7`** | **`+7.618e-7`** |
| `.0200` | `+1.526e-7` | `+3.426e-6` | `+9.371e-7` | `+1.496e-6` |
| `.0225` | `+1.871e-7` | `+5.098e-6` | `+1.363e-6` | `+2.202e-6` |
| `.0250` | `+1.937e-7` | `+6.743e-6` | `+1.761e-6` | `+2.880e-6` |

`.0175`의 2022 incremental gain은 작지만 음수가 아니며 2023과 2024도 같은 방향이다.

## Subset safety for w=.0175

각 cell은 `gain vs original champion / delta vs w=.015`다.

| Subset | 2022 | 2023 | 2024 | Pooled |
| --- | ---: | ---: | ---: | ---: |
| R | `+1.110e-6 / +7.552e-8` | `+6.032e-6 / +7.783e-7` | `+4.590e-6 / +5.754e-7` | `+3.926e-6 / +4.787e-7` |
| F | `+1.976e-6 / +1.949e-7` | `+6.941e-5 / +9.845e-6` | `-6.245e-7 / -2.076e-7` | `+2.118e-5 / +2.932e-6` |
| seen pitcher | `-3.871e-7 / -1.407e-7` | `+1.298e-5 / +1.771e-6` | `+6.309e-6 / +8.158e-7` | `+6.421e-6 / +8.322e-7` |
| unseen pitcher | `+7.095e-6 / +9.369e-7` | `+1.103e-5 / +1.505e-6` | `-3.956e-6 / -6.477e-7` | `+3.932e-6 / +4.843e-7` |
| full count | `+3.721e-7 / -3.851e-8` | `+3.509e-6 / +4.088e-7` | `+5.948e-6 / +7.646e-7` | `+3.275e-6 / +3.780e-7` |
| non-full | `+1.259e-6 / +9.666e-8` | `+1.313e-5 / +1.794e-6` | `+3.877e-6 / +4.690e-7` | `+6.049e-6 / +7.809e-7` |
| positive form | `-2.282e-7 / -1.096e-7` | `+4.292e-6 / +5.377e-7` | `+9.889e-6 / +1.334e-6` | `+4.686e-6 / +5.922e-7` |
| negative form | `+1.110e-6 / +6.461e-8` | `+8.221e-6 / +1.080e-6` | `+4.139e-6 / +5.024e-7` | `+4.721e-6 / +5.820e-7` |

강조 진단:

- 2024 F: original gain `-6.245e-7`, delta vs `.015` `-2.076e-7`
- 2024 full-count: original gain `+5.948e-6`, delta vs `.015` `+7.646e-7`
- Worst original gain: 2024 unseen pitcher `-3.956e-6`
- Worst delta vs `.015`: 2024 unseen pitcher `-6.477e-7`

작은 음수는 존재하지만 두 고정 safety floor 안쪽이다.

## Prediction movement and calibration shift

Pooled OOF 기준이다.

| Weight | Mean prediction | Mean delta | Mean abs delta | p50 abs | p95 abs | p99 abs | Max abs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `.0150` | `0.509059` | `+1.40e-4` | `5.64e-4` | `4.75e-4` | `1.391e-3` | `1.836e-3` | `3.364e-3` |
| `.0175` | `0.509083` | `+1.63e-4` | `6.58e-4` | `5.54e-4` | `1.623e-3` | `2.142e-3` | `3.924e-3` |
| `.0200` | `0.509106` | `+1.86e-4` | `7.52e-4` | `6.33e-4` | `1.855e-3` | `2.448e-3` | `4.485e-3` |
| `.0225` | `0.509129` | `+2.10e-4` | `8.45e-4` | `7.12e-4` | `2.086e-3` | `2.754e-3` | `5.045e-3` |
| `.0250` | `0.509153` | `+2.33e-4` | `9.39e-4` | `7.91e-4` | `2.318e-3` | `3.060e-3` | `5.606e-3` |

Movement는 weight에 비례해 매끄럽게 증가하며 `.0175`에서 급격한 calibration jump는 없다.

## Production audit

새 package는 current `.015` champion을 source로 사용하고 runtime/metadata weight만
`0.9825 / 0.0175`로 변경했다.

- Original champion component max diff: `0.0`
- ET25 model member: byte-identical
- OOF formula max diff: 2022/2023/2024 모두 `0.0`
- Final candidate single/full/shuffle/subset/reverse max diff: `0.0`
- ExtraTrees internal row-order max diff: `1.110e-16` (`atol=1e-12`)
- Isolated execution 3회: byte-identical, prediction max diff `0.0`
- Prediction finite/bounds: 통과
- ZIP CRC 및 package structure: 통과
- Runtime, 253,507 rows: `9.826s`
- Peak RSS: `2,410,430,464 bytes`
- ZIP size: `121,125,294 bytes`
- Filename length: `18 <= 30`

## Candidate and artifacts

새 upper-weight 제출 후보는 하나만 생성했다.

- File: `sub_et25_w0175.zip`
- Filename length: `18`
- SHA-256: `7c884834651d38fa4e734e3a28846367a861fe406f05054398adba4ddbd825da`
- Size: `121,125,294 bytes`
- Tree count: `25`
- Weight: `0.0175`

Artifact:

- `artifacts/et25_upper_weight/weight_results.csv`
- `artifacts/et25_upper_weight/delta_vs_w015.csv`
- `artifacts/et25_upper_weight/subset_results.csv`
- `artifacts/et25_upper_weight/prediction_shift.csv`
- `artifacts/et25_upper_weight/summary.json`

Reproduction:

```bash
cd /Users/wooh/Documents/dev/LG-Aimers-9th
.venv/bin/python scripts/validate_et25_upper_weight.py
.venv/bin/python -m unittest tests.test_et25_upper_weight
```

Leaderboard 제출은 수행하지 않았다. Weight `>.025`, 추가 grid, tree selection 또는 재학습은
실행하지 않았다.

## LB follow-up: w=.0175 promotion and frozen-grid finalization

### Current immutable champion

- LB BSS: `1017.0016684619`
- File: `artifacts/sub_et25_w0175.zip`
- SHA-256: `7c884834651d38fa4e734e3a28846367a861fe406f05054398adba4ddbd825da`
- Weight/tree count: `.0175 / 25`

LB 결과는 champion 승격 기록에만 사용했다. 다음 후보 선택에는 기존 upper-weight temporal
OOF CSV만 사용했으며 새로운 weight는 평가하지 않았다.

### Comparison versus w=.0175

Gain 열은 original pre-ET champion 대비 값이고, 괄호 안은 `.0175` 대비 incremental gain이다.

| Weight | 2022 | 2023 | 2024 | Pooled | Worst subset gain | Worst subset delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`.0200`** | `+1.279e-6` (`+6.235e-8`) | `+1.436e-5` (`+1.699e-6`) | `+4.427e-6` (`+4.544e-7`) | `+6.651e-6` (`+7.339e-7`) | `-4.631e-6` | `-6.752e-7` |
| `.0225` | `+1.313e-6` (`+9.685e-8`) | `+1.603e-5` (`+3.371e-6`) | `+4.853e-6` (`+8.805e-7`) | `+7.357e-6` (`+1.440e-6`) | `-5.334e-6` | `-1.378e-6` |
| `.0250` | `+1.320e-6` (`+1.035e-7`) | `+1.768e-5` (`+5.016e-6`) | `+5.251e-6` (`+1.278e-6`) | `+8.035e-6` (`+2.118e-6`) | `-6.064e-6` | `-2.108e-6` |

세 후보 모두 2023, 2024와 pooled에서 `.0175`를 개선하고, original subset gain
`>=-1e-5` 및 subset delta `>=-5e-6`를 유지했다. Pooled incremental gain도 고정 clarity
threshold `+5e-7`를 넘었다. 따라서 OOF 최대값을 고르는 대신 가장 작은 통과 후보 `.0200`을
정확히 하나 선택했다.

### 2024 F and full-count

| Weight | 2024 F gain | F delta vs `.0175` | 2024 full-count gain | Full-count delta vs `.0175` |
| ---: | ---: | ---: | ---: | ---: |
| **`.0200`** | `-8.716e-7` | `-2.470e-7` | `+6.684e-6` | `+7.362e-7` |
| `.0225` | `-1.158e-6` | `-5.335e-7` | `+7.392e-6` | `+1.444e-6` |
| `.0250` | `-1.484e-6` | `-8.595e-7` | `+8.072e-6` | `+2.123e-6` |

Weight 증가에 따라 2024 F는 완만하게 악화되고 full-count는 개선된다. `.0200`의 2024 F
손실과 전체 worst subset은 모두 고정 safety floor 안쪽이며, 가장 작은 step을 선택하는 근거와
일치한다.

### One next LB candidate

- File: `sub_et25_w020.zip`
- Filename length: `17`
- SHA-256: `316bf6dab11fb78f06cc74089ea38bdb3b957355f4510db5fcc8a485dc26cecf`
- Size: `121,125,327 bytes`
- Weight/tree count: `.0200 / 25`
- Runtime: `9.570 s` for 253,507 rows
- Peak RSS: `2,459,631,616 bytes`

Production audit 결과 original champion component max diff `0.0`, OOF formula max diff 전 fold
`0.0`, final candidate single/full/shuffle/subset/reverse max diff `0.0`, 3회 실행 byte-identical을
확인했다. `.0175` champion과 original R10 ZIP의 SHA도 감사 전후 동일했다. `.0225`와 `.0250`
production ZIP은 생성하지 않았다.

후속 artifact:

- `artifacts/et25_upper_weight/next_weight_comparison.csv`
- `artifacts/et25_upper_weight/next_candidate.json`

Reproduction:

```bash
cd /Users/wooh/Documents/dev/LG-Aimers-9th
.venv/bin/python scripts/finalize_et25_upper_weight.py
.venv/bin/python -m unittest tests.test_finalize_et25_upper_weight
```

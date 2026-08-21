# Frozen TrackMan production candidate

Date: 2026-08-21

## Verdict

**A. READY FOR ONE LB SUBMISSION**

Frozen `ALL TRACKMAN`, Hungarian HIGH-only, matched-row-only `w=.10` baseline을 current ET25 `w=.020` champion 위에 production-equivalent로 패키징했다. 모델·feature·mapping threshold·weight 탐색은 수행하지 않았다.

Candidate:

- File: `artifacts/sub_tm10.zip`
- Filename length: 12
- SHA-256: `e40adc1204e3a841914e4c28dca351143b92be5786e20d6c4fdcf14765cc83d0`
- Size: 156,250,336 bytes
- TrackMan trees: 25
- TrackMan weight: 0.10 on HIGH matched rows only

Prediction formula:

```text
unmatched: p_final = p_current_champion
HIGH:      p_final = round(0.90*p_current_champion + 0.10*p_trackman, 14)
```

Matched 결과의 14자리 반올림은 current champion 내부 BLAS의 `1.11e-16` row-order 흔들림을 제거하기 위한 deterministic serialization 처리다. 최대 변화는 `5e-15` 이하이며 unmatched row에는 적용하지 않는다.

## Immutable inputs

- Champion: `artifacts/sub_et25_w020.zip`
- Champion SHA before/after: `316bf6dab11fb78f06cc74089ea38bdb3b957355f4510db5fcc8a485dc26cecf`
- Crosswalk: accepted Hungarian, HIGH only
- Production cutoff: official train and TrackMan through 2024
- External identity information: none
- Test aggregation/distribution use: none

## Mapping parity

Official train and `trackman_history.csv`에서 mapping을 재생성했다.

| Validation | Cutoff | HIGH pitchers | HIGH row coverage | Signature parity |
|---|---:|---:|---:|---|
| 2022 | 2021 | 90 | 26.3525% | exact |
| 2023 | 2022 | 91 | 28.6065% | exact |
| 2024 | 2023 | 108 | 30.0039% | exact |

Cutoff 2024 production mapping은 두 번 독립 생성했다. 두 run의 main pitcher → TrackMan pitcher, confidence classification 및 canonical signature가 exact-equal이었다. HIGH pitcher는 111명이며 signature는 `9b18812e7c5952cebf81c1b588f0b14511d42d387046c66eea38ad6ca415b82b`이다.

## OOF production parity

| Fold | Prediction max diff | Runtime path max diff | Brier gain |
|---|---:|---:|---:|
| 2022 | 3.331e-16 | 4.441e-16 | +1.2124564658e-5 |
| 2023 | 3.331e-16 | 3.331e-16 | +2.6392871962e-5 |
| 2024 | 3.331e-16 | 3.331e-16 | +2.1054167049e-5 |
| pooled | — | — | +1.9849831856e-5 |

Fold Brier gain은 frozen reference와 정확히 일치했고, pooled는 matched-output 반올림 때문에 `8.33e-17` 차이였다. Full production ET25는 1,475,092개 train row 중 HIGH mapping에 해당하는 560,600개 row로 학습했다. 두 학습 run, serialized reload, runtime feature path prediction의 max diff는 모두 `0.0`이었다.

## Production lookup

Inference package에는 cutoff 2024에서 미리 만든 immutable artifact만 포함한다.

- `model/tm_lookup.npz`: 111개 HIGH pitcher의 accepted ALL TRACKMAN feature lookup
- `model/tm_meta.json`: feature order, category levels, train-only imputation, model recipe
- `model/tm_expert.joblib`: frozen 25-tree ExtraTrees expert
- `trackman_runtime.py`: current-row pitcher ID lookup과 matched-only blend

Test row들로 groupby, mapping, normalization, imputation 또는 threshold 계산을 하지 않는다. 로컬 5행 test에는 HIGH pitcher가 없어서 candidate output이 champion과 exact-equal이다. Routing 검증은 111개 HIGH 행과 5개 unmatched 행을 섞은 별도 audit-only frame으로 수행했다.

## Independence and determinism

Matched-heavy 116행 audit frame 결과:

| Variant | Max diff |
|---|---:|
| single | 0.0 |
| full | 0.0 |
| shuffle | 0.0 |
| subset | 0.0 |
| reverse | 0.0 |

- Champion component max diff: `0.0`
- Formula max diff: `0.0`
- Unmatched max diff: `0.0`
- Isolated local test 3회 submission SHA: byte-identical
- Prediction max diff across 3 runs: `0.0`

## Runtime and package

Runtime은 matched-heavy audit rows를 253,507행으로 확장해 current champion과 같은 조건에서 측정했다.

| Metric | Current champion | Candidate | Delta |
|---|---:|---:|---:|
| Runtime | 12.393 sec | 13.717 sec | +1.323 sec |
| Peak RSS | 1,688,403,968 B | 2,045,394,944 B | +356,990,976 B |
| ZIP | 121,125,327 B | 156,250,336 B | +35,125,009 B |

내부 기준 ZIP 500 MiB, runtime 20 sec, peak RSS 4 GiB를 모두 통과했다. ZIP CRC, required files, model/lookup 존재, requirements byte parity, NaN/Inf, prediction bounds, row count/order, absolute path, 실험 artifact 혼입 검사를 모두 통과했다.

## Competition-rule statement

- 사용 데이터: 공식 `train.csv`, 공식 `trackman_history.csv`
- Crosswalk: 공식 데이터 내부 pitch-mix/time-series fingerprint만 사용
- 외부 선수 이름, roster, 인터넷 또는 수동 mapping: 미사용
- MEDIUM/LOW mapping: 미사용
- Test 다른 행 또는 test distribution: 미사용
- 각 prediction은 현재 행과 train-derived frozen lookup만 사용
- Leaderboard 제출은 수행하지 않음

## Reproduce

Full mapping/parity/build/audit:

```bash
.venv/bin/python scripts/build_tm_candidate.py
```

이미 통과한 mapping/OOF artifact를 SHA 검증 후 재사용하는 package audit:

```bash
.venv/bin/python scripts/build_tm_candidate.py --reuse-parity
```

Results:

- `artifacts/tm_production/mapping_parity.json`
- `artifacts/tm_production/oof_parity.json`
- `artifacts/tm_production/row_independence.json`
- `artifacts/tm_production/determinism.json`
- `artifacts/tm_production/runtime.json`
- `artifacts/tm_production/package_audit.json`
- `artifacts/tm_production/summary.json`

# ExtraTrees production parity and package audit

작성일: 2026-08-20

## Final verdict

**B. TECHNICALLY VALID BUT GAIN TOO SMALL / RISK TOO HIGH**

동결 후보 `0.99 * champion + 0.01 * ExtraTrees`는 production-equivalent OOF,
champion endpoint parity, row independence, 격리 실행 및 결정성을 모두 통과했다. 그러나
candidate ZIP은 `1,400,529,259 bytes`이고 253,507행 격리 추론의 peak RSS는
`8,754,511,872 bytes`였다. 공식 제한은 repository에서 찾지 못했지만 실험 전에 정한
engineering guard인 1 GiB와 champion 대비 4배 메모리를 각각 초과했다. Pooled OOF gain은
`+3.424e-6`에 불과하므로 이 비용과 운영 위험을 정당화하지 못한다.

따라서 candidate는 기술 감사 기록으로만 보존하고 leaderboard에 제출하지 않는다. Frozen
champion을 계속 유지한다.

## Frozen inputs

- champion LB BSS: `1016.4442212358`
- champion artifact: `artifacts/submit_r10_pitcher_w418194.zip`
- champion SHA-256 before/after:
  `ce10c428c9b6bd88ebf713fcc05b91ce22fc555573f3d11f8deb283637484c47`
- candidate formula: `0.99 * frozen champion + 0.01 * frozen ExtraTrees`
- ExtraTrees seed: `42`
- calibration, weight search, seed ensemble, RF/XGBoost, LB probe: 없음

Champion ZIP은 읽기 전용으로 사용했다. Candidate는 별도 경로에 생성했으며 `submission/`과
champion payload를 수정하지 않았다.

## Production contract

Accepted temporal OOF와 동일하게 champion HGB/CatBoost의 ID-free 52열 row-local feature를
순서까지 재사용했다. `row_id`, `control_success`, `pitcher_id`, `batter_id`는 제외했다. Category
map과 numeric median은 full train에서만 fit하고 metadata에 저장했다. Test 통계, test row 간
집계, validation label 기반 preprocessing은 없다.

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

Runtime은 기존 champion의 전체 `predict`를 먼저 실행한 뒤 그 최종 endpoint에만 ExtraTrees
1% convex blend를 한 번 적용한다. 기존 form, batter form 및 cold-start 처리를 다시 적용하지
않는다. `requirements.txt`는 champion과 byte-identical하며 이미 고정된 scikit-learn 1.8.0과
joblib 1.5.3만 사용한다.

## Temporal OOF production parity

양수 gain은 frozen champion 대비 Brier 개선이다. Production feature/preprocessing/model
구현의 ExtraTrees 예측을 accepted OOF cache와 직접 비교했다.

| Validation | Rows | Extra prediction max diff | Brier gain | Accepted gain diff | Pass |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 2022 | 247,472 | `3.331e-16` | `+1.130260901067e-6` | `0.0` | yes |
| 2023 | 245,525 | `3.331e-16` | `+7.195466559029e-6` | `0.0` | yes |
| 2024 | 253,507 | `3.331e-16` | `+2.010170354089e-6` | `0.0` | yes |
| pooled | 746,504 | — | `+3.423916159345e-6` | `0.0` | yes |

세 fold가 모두 양수이고 accepted gain을 그대로 재현했다. 다만 절대 개선량은 매우 작다.

## Full-train reproducibility

동일 환경과 고정 레시피로 full train fit을 반복했다. 첫 fit의 직렬화 모델을 기준으로 새 fit을
비교한 test prediction max diff는 `2.220446049250313e-16`이고, joblib reload max diff는
`1.1102230246251565e-16`이다. 사전 OOF parity와 같은 `1e-12` 수치 허용오차를 충분히
통과해 사실상 동일하다. 새로 측정한 반복 fit 시간은 `194.57s`였다.

## Endpoint and row safety

- Candidate 안의 blend 직전 champion component vs 원본 champion 내부 prediction max diff:
  `0.0`
- ExtraTrees single/full/shuffled/subset/reverse max diff: `1.665e-16`
  (`atol=1e-12`)
- 최종 candidate single/full/shuffled/subset/reverse max diff: `0.0`
- Candidate formula max diff: `5.551e-17` (`atol=1e-12`)
- NaN / Inf: `0 / 0`
- 확률 범위 `[0, 1]`: 통과

Local test는 5행뿐이므로 delta는 성능 근거가 아니라 구현 sanity check로만 사용했다.

| Statistic | Candidate - champion |
| --- | ---: |
| mean | `+4.0498736e-4` |
| std | `1.5379064e-4` |
| min | `+1.7197862e-4` |
| max | `+5.8760526e-4` |
| changed rows | `5 / 5` |

## Determinism and isolated runtime

동일한 candidate ZIP을 새 격리 subprocess에서 3회 실행했다. 세 output SHA-256은 모두
`9ff6a20ccee9350b9385c010cb60286aa341970e5ecbb14557523437350af4aa`로 byte-identical이고
prediction max diff는 `0.0`이다.

2024 validation과 같은 253,507행 규모의 반복 test frame으로 end-to-end 추론을 측정했다.

| Metric | Candidate | Champion reference | Ratio | Guard | Pass |
| --- | ---: | ---: | ---: | ---: | :---: |
| wall time | `20.421s` | `8.964s` | `2.278x` | `<=5x` | yes |
| peak RSS | `8,754,511,872 B` | `1,440,284,672 B` | `6.078x` | `<=4x` | **no** |

Runtime은 허용 범위지만 모델 로딩과 tree object 상주 메모리가 위험 기준을 초과했다.

## Package audit

| Item | Result |
| --- | --- |
| Candidate ZIP size | `1,400,529,259 bytes` |
| Serialized ExtraTrees size | `1,397,180,060 bytes` |
| Predeclared package guard | `1,073,741,824 bytes` |
| Champion files retained | script 외 payload byte-identical |
| Exact additions | runtime, model, metadata 3개 |
| Requirements | champion과 byte-identical |
| New dependency | 없음 |
| Absolute local path | 없음 |
| Model reload | 성공 |

구조 및 dependency 감사 자체는 통과했다. 크기 초과는 parity 실패가 아니라 배포 위험으로
분류했으며, 이 위험과 메모리 초과 때문에 verdict B다. 사용자 지시에 따라 tree 수 축소,
모델 pruning, weight 변경 또는 다른 serialization 실험은 진행하지 않았다.

## Artifacts and reproduction

- `artifacts/extratrees_production/oof_parity.json`
- `artifacts/extratrees_production/test_delta.json`
- `artifacts/extratrees_production/row_independence.json`
- `artifacts/extratrees_production/determinism.json`
- `artifacts/extratrees_production/runtime.json`
- `artifacts/extratrees_production/package_audit.json`
- `artifacts/extratrees_production/summary.json`

```bash
cd /Users/wooh/Documents/dev/LG-Aimers-9th
.venv/bin/python scripts/build_extratrees_candidate.py
.venv/bin/python scripts/build_extratrees_candidate.py --audit-only
.venv/bin/python -m unittest tests.test_extratrees_production
```

Candidate package는 기술 감사 재현용 산출물이며 제출 추천 파일이 아니다. Champion이 계속
유일한 frozen deployment artifact다.

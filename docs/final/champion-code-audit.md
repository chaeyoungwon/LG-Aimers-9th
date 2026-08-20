# Champion code audit

작성일: 2026-08-20

## Current LB champion update

현재 immutable LB champion은 ET25 `w=.0200` endpoint다.

| Item | Current value |
| --- | --- |
| LB BSS | `1017.0233029621` |
| Artifact | `artifacts/sub_et25_w020.zip` |
| SHA-256 | `316bf6dab11fb78f06cc74089ea38bdb3b957355f4510db5fcc8a485dc26cecf` |
| Size | `121,125,327 bytes` |

아래 감사는 이 endpoint의 original champion component인 pre-ET R10 package에 대한 동결
기록이다. 현재 champion을 다시 중첩 blend하지 않으며, ET weight endpoint는 항상 original
R10 prediction과 frozen ET25 prediction을 직접 결합한다.

## Original pre-ET champion overview

| Item | Frozen value |
| --- | --- |
| LB BSS | `1016.4442212358` |
| Artifact | `artifacts/submit_r10_pitcher_w418194.zip` |
| SHA-256 | `ce10c428c9b6bd88ebf713fcc05b91ce22fc555573f3d11f8deb283637484c47` |
| Size | `3,345,014 bytes` |
| ZIP payload | 13 files; required `script.py`, `requirements.txt`, `model/` present |
| Final verdict | **B. READY WITH NON-BLOCKING WARNINGS** |

감사 시작과 종료 시 checksum과 size가 일치했다. 감사 및 재빌드 과정은 champion을 읽기
전용으로 사용했으며, 재빌드 결과는 `artifacts/final_audit/rebuilt_champion.zip`에만 썼다.
ZIP에는 train/test/sample CSV, repository source tree, notebook, cache, 임시 artifact, 외부
데이터, secret 또는 실행 가능한 로컬 절대경로 참조가 없다. 최근 OOF 연구 코드
(`regime_blend`, `residual_discovery`, `pitcher_form_shape`, `trackman_incremental`)도 포함되지
않고 추론 시 import되지 않는다.

## Prediction pipeline

실제 packaged `script.py`의 실행 순서는 다음과 같다.

```text
data/test.csv + data/sample_submission.csv
  -> meta의 고정 category level 적용 및 row-local feature engineering
  -> HGB, CatBoost, ours embedding NN 예측
  -> ours = logit-intercept calibration(0.4*HGB + 0.4*CatBoost + 0.2*NN)
  -> team LightGBM stack, team embedding NN 예측
  -> team = 0.9*team LightGBM + 0.1*team NN
  -> base = 0.4*ours + 0.6*team
  -> pitcher season-form endpoint
  -> batter season-form endpoint
  -> regular-season unseen-pitcher cold-start expert, w=0.4181944103545871
  -> [0, 1] clip
  -> sample_submission.row_id 순서로 output/submission.csv 작성
```

모든 model blend와 calibration은 같은 행의 확률만 사용한다. `sample_submission.csv`는 출력
schema와 순서를 정하는 용도로만 사용한다.

## Model components

| Component | Model file | Serialization | Loader | Required package | Repository module required |
| --- | --- | --- | --- | --- | :---: |
| HGB | `model/hgb_model.pkl` | joblib pickle | `joblib.load` | joblib, scikit-learn | no |
| CatBoost | `model/cat_model.cbm` | native CBM | `CatBoostClassifier.load_model` | catboost | no |
| ours embedding NN | `model/nn_model.npz` | NumPy state arrays | local Torch class + `np.load` + `load_state_dict(strict=True)` | numpy, torch | no |
| team LightGBM | `model/team_model.lgbmstack.json` | four native LightGBM model strings in JSON | `lightgbm.Booster(model_str=...)` | lightgbm | no |
| team embedding NN | `model/team_nn_model.npz` | NumPy state arrays | local Torch class + `np.load` + `load_state_dict(strict=True)` | numpy, torch | no |
| cold-start expert | `model/coldstart_lgbm.txt` | native LightGBM text | `lightgbm.Booster(model_file=...)` | lightgbm | no |
| pitcher season form | `model/meta.json#season_form` | JSON constants/lookup | `json.load`, inline `season_form_values` | stdlib, numpy, pandas | no |
| batter season form | `model/batter_form.json` | JSON constants/lookup | `json.load`, packaged `apply_batter_form` | stdlib, numpy, pandas | no |

모델 class pickle이 필요한 HGB는 training version과 같은 scikit-learn을 pin했다. CatBoost와
LightGBM은 native format, NN은 NPZ state와 ZIP 내부 class definition을 사용하므로 repository
module import가 필요 없다.

## Feature sources

| Feature / source | Origin | Train-only | Current test row-only | Rule-safe |
| --- | --- | :---: | :---: | :---: |
| 경기 문맥, count, 주자, score, 기대승률, leverage, hand/team | organizer `test.csv`의 현재 행 | no | yes | yes |
| as-of pitcher features와 pitch mix | organizer가 제공한 현재 행의 시점 이전 누적값 | no | yes | yes |
| as-of batter features | organizer가 제공한 현재 행의 시점 이전 누적값 | no | yes | yes |
| category level/map | `model/meta.json`, `model/coldstart_meta.json` | yes | no | yes |
| NN median/mean/std, category level, ID vocabulary | `model/meta.json` | yes | no | yes |
| stage weight와 calibration constant | `model/meta.json` | yes | no | yes |
| pitcher/batter form lookup | ≤2024 train에서 계산한 JSON + 현재 행 as-of count/rate | yes | yes | yes |
| cold known-pitcher set와 expert | ≤2024 train의 R 행 + 현재 행 pitcher/game type | yes | yes | yes |
| sample submission | organizer file; output schema/order만 사용 | no | no | yes |
| TrackMan | 사용하지 않음 | no | no | yes |

## Train-derived statistics

- Base members의 category vocabulary, feature order, missing-value preprocessing 및 NN
  normalization/ID vocabulary는 ZIP의 metadata에 이미 고정되어 있다.
- Ours calibration intercept, member weights, stage weights도 고정 상수이며 test 분포로 다시
  fitting하지 않는다.
- Pitcher form의 `(N0, S0, pooled_p0, mu)`와 batter form의 `(N0, S0)` table은 2024년까지의
  train에서 생성되었다. 현재 행의 official as-of `n/rate`에서 이 기준치를 빼 당해 시즌 값을
  계산한다.
- Cold-start known pitcher set, category map 및 ID-free LightGBM은 2024년까지의 정규시즌
  train 행에서 생성되었다. Cold expert feature에는 pitcher/batter ID가 없다.
- 정적 감사에서 prediction에 영향을 주는 `mean`은 cold expert의 동일 행 model-axis 평균
  하나뿐이다. 그 외 `mean`은 verbose 출력용이며 test-wide `groupby`, rolling, rank, shift,
  diff, cumulative 또는 분포 통계는 없다.

## Cold-start handling

Gate는 현재 행이 `game_type == "R"`이고, 그 `pitcher_id`가 train-period regular-season
pitcher table에 없을 때만 열린다. Expert는 ID-free row-local feature로 해당 행을 예측하며,
최종 확률은 다음 고정식이다.

```text
p_final = (1 - 0.4181944103545871) * p_form + 0.4181944103545871 * p_cold_expert
```

그 외 행에는 `p_form`이 그대로 유지된다. Segment boost와 seed ensemble metadata는 champion에
없다.

## Row independence proof

격리된 champion의 실제 `script.py`에 test 원본 앞 5개 행을 single/full/shuffled/reversed/
subset/duplicate-unrelated 형태로 주입했다. Anchor `TEST_000001`의 기준 prediction은
`0.396649133953031`이고, 모든 공통 행의 최대 절대차는 `0.0`이었다(`atol=0`).

정적 감사도 test-wide prediction aggregation과 실행 가능한 `test["control_success"]` 접근이
없음을 확인했다. Synthetic smoke test 11종(unseen pitcher/batter, categorical/numeric missing,
rare category, F/R, full count, bases empty/full, extreme as-of count)은 모두 crash, NaN, Inf 없이
`[0,1]` 범위를 통과했다. Output length, column, unique row ID 및 sample order도 모두 일치했다.

## Temporal leakage safeguards

- Test label은 ZIP에 없고 추론 경로가 `control_success`를 읽지 않는다. Source 주석/상수명에
  target/label 문자열이 있으나 executable test target access는 0개다.
- Form lookup, category/normalization constants, calibration과 cold-start model/gate는 모두
  train에서 미리 계산되었다. Test 전체의 평균·빈도·순위나 다른 test 행을 사용하지 않는다.
- 공식 as-of feature는 현재 행에 제공된 시점 이전 누적값만 읽는다.
- OOF 연구 코드는 production ZIP에 포함되지 않으며 champion 추론과 분리되어 있다.
- Sample submission은 row alignment에만 관여하고 prediction feature로 사용되지 않는다.

## TrackMan exclusion rationale

공식 `train.pitcher_id`와 `trackman_history.pitcher_trackman_id`는 서로 다른 namespace이며 공식
crosswalk가 없다. 이름, 외부 데이터 또는 fuzzy identity inference를 쓰지 않은 exact
cutoff-safe coverage는 2022/2023/2024 모두 `0%`였다. 따라서 TrackMan feature를 안전하게
player row에 연결할 수 없어 champion에서 의도적으로 제외했다. 이는 임의 대치로 coverage를
만들지 않기 위한 규정 준수 결정이다. Packaged script에 dormant 호환 코드가 있더라도 실제
member feature group/model input에는 TrackMan이 선언되어 있지 않다.

## Serialization/dependency

`requirements.txt` pin은 scikit-learn `1.8.0`, joblib `1.5.3`, pandas `2.3.3`, LightGBM
`4.6.0`, CatBoost `1.2.8`, NumPy `1.26.4`이며 로컬 검증 버전과 일치했다. SciPy는 import하지
않고 직접 requirement도 아니다.

비차단 의존성 경고는 두 가지다.

1. 두 embedding NN이 Torch를 사용하지만 `requirements.txt`에는 없다. 제출물은 평가 base
   image에 이미 존재하는 Torch에 의존하며 로컬 감사 버전은 `2.8.0`이었다.
2. 로컬 macOS에서는 LightGBM의 OpenMP loader를 위해 sklearn wheel의 `libomp.dylib` 위치를
   자식 process의 `DYLD_LIBRARY_PATH`에 주입했다. ZIP이 해당 절대경로를 읽는 것은 아니며
   audit runner가 host runtime을 보완한 것이다.

`model/meta.json`에 repository 이름 `LG-Aimers-9th`가 provenance 문자열로 한 번 남아 있으나
로컬 절대경로가 아니고 파일 open/import에 쓰이지 않는다. Champion immutable 원칙 때문에
metadata를 다시 쓰지 않고 비차단 경고로 보존했다.

## Reproduction instructions

필수 local data는 `/Users/wooh/Documents/dev/open/data/test.csv`와
`sample_submission.csv`다. 감사 script는 champion을 빈 임시 디렉터리에 안전하게 풀고,
추출본과 `data/`, `output/`만 있는 상태에서 `PYTHONPATH=""`, `PYTHONNOUSERSITE=1`로 packaged
`script.py`를 실행한다.

```bash
cd /Users/wooh/Documents/dev/LG-Aimers-9th
shasum -a 256 artifacts/submit_r10_pitcher_w418194.zip
.venv/bin/python scripts/build_r10_coldstart.py \
  --weight 0.4181944103545871 \
  --output artifacts/final_audit/rebuilt_champion.zip
.venv/bin/python scripts/audit_final_submission.py --runtime-rows 253507
.venv/bin/python -m unittest tests.test_final_submission_audit
```

macOS에서 재빌드 단계만 OpenMP loader 오류가 나면 현재 환경의 sklearn wheel에 포함된
`.dylibs` 경로를 `DYLD_LIBRARY_PATH`로 지정한다. 감사 script 자체는 이 경로를 자동 탐색한다.

재현 결과:

| Check | Result |
| --- | --- |
| Determinism | 3/3 byte-identical; output SHA `c42ddac2acbeac5f2ea58c64bfc790df894c37fa2f21703960be57ef80aee163` |
| Near-full runtime | 253,507 rows in `8.964 s` (`28,282 rows/s`) |
| Peak child RSS | `1,440,284,672 bytes` (about 1.34 GiB) |
| Runtime output | `10,951,526 bytes` |
| Rebuilt ZIP | same 13 payload names; prediction max absolute difference `0.0` |
| Rebuild byte differences | LightGBM의 unused empty GPU footer 1줄, cold metadata의 non-inference provenance fields만 다름 |

로컬 대회 문서에는 명시적인 runtime/memory limit가 없어 수치적 여유를 증명할 수 없다.
따라서 측정 통과는 기록하되 이를 비차단 경고로 남긴다.

상세 machine-readable 결과는 `artifacts/final_audit/summary.json`과 같은 디렉터리의
`zip_inventory.txt`, `dependency_audit.json`, `static_independence_audit.json`,
`row_independence.json`, `determinism.json`, `smoke_test.json`, `runtime.json`,
`rebuild_parity.json`에 있다.

## Final checksum

```text
artifact: artifacts/submit_r10_pitcher_w418194.zip
size:     3,345,014 bytes
SHA-256:  ce10c428c9b6bd88ebf713fcc05b91ce22fc555573f3d11f8deb283637484c47
start:    ce10c428c9b6bd88ebf713fcc05b91ce22fc555573f3d11f8deb283637484c47
end:      ce10c428c9b6bd88ebf713fcc05b91ce22fc555573f3d11f8deb283637484c47
```

## Final verdict

**B. READY WITH NON-BLOCKING WARNINGS**

Blocker는 0개다. Torch base-image dependency, local macOS OpenMP loader, non-executable
repository provenance 문자열, 공개 limit 부재, 재빌드의 non-inference byte 차이를 경고로
기록한다. 현재 champion ZIP은 그대로 최종 제출 대상으로 유지한다.

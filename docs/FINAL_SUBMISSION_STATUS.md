# Final Submission Status

감사일: 2026-08-26

## Operational status

- **ACTIVE MODEL SEARCH: STOPPED**
- **CHAMPION FROZEN: YES**
- **READY FOR SUBMISSION: YES**
- 최종 결정: **READY — KEEP AND SUBMIT 21ST CHAMPION**

## Golden champion

| 항목 | 값 |
| --- | --- |
| 모델 | CatBoost 30% + LightGBM team 70% |
| LB | `1058.074429882` |
| ZIP | `artifacts/sub_tree_reblend_w0p300.zip` |
| SHA-256 | `8ce93a197efea18cf89b7b988c59a8b369d683983152b24b74976a96885da7cb` |
| ZIP CRC | PASS — 17/17 files |

Golden ZIP은 이번 감사에서 수정·재생성하지 않았다. Expected 17-file inventory와 정확히
일치했고 unexpected file, 절대경로, `..`, `__MACOSX`, 캐시·임시 파일, 사용자 로컬
경로를 포함하지 않는다.

## Isolated execution

ZIP을 repository 밖 임시 디렉터리에 단독으로 풀고 공식 로컬 5행 test/sample fixture만
복사해 실행했다. `libomp.dylib`를 Codex 런타임이 기본 탐색하지 못한 첫 시도는 환경
문제로 중단됐으며, 설치된 scikit-learn bundled OpenMP 경로를 명시한 재실행은 정상
종료했다.

- 모델 로딩과 return code: PASS
- 출력: `output/submission.csv`, 5 rows
- schema: `row_id`, `control_success`
- dtype/finite: numeric, NaN/Inf 없음
- probability range: `[0.4024170870203402, 0.4991351595077437]`
- ZIP 밖 프로젝트 코드·모델 의존: 없음
- 인터넷/API 호출: 없음

실행 환경은 Python `3.12.13`, NumPy `1.26.4`, pandas `2.3.3`, scikit-learn `1.8.0`,
joblib `1.5.3`, LightGBM `4.6.0`, CatBoost `1.2.8`이었다. 이는 ZIP의 pinned
`requirements.txt`와 일치한다. macOS 로컬에서는 LightGBM용 OpenMP runtime이 별도로
필요하지만 제출 코드나 artifact 결함은 아니다.

## Row independence and rule audit

공식 train에서 target을 제거한 3,000행 fixture와 실제 ZIP의 `script.predict`를 사용했다.

| 구성 | max_abs_diff | mean_abs_diff | changed_rows |
| --- | ---: | ---: | ---: |
| full | 0.0 | 0.0 | 0 |
| shuffled | 0.0 | 0.0 | 0 |
| reversed | 0.0 | 0.0 | 0 |
| subset | 0.0 | 0.0 | 0 |
| single | 0.0 | 0.0 | 0 |
| unrelated-row mutation | 0.0 | 0.0 | 0 |

Inference data flow는 현재 row의 공식 predictor와 frozen official-train-derived
model/state/lookup만 사용한다. `asof_*`는 현재 row에서 읽고 `season_state.json`의
학습 유래 고정 baseline과 결합한다. Test groupby/frequency/rolling/batch statistic,
test-fitted normalization, prediction-distribution calibration, adaptive threshold,
test-test retrieval/attention/regime detection은 없다. 코드의 평균 출력은 logging 또는
고정 model ensemble이며 test 통계를 다시 feature·weight·prediction에 주입하지 않는다.
공식 규칙/Q&A의 저장된 snapshot과 충돌하거나 변경된 로컬 공지는 발견되지 않았다.

## Artifact inventory

- runtime: `script.py`, `batter_form_runtime.py`, `coldstart_runtime.py`,
  `corrections_runtime.py`, `season_state_runtime.py`
- contract: `requirements.txt`, `model/meta.json`
- model/artifact: `model/batter_form.json`, `model/cat_model.cbm`,
  `model/coldstart_lgbm.txt`, `model/coldstart_meta.json`, `model/corrections.json`,
  `model/hgb_model.pkl`, `model/nn_model.npz`, `model/season_state.json`,
  `model/team_model.lgbmstack.json`, `model/team_nn_model.npz`

`meta.json`의 실제 활성 경로는 Cat member `[0,1,0]`, team LightGBM member `[1,0]`,
stage `[0.3,0.7]`이다. HGB/NN/team-NN artifact는 호환을 위해 남아 있지만 최종 blend
weight는 0이다. 현재 투구 TrackMan과 외부 데이터는 활성 inference 경로에 없다.

## Search closure

최종 근거는 `NO MATERIAL RECOVERABLE HEADROOM — STOP MODEL SEARCH`다. Two-strike,
Cat/team MoE·weight tuning, calibration, residual specialist, hierarchical EB,
hard/uncertainty weighting, GroupDRO/stable objective, role state, feature derivative,
TrackMan profile/LUPI/distillation/self-supervised, external data, XGBoost,
FT-Transformer, TabM, retrieval/kNN, prototype, low-rank ALS, DeepFM/ID-aware attention과
architecture/representation search는 모두 CLOSED다.

마지막 기각 제출은 22차 two-strike로 LB `1056.264418601`, champion 대비
`-1.810011281`이다. Robust local validation과 paired 2SE를 통과한 작은 개선도 LB로
전이되지 않은 negative evidence로 보존한다. 과거 ZIP은 삭제·이름 변경하지 않으며,
`sub_tree_reblend_w0p300.zip`만 FINAL CHAMPION이고 나머지는 HISTORICAL/REJECTED다.

## Reopen conditions

다음 중 하나가 확인될 때만 사람의 판단을 거쳐 재검토한다.

1. 새 공식 Phase 2 데이터 제공
2. 외부 데이터·정보 사용 규칙 또는 평가 규칙의 공식 변경
3. champion/OOF pipeline의 실질적 누수·정렬·구현 결함 발견
4. 기존 headroom 감사 밖의 official-data-only 구조 증거가 사전 2024 `+10 BSS` 이상의
   deployable potential을 설명
5. 공식 metric 또는 evaluation protocol 변경

단순 아이디어나 작은 로컬 개선은 재개 조건이 아니다.

## Verification commands

- `pytest`: 13 passed
- `compileall src scripts tests`: PASS
- `git diff --check`: PASS

Builder는 `scripts/build_tree_reblend.py --cat-weight 0.30`, rolling member OOF 경로는
`scripts/build_champion_oof.py`다. 공식 train source와 pinned dependency·seed/model recipe는
동결 문서에 기록돼 있다. 이번 운영 감사에서는 builder를 실행하지 않았다.

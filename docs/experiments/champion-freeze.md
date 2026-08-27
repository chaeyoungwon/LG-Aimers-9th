# 21차 Champion 최종 동결 감사

실행일: 2026-08-22

## Freeze status

**CHAMPION FROZEN = YES**

| 항목 | 최종 값 |
| --- | --- |
| ZIP | `artifacts/sub_tree_reblend_w0p300.zip` |
| 구성 | CatBoost 30% + LightGBM team 70% |
| LB | `1058.074429882` |
| SHA-256 | `8ce93a197efea18cf89b7b988c59a8b369d683983152b24b74976a96885da7cb` |
| ZIP CRC | PASS — 17/17 files |
| 현재 action | FINAL SUBMISSION OPERATIONS |
| 탐색 | ACTIVE MODEL SEARCH STOPPED |

기존 artifact의 SHA-256이 기준값과 일치한 뒤에만 나머지 감사를 진행했다. ZIP을
재생성하거나 덮어쓰지 않았고 모델 및 inference 코드를 변경하지 않았다.

## Latest rejected submission

- 22차: two-strike LGB/Cat expert ensemble
- LB: `1056.264418601`
- champion 대비: `-1.810011281`
- 상태: **REJECTED — TWO-STRIKE SEARCH CLOSED**
- 제출 당시 SHA-256: `48add68ddd754f496256483aa41057de9612592a601f2ea18e5f445a1bfeb297`

0-2/1-2/2-2, LGB/Cat expert, ensemble, residual specialist, interaction, count
세분화, weight tuning을 포함한 모든 two-strike 파생을 재탐색·재튜닝·재제출하지 않는다.

## ZIP integrity 및 dependency 감사

- 17개 파일: `script.py`, runtime module 4개, `requirements.txt`, `model/meta.json`,
  model/artifact 10개가 존재한다.
- `unzip -t`: 전 파일 CRC 정상.
- absolute member path, `..` member, 사용자 로컬 경로: 없음.
- 인터넷/API/원격 모델 호출: 없음.
- ZIP 밖 프로젝트 모듈 import: 없음. 모델은 ZIP 내부 `./model`만 읽는다.
- 입력 의존성: 평가 환경이 제공하는 `./data/test.csv`, `./data/sample_submission.csv`.
- 출력: `./output/submission.csv`.
- requirements: `scikit-learn==1.8.0`, `joblib==1.5.3`, `pandas==2.3.3`,
  `lightgbm==4.6.0`, `catboost==1.2.8`, `numpy==1.26.4`.
- HGB pickle은 Python 3.12 + NumPy 1.26.4 환경에서 생성된 호환 artifact다.

`meta.json`의 실제 stage는 Cat only `[0,1,0]`, team LightGBM only `[1,0]`, 최종
stage weight `[0.3,0.7]`이다. dormant HGB/NN/team-NN 파일은 loader 호환을 위해 ZIP에
남아 있지만 최종 probability blend weight는 0이다.

## Isolated execution

ZIP만 임시 디렉터리에 풀고 공식 로컬 test/sample fixture 5행을 `./data`에 복사하여
repository 밖에서 `script.py`를 실행했다.

| 검사 | 결과 |
| --- | --- |
| import / model loading / return code | PASS |
| output schema | `row_id`, `control_success` — PASS |
| output rows | 5 — PASS |
| finite | PASS |
| prediction range | `[0.4024170870203402, 0.4991351595077437]` — PASS |
| 원본 repository runtime 의존 | 없음 |

공식 로컬 test fixture는 5행 샘플이므로 full hidden evaluation 전체 실행을 검증한 것은
아니다. 다만 ZIP 자체의 import, model loading, end-to-end inference는 격리 상태에서
실측 PASS했다.

## Row independence

공식 train에서 target을 제거하고 test schema로 만든 3,000행 대체 fixture에 실제 ZIP의
`script.predict`를 사용했다. 고정 artifact와 현재 행만 사용하는지 다음 구성을 비교했다.

| 구성 | max_abs_diff | mean_abs_diff | changed_rows |
| --- | ---: | ---: | ---: |
| full | 0.0 | 0.0 | 0 |
| shuffled | 0.0 | 0.0 | 0 |
| reversed | 0.0 | 0.0 | 0 |
| subset | 0.0 | 0.0 | 0 |
| single | 0.0 | 0.0 | 0 |
| unrelated-row mutation | 0.0 | 0.0 | 0 |

3,000행 예측도 전부 finite였고 범위는
`[0.3752442446776844, 0.7871929991376083]`였다. 행 독립성은 **PASS**다.

## Reproducibility contract

최종 ZIP은 재생성하지 않는다. 재현 경로는 다음과 같다.

1. base season-state artifact: `artifacts/submit_season_state.zip`
2. rolling member OOF: `scripts/build_champion_oof.py`
3. final metadata-only builder: `scripts/build_tree_reblend.py --cat-weight 0.30`
4. required official data: `train.csv`, `test.csv`, `sample_submission.csv` 및 해당 실험이
   요구하는 공식 historical file
5. 일반 runtime: project `.venv`; HGB 재학습 시에만 Python 3.12 + NumPy 1.26.4의
   `.venv312`

Builder는 `model/meta.json` 하나만 변경되는지 검사한다. 그러나 freeze 이후에는 기존
artifact를 보존하며, 재생성 결과를 동일 champion이라고 간주하려면 반드시 SHA-256까지
다시 검증해야 한다.

`tests/test_row_independence.py`는 `ROWINDEP_TRAIN`과 `ROWINDEP_ZIP` 환경변수를 이미
지원한다. 저장소 `data/train.csv`가 없는 환경의 실패는 코드 결함이 아니라 fixture 경로
문제이며 다음처럼 실행한다.

```bash
OMP_NUM_THREADS=1 \
ROWINDEP_ZIP=artifacts/sub_tree_reblend_w0p300.zip \
ROWINDEP_TRAIN=/path/to/official/train.csv \
python tests/test_row_independence.py artifacts/sub_tree_reblend_w0p300.zip
```

## Exploration freeze

High-margin inventory 결론은 **NO HIGH-MARGIN SUBMISSION CANDIDATE**다. 공식 main
test-visible predictor 47/47을 champion이 이미 사용하고, SAFE 신규 raw predictor가
없으며, 주요 model/representation 축도 검증이 끝났다. 사전 `2024 +10 BSS` 기대 근거가
없어 Stage 1을 시작하지 않았다.

탐색은 다음 중 하나가 있을 때만 재개한다.

- 새로운 공식 데이터 또는 규칙/운영진 답변 변경
- 기존 데이터 해석 오류 또는 champion 구현 버그 발견
- 누락된 구조적 정보축이 명확한 근거와 함께 발견
- 2024 `+10 BSS` 이상을 기대할 근거가 있는 완전히 새로운 방법

새 알고리즘 이름, 작은 파라미터·blend weight 변경, 작은 slice는 재개 조건이 아니다.

최종 action: **READY — KEEP AND SUBMIT 21ST CHAMPION**. 최신 제출 운영 감사는
`docs/FINAL_SUBMISSION_STATUS.md`를 따른다.

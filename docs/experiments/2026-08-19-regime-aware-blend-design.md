# Regime-aware ensemble 연구 설계

작성일: 2026-08-19  
상태: 설계 완료, 실행 입력 부재로 실험 대기

## 1. 저장소 감사 결과

요청서에 적힌 `R11-C-a28`, LB 약 997.30은 현재 저장소의 최신 기준점이 아니다.
`README.md`, 최근 커밋, 보존 제출물을 서로 대조한 결과 현재 검증된 최고 제출은
`artifacts/submit_r10_pitcher_w418194.zip`이고 LB BSS는
`1016.4442212358`이다.

현재 champion의 예측 순서는 다음과 같다.

1. `ours` stage: `0.4 HGB + 0.4 CatBoost + 0.2 player-embedding NN` 뒤
   고정 logit intercept calibration(`b=-0.0508596022900133`)
2. `team` stage: `0.9 calibrated LightGBM stack + 0.1 team player-embedding NN`
3. stage blend: `0.4 ours + 0.6 team`
4. 투수 season-to-date form: `M=75`, `alpha=0.28`, 정규시즌 R만 적용
5. 타자 season-to-date form: `M=50`, `alpha=0.11250648296393913`, R만 적용
6. train 기간 정규시즌 기록이 없는 투수의 R 행에 ID-free LightGBM expert를
   `0.4181944103545871` 혼합

마지막 세 단계의 분기와 lookup은 모두 현재 행의 값과 train-only 고정 테이블만
사용한다. 보존 ZIP의 메타에는 학습 행 수 `1,475,092`, target season 2025,
train max season 2024가 기록되어 있다.

현재 작업 트리에는 요청서가 지목한 `experiments/`, `tests/`, `submit/`,
`CLAUDE.md`가 없다. 제출 원본 디렉터리 이름은 `submission/`이며, 기존 실험은
주로 `scripts/`, `README.md`, Git commit history, `artifacts/*.zip`에 남아 있다.
또한 `submission/` 자체는 최신 champion이 아니라 초기 4-LightGBM 구성이다.
따라서 성공 후보를 이 디렉터리에 무심코 덮어쓰면 성능 회귀가 발생한다.

## 2. 확인된 성공/실패 축

### 성공하여 champion에 남은 축

- 이질적 HGB/CatBoost/NN/LightGBM stage 결합
- calibrated stage endpoint 유지
- 투수 season-to-date form
- 타자 season-to-date form
- 신규 정규시즌 투수용 ID-free expert
- 콜드스타트 expert의 LB 이차식 최적 비중 `0.4181944103545871`

### 실패하여 반복하지 않을 축

- 단조 LightGBM 비중 과대 적용
- residual cell correction: 로컬 개선, LB `-11.7877`
- 타자 form `alpha=0.20`: LB `-12.5901`
- 콜드스타트 expert `w=0.75`: LB `-5.2857`
- 콜드스타트 3-seed 평균: LB `-0.6496`
- 특정 주자/손 조합의 콜드스타트 `w=0.60`: LB `-1.0311`
- 콜드스타트 위에서 타자 alpha 재최적화: LB `-0.6690`

기본 모델 추가, 일반 calibration, seed ensemble, 단순 feature ablation은 이번
round의 주 연구 축으로 삼지 않는다.

## 3. 실험 입력과 누수 방지 계약

각 outer fold는 두 종류의 prediction을 가져야 한다.

- inner: `train <= t-1`로 학습한 모델이 season `t`에 낸 prediction
- validation: `train <= t`로 다시 학습한 모델이 season `t+1`에 낸 prediction

가중치, local regime optimum, shrinkage strength 선택은 **inner 데이터에서만**
수행하고, 고정된 결합기를 validation에 적용한다. validation label로 결합기를
다시 맞추지 않는다.

필수 fold는 다음과 같다.

| inner | validation | 보고 이름 |
| ---: | ---: | ---: |
| 2021 | 2022 | 2022 |
| 2022 | 2023 | 2023 |
| 2023 | 2024 | 2024 |

OOF artifact에는 최소한 아래 배열이 필요하다.

- `y_inner`, `y_val`
- `p_inner_hgb`, `p_val_hgb`
- `p_inner_cat`, `p_val_cat`
- `p_inner_nn`, `p_val_nn`
- `p_inner_team`, `p_val_team`
- `p_inner_team_nn`, `p_val_team_nn`
- 보정 완료 endpoint인 `p_inner_ours_stage`, `p_val_ours_stage`
- 보정 완료 endpoint인 `p_inner_team_stage`, `p_val_team_stage`
- 비교 기준인 `p_inner_champion_base`, `p_val_champion_base`
- 기존 form까지 적용한 비교용 `p_inner_champion_form`, `p_val_champion_form`
- 진단/행별 regime 컬럼: `game_type`, `balls_before`, `strikes_before`,
  `pitcher_hand`, `batter_hand`, `base_state`, `inning`, `top_bottom`,
  `asof_pitcher_n`
- fold train에서만 만든 `pitcher_seen` boolean

`pitcher_seen`은 validation 선수 집합이나 빈도로 만들지 않는다. 각 fold의 학습
cutoff 이하 정규시즌 투수 ID 고정 집합에 현재 행의 `pitcher_id`가 있는지만 본다.

현재 저장소의 검증 스크립트가 기대하는 공식 train 경로
`../open/data/train.csv`, 기존 이종 모델 학습 코드 `../LG-AIMERS_9TH`, OOF cache
`artifacts/validation_hetero/exp23_cache`는 이 머신에 존재하지 않는다. 공식 데이터나
동등한 OOF artifact가 복구되기 전에는 수치 결과를 만들거나 후보를 채택하지 않는다.

## 4. Member diagnostic

원시 5개 멤버와 보정 완료 두 stage를 분리해 진단한다.

- overall Brier
- R/F
- exact count state(12개)
- pitcher hand, batter hand, hand combination
- seen/unseen pitcher
- prediction Pearson correlation
- residual(`prediction - y`) correlation
- 모델 A가 기준보다 맞고 B가 틀리는 영역: 행별 squared-error 차이의 표본 수,
  평균, 표준오차

저표본 cell은 진단 표에는 남기되 regime 결합 후보로 자동 채택하지 않는다.
결과는 JSON과 Markdown을 모두 보존한다.

## 5. Brier-optimal constrained blend

목적함수는 다음과 같다.

```text
min_w mean((P w - y)^2) + rho * ||w - w_anchor||^2
subject to w_i >= 0, sum(w_i) = 1
```

첫 후보는 보정 완료 stage 두 개만 사용하고 `w_anchor=(0.4, 0.6)`으로 둔다.
이는 champion의 calibrated endpoint를 보존한다. 원시 5개 멤버 평평 결합은 기존
stage calibration을 제거하므로 진단 및 보조 후보로만 취급한다.

각 fold에서 inner로 학습한 weight와 validation 결과를 기록한다. 별도로 전체 OOF를
합친 descriptive global optimum도 출력하지만, 이는 같은 행에 학습·평가한 값이므로
채택 판정에 사용하지 않는다. fold weight의 표준편차와 범위를 출력하고, 멤버별
range가 `0.20`을 넘으면 불안정 경고를 낸다.

`rho` 후보는 anchor 쪽 수축 강도로 해석 가능한 소수의 사전 등록 grid만 쓴다.
validation이나 test 결과를 보고 연속적으로 미세 조정하지 않는다.

## 6. Hierarchical/shrunk regime blend

우선순위는 다음 네 가지로 제한한다.

1. `game_type`
2. `pitcher_hand × batter_hand`
3. exact count state
4. `game_type × count_bucket`

`count_bucket`은 현재 행의 balls/strikes만으로 `ahead`, `even`, `behind`를 만든다.
local optimum을 바로 사용하지 않고 다음처럼 global 쪽으로 수축한다.

```text
lambda_g = n_g / (n_g + tau)
w_g = lambda_g * w_local_g + (1 - lambda_g) * w_global
```

`tau`는 inner OOF에서만 선택한다. 기본 후보는 `500, 2_000, 10_000, 50_000`이며,
local cell이 최소 표본 수를 만족하지 않으면 global weight를 그대로 사용한다.
테스트 cell 빈도는 fitting, 후보 선택, 수축 강도 계산 어디에도 사용하지 않는다.

## 7. 비교와 채택 규칙

각 방법은 아래 세 출력을 분리한다.

- A: 기존 champion base(stage blend)
- B: 새 blend base
- C: 새 blend base + **기존 고정** pitcher/batter season-form endpoint

콜드스타트 expert도 별도 단계로 유지해 새 blend 자체의 효과와 섞지 않는다. 새
blend가 채택된 뒤에만 기존 expert와의 합성 결과를 마지막 안전성 확인으로 본다.

보고 표는 다음 열을 갖는다.

| method | 2022 | 2023 | 2024 | mean | worst | recent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

각 셀에는 champion 대비 Brier delta와 해당 fold prevalence로 환산한 BSS delta를
같이 기록한다. 채택 조건은 모두 만족해야 한다.

1. 2024 fold 개선
2. 3개 fold 중 최소 2개 개선
3. pooled/mean 개선
4. worst fold 손실이 사전 등록 허용치 이하
5. form endpoint를 붙인 C에서도 개선 방향 유지

## 8. Row-independence와 lookup 안전성

결합 runtime은 각 행의 regime label로 train-only 고정 weight table의 한 행을 읽을
뿐이어야 한다. 다음 입력에서 동일한 row A의 결과가 같아야 한다.

1. A 단독
2. 전체 batch의 A
3. shuffle된 batch의 A
4. subset의 A
5. reverse된 batch의 A

트리/룩업 경로는 `atol=0`을 요구한다. NN batch 연산 때문에 기존 champion에 이미
문서화된 float32 1 ULP 흔들림이 재현될 경우에만 `atol=6e-8`을 허용하고 이유와
적용 범위를 테스트 이름에 남긴다.

추가로 weight artifact를 만드는 함수에는 validation label을 뒤집거나 validation
row를 제거해도 inner-derived global/local weight가 바뀌지 않는 테스트를 둔다.

## 9. 배포 gate

실험이 채택 조건을 통과하기 전에는 `submission/`이나 champion ZIP을 수정하지
않는다. 통과한 경우에만 다음 순서로 배포한다.

1. train <= 2024 OOF만으로 최종 global/regime weight table 생성
2. champion 보존 ZIP을 새 candidate 디렉터리에 풀어 runtime과 JSON 추가
3. source/candidate prediction parity
4. row-independence 테스트
5. clean temporary directory에서 package smoke test
6. ZIP 크기와 CPU inference 시간 기록
7. 구조적으로 다른 후보를 최대 2개만 보존


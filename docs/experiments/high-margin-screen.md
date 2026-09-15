# 21차 기준 high-margin candidate screening

실행일: 2026-08-22

## 결론

**NO HIGH-MARGIN SUBMISSION CANDIDATE**

21차 `CatBoost 30% + LightGBM team 70%`를 절대 기준으로 고정했다.

- LB: `1058.074429882`
- ZIP: `artifacts/sub_tree_reblend_w0p300.zip`
- SHA-256: `8ce93a197efea18cf89b7b988c59a8b369d683983152b24b74976a96885da7cb`

22차 two-strike는 LB `1056.264418601`로 champion보다 `-1.810011281` 낮았다.
이 축은 CLOSED이며 count 분리, expert 구조, weight를 포함해 재탐색하지 않는다.

이번에는 새 모델을 학습하지 않았고 제출 ZIP도 만들지 않았다. 이유는 inventory 후
남은 아이디어 중 **2024에서 +10 BSS 이상을 기대할 사전 근거를 가진, 기존 실험과
중복되지 않는 구조적 후보가 0개**였기 때문이다. 약한 아이디어를 Stage 1에 올리지
말라는 실험 계약을 그대로 적용했다.

## 새 submission gate

앞으로 submission candidate는 다음을 모두 만족해야 한다.

1. 2022 ΔBSS `> 0`
2. 2024 ΔBSS `>= +10`
3. 2024 gain `>= 1.5 × paired 2SE`
4. 세 fold median `> 0`
5. seed 42/43/44에서 2022·2024 방향 안정
6. 2023-only gain으로 평균을 구제하지 않음
7. R/F 한쪽의 큰 악화 없음
8. residual complementarity 또는 명확한 구조적 근거
9. 완전한 row independence
10. 충분한 효과 크기가 없으면 ZIP 생성 금지

`+1~5 BSS` 후보는 통계적으로 유의해도 제출 후보로 승격하지 않는다.

## Inventory 범위

현재 작업트리의 HANDOFF와 모든 experiment 문서뿐 아니라, 선행 구조 실험이 보존된
`/Users/wooh/Documents/dev/LG-Aimers-9th/docs/experiments/`도 확인했다. 현재 champion의
47개 공식 predictor가 모두 사용 중이라는 feature inventory도 재확인했다.

이미 검증되어 닫힌 구조 축은 다음과 같다.

| 계열 | 실제 검증 | high-margin 관점의 종료 근거 |
| --- | --- | --- |
| 다른 tree/tabular learner | Histogram XGBoost, ExtraTrees 계열, TabM, FT-Transformer | TabM/FT/XGB의 2024 standalone gain이 모두 음수. 현재 작업트리 XGB 5% blend도 `-1.90 BSS` |
| local/retrieval representation | exact kNN, prototype/codebook | kNN의 2024 blend gain은 사실상 0, prototype은 2023 gain 집중 `94.18%` 및 2022 악화 |
| reliability/shrinkage representation | as-of posterior mean/variance, cold-start shrinkage, hierarchical EB | reliability seed 43에서 2024 방향 반전; EB는 2022·2024 모두 악화 |
| nonlinear representation | masked-feature pretraining, sparse oblique projection, additive shape, latent interaction | 모두 KILL; masked pretraining 2024 standalone `-0.000638`, latent interaction은 2022 악화/pooled 음수 |
| robust/objective family | GroupDRO, direct Brier, hard/uncertainty weighting | champion blend가 음수 또는 `1e-7` 수준이고 seed 안정성 실패 |
| residual/specialist family | residual LightGBM, role-state, two-strike/count expert | residual corr/효과 크기 부족 또는 실제 LB 실패; 사용자 지정 CLOSED 축 |
| privileged/identity family | TrackMan LUPI, ID CTR, low-rank pitcher-batter, existing NN embedding | temporal 전이 실패, unseen-pair 일반화 실패 또는 2022·2024 악화 |

## 후보 선정 전 구조 검토

최대 3개 후보를 고르는 대신, 아래 세 방향을 pre-training evidence gate에서 검토했다.
셋 모두 기존 실험과 중복되거나 +10 근거가 없어 **후보로 선정하지 않았다**.

### 1. Attention/ensemble형 tabular neural family

- champion과 다른 점: feature token attention 또는 여러 prediction head로 tree와 다른
  conditional structure를 표현할 수 있다.
- 중복 감사: FT-Transformer와 TabM이 이미 temporal OOF로 검증됐다. 기존 NN도
  champion에서 제거될 만큼 모든 시즌에서 tree보다 약했다.
- 기대 residual: 고차 categorical interaction과 tree leaf 경계의 보완.
- inference 안전성: frozen preprocessing과 현재 row만 쓰면 안전하다.
- coverage: 100%.
- `2024 +10` 근거: **없음**. FT-Transformer와 TabM의 2024 standalone gain이 각각
  `-0.0001278`, `-0.001008` Brier였고 hard-error 보완율도 각각 `26.01%`, `23.47%`였다.
- 판정: 이름이나 backbone만 바꾼 TabNet/DCN/NODE는 신규 정보축이 아니므로 학습 금지.

### 2. Retrieval/prototype/reliability representation

- champion과 다른 점: point estimate 대신 local historical neighborhood, codebook posterior,
  표본 신뢰도를 표현한다.
- 중복 감사: exact kNN, KMeans codebook, Beta posterior reliability, cold-start shrinkage,
  hierarchical EB를 이미 검증했다.
- 기대 residual: 저표본 선수와 희귀 context.
- inference 안전성: train-only frozen index/lookup이면 안전하다.
- coverage: 100%, 단 유효 historical support는 부분집합.
- `2024 +10` 근거: **없음**. kNN 선택안의 2024 blend gain은 Brier
  `+0.000000612`, reliability는 seed 43에서 음수로 반전했고 prototype은 2022가
  악화했다. low-n calibration 개선도 champion residual 개선으로 전이되지 않았다.
- 판정: 기존 EB의 재매개변수화 또는 kernel/soft assignment는 닫힌 축 재탐색이므로 학습 금지.

### 3. Learned nonlinear feature representation

- champion과 다른 점: self-supervised encoder, oblique feature projection, additive shape,
  latent pitcher-batter geometry로 axis-aligned tree가 놓친 구조를 표현한다.
- 중복 감사: masked pretraining, sparse random projection ET, additive model, low-rank
  residual interaction이 모두 temporal OOF로 검증됐다.
- 기대 residual: correlated numeric manifold, smooth marginal, unseen matchup interaction.
- inference 안전성: train-only frozen transform이면 안전하다.
- coverage: row representation은 100%; identity interaction은 양쪽 선수가 seen인 일부.
- `2024 +10` 근거: **없음**. masked pretraining은 2024 standalone이 크게 악화했고,
  low-rank interaction은 2024만 소폭 양수였지만 2022/2023과 pooled가 음수이며 핵심
  unseen-pair subset gain도 음수였다.
- 판정: 새로운 architecture 명칭만으로 이 실패 근거를 뒤집을 수 없어 학습 금지.

## Stage 1 — Cheap screening

실행 후보: **0개**.

Stage 1은 계산 자원이 없어서 생략한 것이 아니다. 사전에 요구된 다음 조건을 적용한 결과다.

- 기존 실험과 구조적으로 중복되지 않을 것
- champion residual을 줄일 구체적 근거가 있을 것
- 특히 2024에서 최소 `+10 BSS`를 기대할 근거가 있을 것

세 조건을 모두 만족하는 후보가 없으므로 seed 42의 `<=2021 → 2022`,
`<=2023 → 2024` 학습을 시작하지 않았다.

## Stage 2 — Full validation

Stage 1 통과 후보가 없으므로 미실행했다. 따라서 새 후보에 대한 seed 42/43/44,
paired 2SE, R/F, calibration, hard-error, prediction/residual correlation 측정치는 없다.
이는 후보를 PASS 처리한 것이 아니라 **pre-training rejection**이다.

## 무결성 및 변경 여부

- 21차 champion SHA-256 재계산: 일치
- champion ZIP 수정: 없음
- 모델/학습/추론 코드 수정: 없음
- 새 모델 artifact: 없음
- 새 submission ZIP: 없음
- 22차 two-strike 재탐색: 없음

최종 판정: **NO HIGH-MARGIN SUBMISSION CANDIDATE**. 21차 champion을 유지한다.

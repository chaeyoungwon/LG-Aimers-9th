# Architecture / Representation 재개 사전 감사

실행일: 2026-08-25

## 결론

**ARCHITECTURE AXIS IS NOVEL BUT LOW EXPECTED VALUE**

정확한 ID-aware field attention과 DeepFM/FFM 구현은 없으므로 제한적인 구조 신규성은
남아 있다. 그러나 이를 지지해야 할 2022·2024 공통 sparse/cold-start 오류와 안정적인
low-rank interaction은 관측되지 않았다. 실제 FT-Transformer, TabM, player-embedding
MLP, exact kNN, prototype/codebook 및 weighted low-rank residual factorization은 이미
temporal OOF로 검증됐고 모두 champion을 안정적으로 보완하지 못했다.

따라서 재개 후보는 **NONE**, 다음 행동은 **DO NOT TRAIN**이다. 신규 모델, inference
코드, submission ZIP을 만들지 않았으며 21차 champion을 그대로 유지했다.

- champion: CatBoost 30% + LightGBM team 70%
- LB: `1058.074429882`
- ZIP: `artifacts/sub_tree_reblend_w0p300.zip`
- SHA-256: `8ce93a197efea18cf89b7b988c59a8b369d683983152b24b74976a96885da7cb`

## A. Existing architecture inventory

문서뿐 아니라 실제 구현, cache 및 summary를 대조했다. `FULLY TESTED`는 해당
사전 고정 recipe가 2022/2023/2024 temporal OOF를 완주했다는 뜻이며 architecture의
모든 hyperparameter를 탐색했다는 뜻은 아니다.

| Family | Previous status | Actual depth of test | Closed? |
| --- | --- | --- | --- |
| Player-ID embedding MLP | **FULLY TESTED** | pitcher/batter 8-d embedding + ID dropout + 128/64 MLP; 두 NN member rolling OOF 및 실제 submission 구성 | yes |
| FT-Transformer | **FULLY TESTED** | official `rtdl` 2-block, 128-d/8-head 계열 default, seed 42, ID-free HGB-52, 3 temporal folds, complementarity gate | yes as ID-free attention |
| TabM | **FULLY TESTED** | official TabM, k=8, 2x128, 8 epochs, seed 42, 3 folds | yes |
| TabTransformer exact implementation | **NOT TESTED** | 별도 package/model 없음; FT의 field-token attention과 구조적으로 크게 겹침 | no exact implementation; low value |
| ID-aware field attention | **NOT TESTED** | ID embedding MLP와 ID-free FT는 각각 있음; 둘을 결합한 attention은 없음 | no; screened out here |
| Exact train-only kNN | **FULLY TESTED** | full/compact, k 16/32/64/128, uniform/inverse, shrinkage, 3 folds, row-independence tests | yes |
| X-only prototype/codebook | **FULLY TESTED** | K 128/512/2048, alpha 0/32/128, 3 folds, repeatability/subset audit; 이후 실제 LB locality failure도 기록 | yes |
| Learned metric / Siamese retrieval | **SCREENED ONLY** | 학습 metric 구현 없음; fixed metric/locality와 prototype의 temporal transfer 실패로 pre-training rejection | no exact implementation; very low value |
| Pitcher-batter low-rank ALS | **FULLY TESTED** | residual marginals + rank 2/4/8 weighted ALS, weights `.05-.50`, seed 42/43, unseen-pair audit | yes |
| FM / FFM / DeepFM exact | **NOT TESTED** | exact library/model 없음; 핵심 player-pair factor sharing은 ALS, independent embeddings은 MLP로 각각 직접 검증 | no exact implementation; low value |
| Masked-feature pretraining | **FULLY TESTED** | label-free encoder pretraining + supervised fine-tune, temporal OOF/seed control | yes |
| Additive/oblique/latent representation | **FULLY TESTED** | stable additive shape, sparse oblique projection, latent pair residual 각각 temporal OOF | yes |
| GNN / graph attention | **MENTIONED ONLY** | 구현 없음; pair residual repeatability와 unseen-pair transfer 부재 때문에 선행 low-rank 감사에서 종료 권고 | no; very low value |

실제 코드는 각각 `src/nn_embed.py`, 선행 저장소의
`scripts/validate_model_diversity.py`, `validate_local_knn.py`,
`validate_prototype_codebook.py`, `validate_interaction_latent_oof.py`에 존재한다.

## B. Existing NN failure diagnosis

### primary reason

**TREE DOMINANCE + LOW SAMPLE EFFICIENCY**다. Player embedding NN은 단순 미구현
prototype이 아니라 실제 5-member submission의 20%/10% stage component였다. Rolling
BSS는 다음과 같다.

| Season | Cat | team LightGBM | ours NN | team NN |
| ---: | ---: | ---: | ---: | ---: |
| 2021 | 1369.6 | 1188.8 | 239.0 | 487.2 |
| 2022 | 2394.4 | 2392.5 | 1951.9 | 1949.6 |
| 2023 | -836.4 | -1025.3 | -2497.1 | -3063.5 |
| 2024 | 783.2 | 856.4 | 190.9 | -37.6 |

두 NN을 제거한 tree-only 재배합은 proxy `+31.89 BSS`, 실제 LB `+6.5207`을 기록했다.
즉 NN removal은 단순 local noise가 아니라 hidden LB에서도 올바른 방향이었다.

### secondary reason

**TEMPORAL INSTABILITY / FEATURE REPRESENTATION FAILURE**다. 기존 MLP는 independent
pitcher/batter embedding을 context와 concatenate하므로 explicit bilinear interaction이나
field attention은 없다. 그러나 더 강한 FT-Transformer도 2022/2023/2024 current 21차
champion 대비 Brier gain `-.000398/-.002921/-.000538`로 모두 악화했다. TabM은
`-.001087/-.003336/-.001418`이었다.

이는 단순 optimization failure로 단정하기 어렵다. 두 모델 모두 official/default-like
optimizer contract와 전체 수십만 행을 사용했고 finite prediction을 냈으며, 문제는
미수렴 증거보다 미래 season에서의 calibration/sample-efficiency 열세였다.

### does it generalize to Transformer/FM?

- Transformer: 상당 부분 **yes**. 실제 FT 결과가 같은 방향으로 약했다. 단 ID-aware
  attention exact 구조까지 실패했다고 과장하지 않는다.
- FM: 부분적으로 **yes**. Independent ID embedding과 pair low-rank ALS가 모두
  실패했지만 exact DeepFM의 nonlinear main tower 자체는 미검증이다.
- 결론: 정확한 architecture 신규성은 남아도 architecture-independent한 official-signal
  한계와 temporal instability 때문에 expected value가 낮다.

## C. Tree interaction saturation

### evidence

현재 ZIP의 실제 metadata/model을 확인했다.

- CatBoost: 1,000 trees, depth 6, `max_ctr_complexity=4`, 65 features. 중요도 상위에
  `game_type 17.47%`, `season 12.11%`, pitcher/batter state, 양 team, `hand_combo`,
  month, `same_hand`, recent-vs-career가 함께 존재한다.
- team LightGBM: 70 features, `eng31`, `leaf63`, `decay85`, `mono63` booster를 두 group으로
  stacking한다. Raw pitcher/batter ID, team, count, handedness, recent/cumulative rate,
  season-state derived feature가 동일 tree paths에서 상호작용할 수 있다.
- Cat/team prediction은 동일하지 않다. 21차는 이 diversity를 30:70으로 사용해 실제 LB를
  개선했다.
- 기존 conditional Cat/team MoE는 17개 current-row condition에서 전-season stable-positive
  coverage가 0%여서 interaction routing headroom을 찾지 못했다.
- explicit two-strike interaction은 local positive였지만 실제 LB `-1.8100`으로 실패했다.

### remaining headroom

Tree가 모든 interaction을 완전히 포착했다고 증명할 수는 없다. 남은 이론적 headroom은
continuous player similarity와 unseen combination에 대한 parameter sharing이다. 그러나
아래 cold-start 및 low-rank 진단에서 이 headroom이 반복 loss로 나타나지 않는다.
따라서 **존재 가능하지만 high-margin 근거 없음**이다.

## D. Sparse/cold-start evidence

공식 train에서 validation season 이전 행만 reference로 사용했다. Current row의
`asof_*_n`과 frozen historical pair/count support로 subgroup을 정의했다. Gain을 튜닝하거나
feature로 쓰지 않았다.

| Subgroup | 2022 share / Brier excess | 2024 share / Brier excess | Common weakness? |
| --- | ---: | ---: | --- |
| pitcher history `<100` | 3.07% / `-.006812` | 3.12% / `-.000697` | no; both easier |
| batter history `<100` | 3.07% / `-.021153` | 2.65% / `-.001896` | no; both easier |
| unseen matchup | 50.86% / `-.002572` | 48.84% / `-.000162` | no; both easier |
| matchup history `1-19` | 24.45% / `+.000858` | 25.30% / `+.000255` | weak yes |
| pitcher x count support `<20` | 22.06% / `-.002929` | 23.47% / `-.000455` | no |
| batter x count support `<20` | 17.19% / `-.007739` | 13.11% / `-.000618` | no |

`excess = subgroup Brier - season overall Brier`다. Cold/unseen/rare combination이 반복
취약하다는 핵심 전제는 반박된다. 유일한 공통 양수인 matchup 1-19도 2024 excess가
`.000255`로 작고, exact pair residual repeatability가 거의 0이어서 learnable headroom으로
볼 수 없다.

## E. Candidate architecture audit

Score는 0(없음/매우 낮음)~5(매우 높음)이다. Implementation cost만 5가 가장 비싸다.

| Rank | Family | Novelty | Expected value | Structural novelty | 2022 evidence | 2024 evidence | Residual independence | Sparse potential | Temporal robustness | +10 plausibility | Rule safety | Cost |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | ID-aware field attention / TabTransformer | PARTIALLY NOVEL | LOW | 3 | 1 | 1 | 1 | 2 | 1 | 0 | 5 | 5 |
| 2 | DeepFM/FFM exact | PARTIALLY NOVEL | VERY LOW | 3 | 0 | 1 | 1 | 2 | 1 | 0 | 5 | 4 |
| 3 | learned metric retrieval | PARTIALLY NOVEL | VERY LOW | 3 | 1 | 1 | 2 | 2 | 1 | 0 | 5 | 5 |

FT 자체, exact kNN, prototype 및 low-rank ALS는 `REDUNDANT / VERY LOW`라 후보표에서
제외했다. 세 partial novelty 모두 2024 `+10` plausibility가 0이므로 실제 REOPEN은 없다.

## F. Retrieval/prototype evidence

### result

- Exact kNN selected blend Brier gain: 2022 `+.000000338`, 2023
  `+.000023839`, 2024 `+.000000612`; 96.17%가 2023에 집중.
- kNN standalone은 global prevalence보다 pooled `.013481` 나빴다.
- residual correlation `.965301`; 2024 closest-distance quartile만 약하게 양수였으나
  season별 distance-quality ordering이 반복되지 않았다.
- Prototype selected blend: `-.000013855/+.000251402/+.000001673`; 94.18%가
  2023에 집중, residual correlation `.994204`, K에 따라 2024 부호 반전.
- Prototype/locality submission은 이후 LB `1013.6071`, 당시 champion 대비 `-3.4234`로
  실제 전이에도 실패했다.

### temporal safety

두 구현 모두 cutoff-train scaler/reference/centroid만 사용했고 query는 current row에서
frozen train bank로만 향했다. Exact kNN single/full/shuffle도 동일했다. 규칙상 안전하다.

### expected value

Learned metric은 exact implementation 면에서 새롭지만, supervised metric target이
이미 반복되지 않는 historical labels/residual을 더 강하게 맞출 위험이 크다. Locality
quality가 2022·2024에서 같은 방향으로 residual을 설명하지 않아 **VERY LOW**다.

## G. Low-rank interaction evidence

### result

기존 pair residual ALS에서 historical-to-future pair residual correlation은
2022/2023/2024 `-.00031/+.00817/+.01133`, sign agreement는 약 50%였다. Rank 4,
weight `.05`는 `-.00002290/-.00002688/+.00001717` Brier gain이고, 핵심 unseen-pair
both-seen 191,311행 gain은 `-.00004226`였다. Seed 43도 동일 방향이다.

### singular structure

추가 학습 없이 cutoff-safe coarse interaction matrix를 진단했다. Cell effect는 main
row/column effect를 제거하고 support `n/(n+20)`로 shrink했다. Rank는 validation 결과로
선택하지 않았다. `2019-2021` matrix의 singular energy와 독립 후속 block
`2022-2023`의 common-cell effect repeatability는 다음과 같다.

| Matrix | Shape | Rank-1 / rank-2 / rank-4 energy | Later corr / sign agreement |
| --- | ---: | ---: | ---: |
| pitcher x count | 560x12 | 12.88% / 24.28% / 44.64% | `.0605 / 52.50%` |
| batter x count | 583x12 | 12.02% / 23.24% / 43.42% | `.0695 / 51.70%` |
| team x team | 12x12 | 36.30% / 69.85% / 88.79% | `.3509 / 59.78%` |
| pitcher hand x count | 2x12 | 99.96% / 100% / 100% | `.2503 / 58.33%` |
| batter hand x count | 2x12 | 99.89% / 100% / 100% | `.1136 / 50.00%` |

Player x count는 low-rank concentration도, temporal repeatability도 약하다. Team/hand
matrix는 cardinality가 작아 수학적으로 low rank이며 현재 Cat/LightGBM이 이미 직접
표현한다. 이것은 DeepFM을 정당화하는 신규 sparse-sharing 근거가 아니다.

### expected value

Exact DeepFM은 main tower와 FM term의 공동 최적화라는 제한적 신규성이 있다. 하지만
FM term이 해결해야 할 player-context 및 player-pair matrix의 사전 구조가 없으므로
**VERY LOW**다.

## H. Attention evidence

### result

FT-Transformer를 현재 21차 champion과 다시 직접 비교했다.

| Season | Standalone Brier gain | Prediction corr | Residual corr | Champion top-10% error win rate |
| ---: | ---: | ---: | ---: | ---: |
| 2022 | `-.000398054` | `.956222` | `.998932` | 59.79% |
| 2023 | `-.002920905` | `.957626` | `.998715` | 30.00% |
| 2024 | `-.000537787` | `.855548` | `.998850` | 67.50% |

2022/2024 champion hard-error slice에서 국소 win rate는 보이지만 그 slice는 target을
본 사후 oracle이며 inference route로 사용할 수 없다. 전체 Brier는 둘 다 악화하고
residual correlation은 `.99885+`라 high-margin 독립 축이 아니다. TabM residual
correlation도 `.997160-.998696`이고 전 fold standalone이 악화했다.

### expected value

ID-aware attention은 exact FT와 다르지만, ID embedding MLP의 큰 열세와 low-rank pair
비반복성을 동시에 극복해야 한다. Attention이 놓친 ordering도 없다. 입력은 순서 없는
tabular fields이며 현재 문제의 투구 sequence 자체가 제공되지 않는다. 따라서 **LOW**다.

## I. Rule safety

- test-test dependency: 필요 없음; 금지
- batch dependency: BatchNorm, test-fitted normalization, cross-row attention 없이 row별
  field-token attention만 허용하면 없음
- row independence: current row + frozen train-derived vocab/model/reference로 PASS 가능
- external data: 사용하지 않음

Transformer의 self-attention은 한 행 내부 feature token 사이에서만 수행해야 한다.
Batch dimension을 sequence로 취급하거나 test player frequency, test reference bank,
test-set calibration을 사용하면 즉시 FAIL이다. Retrieval은 query-to-frozen-train만 허용한다.
이번에는 후보 runtime을 만들지 않았으므로 신규 dynamic row-independence 측정 대상은 없다.

## J. Best candidate

**NONE**

가장 가까운 이론적 후보는 ID-aware field attention이지만 다음 high-margin gate를 모두
충족하지 못한다.

- genuinely new: independent ID embeddings와 ID-free FT의 결합이라는 제한적 신규성
- champion may miss: context-conditioned continuous ID sharing 가능성
- affected coverage: seen IDs는 넓지만 실제 반복 취약인 matchup 1-19는 약 25%
- required effect: 2024 전체 +10을 위해 이 slice에서 약 `+39.53 subset BSS`, Brier
  `.00009874` 감소 필요
- observed comparable effect: kNN 2024 약 `+0.25 BSS`, prototype 약 `+0.67 BSS`,
  low-rank pair는 unseen-pair 악화; FT 전체도 악화
- 2024 +10 rationale: **없음**

따라서 Stage 1 recipe를 설계하지 않는다.

## K. Final decision

**ARCHITECTURE AXIS IS NOVEL BUT LOW EXPECTED VALUE**

Exact ID-aware attention/DeepFM이라는 구현 차이는 남지만, 이는 새 official information
axis가 아니다. 핵심 parameter-sharing 전제와 residual complementarity, 2022·2024 공통
오류, +10 effect-size 근거가 모두 부족하다. 신규성과 기대값을 분리해 partial novelty는
인정하되 architecture search를 재개하지 않는다.

## L. Next action

**DO NOT TRAIN**

- FT/TabM backbone·epoch·embedding dimension 재탐색 금지
- ID-aware attention/DeepFM/FFM Stage 1 금지
- metric learning, kNN distance, prototype K/soft assignment 재탐색 금지
- FM rank/regularization, GNN/graph attention 파생 금지
- production model 및 submission ZIP 생성 금지
- 21차 champion 유지


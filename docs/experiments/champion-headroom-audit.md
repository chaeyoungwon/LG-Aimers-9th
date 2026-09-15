# Champion Error Ceiling / Recoverable Headroom Audit

실행일: 2026-08-26

## 결론

**NO MATERIAL RECOVERABLE HEADROOM — STOP MODEL SEARCH**

21차 champion의 theoretical per-row oracle은 매우 크지만 배포 불가능하다. 정답을 보지
않고 2022에서 선택해 2024로 넘긴 가장 좋은 global candidate는 two-strike
`+4.34 BSS`, 반대 방향은 `+5.75 BSS`였다. 사전 정의 legal axis별 selector는 대부분
다음 season에서 악화했고, 두 normal season을 모두 본 retrospective stable single-axis
상한도 2024 `+3.87 BSS`였다. 세 수치 모두 +10 high-margin gate에 못 미친다.

더 결정적으로 이 회수 신호는 전부 22차에서 실제 제출 후 champion 대비
`-1.810011281 LB`를 기록한 two-strike 하나에서만 나온다. 따라서 숨은 test에서 입증된
deployable headroom은 0보다 크다고 주장할 수 없다. 현 evidence의 낙관적 local upper
proxy는 `<+5 BSS`, 실제 transfer evidence는 음수다.

- champion: CatBoost 30% + LightGBM team 70%
- LB: `1058.074429882`
- 1100 gap: `41.925570118`
- ZIP: `artifacts/sub_tree_reblend_w0p300.zip`
- SHA-256: `8ce93a197efea18cf89b7b988c59a8b369d683983152b24b74976a96885da7cb`
- 모델 학습: 없음
- test read: 없음
- inference/ZIP 변경: 없음

## 감사 계약과 candidate union

공식 train의 season 행 순서와 실제 저장 prediction의 길이/finite 여부를 검증했다.
동일 행 OOF가 남아 있는 16개 prediction만 포함했다.

`champion`, Cat, team, HGB, NN, team-NN, FT-Transformer, TabM, Hist XGBoost,
hard-weight, uncertainty `.5`, role-state, GroupDRO, LUPI auxiliary, LUPI distillation,
two-strike다. Two-strike는 저장된 3-seed LGB/Cat slice prediction과 실제 runtime의
75:25/50:50 formula로 재구성했다.

Forward residual LGB, TrackMan profile, hierarchical EB, stable additive,
prototype/codebook은 row-level fold prediction이 현재 artifact에 없으므로 문서 숫자로
복원하지 않고 union에서 제외했다. 이들의 기존 aggregate 판정은 해석에만 사용했다.

재현 코드는 `scripts/audit_champion_headroom.py`, 결과는
`artifacts/headroom_audit/`에 있다.

## A. Champion error decomposition

아래 class는 겹친다. 합계가 100%가 되어야 하는 partition이 아니다. `share`는 row
coverage이며, 괄호는 해당 mask가 차지하는 전체 Brier 합 비중이다.

| Error type | 2022 share | 2024 share | Recoverable? |
| --- | ---: | ---: | --- |
| CALIBRATION-LIKE | 25.22% (25.81%) | 25.18% (25.26%) | 같은 game_type×count bias 방향은 있으나 효과가 작고 반복 correction 후보 없음 |
| DISCRIMINATION-LIKE | 15.45% (22.37%) | 15.30% (19.43%) | top-error concentration은 크지만 사전 식별 불가 |
| SPARSE / RELIABILITY | 50.98% (50.43%) | 48.92% (48.89%) | no; Brier가 season 전체보다 오히려 낮음 |
| INTERACTION FAILURE proxy | 24.45% (24.53%) | 25.30% (25.32%) | matchup 1-19가 약하게 어려우나 low-rank/retrieval 전이 실패 |
| TEMPORAL PRIOR FAILURE | 0% | 0% | primary normal seasons에는 없음 |
| MODEL-FAMILY proxy, hand 2-2 | 36.83% (36.85%) | 36.45% (36.45%) | two-strike route retrospective +2.89/+3.87 overall BSS; hidden LB 실패 |
| UNEXPLAINED / IRREDUCIBLE-LOOKING | 19.42% (19.69%) | 19.63% (19.60%) | low model disagreement + coarse-state entropy >=.95 bit |

Sparse mask는 pitcher/batter history `<100` 또는 unseen matchup이다. Interaction proxy는
matchup history `1-19`다. 이들은 큰 coverage지만 high-loss 영역이 아니다. 2022/2024
unseen matchup Brier excess는 각각 `-.002572/-.000162`; low-history도 모두 음수였다.

### Champion error percentile geometry

| Season | Band | Mean Brier | Total Brier contribution | Target mean | Pred mean | Bias y-p |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2022 | top 5% | .453764 | 9.33% | .181510 | .635628 | -.454118 |
| 2022 | top 10% | .388280 | 15.97% | .303176 | .572916 | -.269740 |
| 2022 | top 20% | .342059 | 28.14% | .400202 | .536671 | -.136469 |
| 2024 | top 5% | .348109 | 7.03% | .634743 | .473450 | +.161292 |
| 2024 | top 10% | .332386 | 13.42% | .641592 | .477518 | +.164074 |
| 2024 | top 20% | .313694 | 25.33% | .643801 | .482450 | +.161351 |

큰 오류의 signed direction 자체가 바뀐다. 2022 top error는 주로 false-positive,
2024는 false-negative다. Top 5%의 반복 enrichment는 hand `1-1`, hand `2-2`,
`strikes<balls`, 일부 bases loaded였지만 winner를 사전 결정하는 deployable rule은
없었다. 2022 top 5%는 F가 71.66%로 shock 이전 높은 F prior의 반대 outcome에
집중됐으나 2024 F enrichment는 14.92%뿐이다.

## B. Candidate recovery

수치는 current 21차 champion 대비 BSS다. `Stable corrected rows`는 target을 본 row별
win share가 아니라 양 primary season에서 같은 legal condition/correction이 확인됐는지다.

| Candidate | 2022 gain | 2024 gain | Stable corrected rows | Deployable? |
| --- | ---: | ---: | --- | --- |
| two-strike | **+5.75** | **+4.34** | yes; 여러 broad condition에 걸치지만 실제 변경은 two-strike 24% | 규칙상 yes, **actual LB no** |
| team standalone | -28.24 | -5.18 | no | no |
| uncertainty `.5` | -23.30 | -9.16 | no | no |
| role-state | -25.65 | -50.17 | no | no |
| Cat standalone | -26.32 | -78.38 | top-error 일부 개선, overall 반복 악화 | no |
| FT-Transformer | -159.76 | -215.28 | top-error oracle win은 있으나 overall/residual 실패 | no |
| GroupDRO | -504.61 | -417.52 | no | no |
| LUPI auxiliary | -373.06 | -515.76 | no | no |
| LUPI distillation | -129.10 | -175.53 | no | no |
| hard-weight | -8376.42 | -8018.13 | badly different/calibration collapse | no |

Cat과 FT는 champion top-error에서 양수 Brier gain이 있지만 전체에서는 크게
악화한다. Top error는 y를 본 사후 정의라 selector로 쓸 수 없다. FT residual correlation도
`.99893/.99885`; two-strike도 `.999985/.999988`이다. 독립적인 high-margin residual
source가 아니다.

후보 positive-row 집합의 겉보기 다양성도 deployability를 만들지 못했다. Two-strike와
다른 후보의 positive-row Jaccard는 대체로 `.07-.15`, gain correlation은 약
`-.22~+.22`였지만 다른 후보가 primary season overall을 개선하지 않아 union route를
학습할 안정적 label이 없다.

## C. Oracle hierarchy

### per-row oracle

- all 16 candidates: 2022 `+26,567.99`, 2024 `+27,971.42 BSS`
- quality-filtered 6 candidates: 2022 `+3,824.97`, 2024 `+3,831.48 BSS`

**UNDEPLOYABLE ORACLE**이다. Binary y를 본 뒤 y=1이면 가장 큰 p, y=0이면 가장 작은
p를 골랐으므로 candidate diversity가 조금만 있어도 비현실적으로 커진다. 이는
recoverable headroom 추정치가 아니다.

### cross-fitted proxy

- 2022 global winner -> 2024: two-strike `+4.336 BSS`
- 2024 global winner -> 2022: two-strike `+5.754 BSS`
- 2022 legal-axis rules -> 2024: 8개 중 최고 `+4.336`, 나머지 다수
  `-4.23~-29.35 BSS`
- 2024 legal-axis rules -> 2022: 최고 `+5.754`, 다수 음수

이것이 **CROSS-FITTED DEPLOYABLE PROXY**다. Best axis가 global two-strike와 같은
결과여서 subgroup selector가 추가 headroom을 만들지 않았다.

### legal deployable oracle bound

두 season을 모두 본 뒤 동일 candidate, positive gain, 동일 prediction direction,
각 season 5,000행 이상을 요구한 retrospective stable route의 2024 결과:

| Legal axis | Stable route | 2022 overall | 2024 overall |
| --- | --- | ---: | ---: |
| game type | R -> two-strike | +5.66 | +1.55 |
| pitcher history | 500+ -> two-strike | +6.10 | +3.69 |
| batter history | 500+ -> two-strike | +7.09 | +3.29 |
| hand combo | 2-2 -> two-strike | +2.89 | **+3.87** |
| inning phase | 1-3, 7+ -> two-strike | +4.07 | +3.71 |
| matchup history | unseen, 1-19 -> two-strike | +4.44 | +2.81 |

최고 `+3.87`은 양 season을 모두 본 사후 상한이라 cross-fit보다도 낙관적이다.
Hand 2-2 condition coverage는 36.45%지만 candidate가 실제로 바꾸는 hand 2-2 ∩
two-strike 행은 2024 **8.88%**다. 그럼에도 +10 overall에 못 미친다.

정리:

- per-row oracle = huge, **UNDEPLOYABLE**
- cross-fitted proxy = `+4.34/+5.75`, 전부 two-strike
- legal deployable oracle bound = 2024 `+3.87`, retrospective

## D. Model disagreement

Badly different candidate가 dispersion을 부풀리지 않도록 primary 해석은 champion, Cat,
team, uncertainty, role-state, two-strike 6개 quality-filter union을 사용했다.

| Season | Band | Champion Brier | Mean pred std | Quality oracle BSS | Entropy >=.95 share |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2022 | high disagreement 20% | .239431 | .012418 | +7,176 oracle | 90.98% |
| 2022 | low disagreement 20% | .245252 | .002442 | +1,441 oracle | 97.10% |
| 2024 | high disagreement 20% | .247627 | .012110 | +7,188 oracle | 94.72% |
| 2024 | low disagreement 20% | .247246 | .002394 | +1,418 oracle | 98.16% |

High disagreement에서 oracle은 크지만 season-stable winner가 없다: case C, unstable /
nondeployable이다. Low disagreement 20%는 model range도 약 `.0072`, champion Brier는
거의 random baseline, coarse-state entropy는 거의 1 bit다: case B, information
limitation의 증거다. Oracle BSS는 여전히 y를 보고 고른 값이라 회수 가능량이 아니다.

## E. Stable recoverable headroom

- best legal condition coverage: hand 2-2 `36.45%`
- actual changed coverage: hand 2-2 x two-strike `8.88%`
- retrospective 2024 overall headroom: `+3.87 BSS`
- cross-fitted global 2024 proxy: `+4.34 BSS`
- main error type: two-strike discrimination
- independent hidden evidence: actual LB `-1.8100`

따라서 evidence-backed deployable headroom은 **local optimistic bound `<+5 BSS`**, hidden
transfer를 포함하면 **material positive로 확인되지 않음**이다. 이것이 현재 정책의
`<+5 -> 탐색 종료`에 해당한다.

## F. Goal feasibility

대회 BSS transform의 season prevalence denominator를 사용했다.

| Goal | 2024 required overall Brier gain | Judgment |
| ---: | ---: | --- |
| +10 BSS | `.000024981` | 현재 best cross-fit의 약 2.3배; 근거 없음 |
| +20 BSS | `.000049961` | 비현실적 |
| +40 BSS | `.000099923` | 비현실적 |

Hand 2-2 x two-strike 8.88%만 바꿔 +10을 얻으려면 해당 행에서 평균 Brier
약 `.00028134`를 줄여야 한다. 관측 route 효과보다 약 2.6배 크며 이미 실제 LB에서
반전됐다.

1100까지 필요한 LB `+41.9256`과 local BSS는 1:1이 아니다. 유일한 큰 성공인 21차는
local proxy `+31.89`가 LB `+6.52`로 약 20.4% 전이됐다. 이 비율을 미래 예측식으로
사용할 수는 없지만 같은 수준이라면 +41.93 LB에는 local +200 이상이 필요하다.
현재 `<+5` deployable proxy와 두 order 가까이 차이 난다. **현재 information set에서
1100은 현실적이라고 볼 근거가 없다.**

## G. Proxy-to-LB evidence

| Submission | Local expectation | Actual LB delta | Interpretation |
| --- | ---: | ---: | --- |
| 19 full-state | 약 +8 | -4.0091 | coverage extrapolation 실패 |
| 20 stage `.5` | 약 +1.99 | -3.4924 | 작은 proxy 역전 |
| 21 tree reblend | +31.89 | **+6.5207** | 유일한 큰 구조 성공; 약 20.4% transfer |
| 22 two-strike | rolling mean +11.37 | **-1.8100** | multi-seed/2SE/row-independent gate도 hidden transfer 보장 못 함 |

표본이 작고 proxy 정의가 서로 달라 regression은 하지 않았다. 관측 가능한 결론은 작은
local gain이 actual LB 양수를 보장하지 않고, 큰 구조 변화도 1:1 전이하지 않는다는
것이다. 따라서 +4 수준 local headroom을 submission 근거로 해석할 수 없다.

## H. Irreducible-looking error

보수적 proxy는 quality-model disagreement bottom 20%이면서 frozen coarse-state target
entropy가 `.95 bit` 이상인 행이다.

- 2022: rows `19.42%`, total Brier `19.69%`
- 2024: rows `19.63%`, total Brier `19.60%`
- 전체 coarse-state entropy mean: 2022 `.9912`, 2024 `.9937 bit`

Coarse state는 `game_type, count, pitcher/batter hand, base, outs`를 cutoff 이전 공식
train에서만 집계하고 alpha 100으로 shrink했다. 이 값은 feature나 prediction에 쓰지 않은
ceiling diagnostic이다. 동일 observable coarse state에서도 binary outcome uncertainty가
매우 높고, 강한 모델들이 거의 같은 p를 낸다.

이는 정확한 Bayes error가 아니다. 다만 **IRREDUCIBLE-LOOKING ERROR SHARE**의 보수적
proxy로 약 20% rows/total loss를 제시할 수 있다. 나머지가 전부 recoverable하다는 뜻도
아니다. 오히려 per-row oracle과 cross-fit 사이의 거대한 간극이 많은 row를 사전 식별할
수 없음을 보여준다.

2023 F는 별도 **NON-FORECASTABLE TEMPORAL SHOCK**다. 10.46% rows가 전체 Brier의
11.88%를 차지했고 F target/pred가 `.4729/.6688`이었다. 이 구간의 candidate oracle
gain은 2025 headroom에 포함하지 않았다.

## I. Final decision

**NO MATERIAL RECOVERABLE HEADROOM — STOP MODEL SEARCH**

Theoretical oracle은 크지만 deployable하지 않다. 2022·2024 공통, current-row legal,
동일 correction, 독립 season transfer 조건을 만족하는 local headroom은 +10에 미달하고
two-strike 한 축에만 의존한다. 그 축은 이미 hidden LB에서 음수였다. 새로운 error-targeted
method를 설계할 high-margin 오류 유형이 남아 있지 않다.

## J. Next action

**STOP ACTIVE MODEL SEARCH — KEEP 21ST CHAMPION**

- two-strike 및 파생 legal routing 재개 금지
- top-error oracle, per-row oracle을 selector 학습 근거로 사용 금지
- 후보 union/stacking/MoE 재개 금지
- 새 model/ZIP 생성 금지
- 새 공식 정보, 규칙 변경, 구현 버그 또는 독립적인 +10 구조 증거가 생길 때만 재개

## 산출물

- `artifacts/headroom_audit/champion_error_summary.csv`
- `artifacts/headroom_audit/candidate_recovery_matrix.csv`
- `artifacts/headroom_audit/candidate_recovery_overlap.csv`
- `artifacts/headroom_audit/model_disagreement.csv`
- `artifacts/headroom_audit/stable_recoverable_slices.csv`
- `artifacts/headroom_audit/oracle_bounds.csv`
- `artifacts/headroom_audit/proxy_lb_transfer.csv`
- `artifacts/headroom_audit/headroom_decomposition.csv`
- `artifacts/headroom_audit/bss_targets.csv`
- `artifacts/headroom_audit/manifest.json`

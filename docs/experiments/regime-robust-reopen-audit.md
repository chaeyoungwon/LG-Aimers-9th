# Regime-robust learning 재개 사전 감사

실행일: 2026-08-25

## 결론

**REGIME-ROBUST OBJECTIVE REDUNDANT — DO NOT TRAIN**

21차 champion을 변경하지 않았다.

- champion: CatBoost 30% + LightGBM team 70%
- LB: `1058.074429882`
- ZIP: `artifacts/sub_tree_reblend_w0p300.zip`
- 재확인 SHA-256: `8ce93a197efea18cf89b7b988c59a8b369d683983152b24b74976a96885da7cb`

이번 감사에서는 모델을 학습하거나 ZIP을 생성하지 않았다. 핵심 후보인 season
GroupDRO는 선행 저장소에서 이미 동일한 legal 52-feature MLP, cutoff-safe temporal OOF,
eta grid, 두 seed, champion blend 및 subgroup safety까지 검증되어 **KILL**됐다.
Leave-one-regime-out은 현재 rolling pseudo-future 및 worst-fold 선택 정책과 중복되고,
season-loss variance penalty는 이미 측정한 mean/worst/std frontier의 smooth
재매개변수화다. Stable/invariant feature도 실제 shape-stability 분류와 stable-only
학습까지 끝났으며 3/3 champion 개선이 없었다.

무엇보다 2022·2024에 공통으로 남은 큰 오류 geometry가 없다. 기존 GroupDRO의
champion 0.5% blend 효과는 `+0.43/-1.47/+0.99 BSS`, stable additive는
`-0.76/+2.90/+1.07 BSS`였다. 2024 `+10 BSS` 사전 근거와 한 order 차이다.

## A. Existing objective inventory

### hard-example

Champion inner-OOF error q50/q75/q90에 따라 LightGBM 학습행을 가중했다. 이는 row
difficulty objective다. Hard 후보는 calibration이 붕괴했고 5% champion blend가
2022/2023/2024 `-39.12/+173.58/-10.44 BSS`로 2023 충격에만 맞았다.

### uncertainty

Champion prediction uncertainty로 학습행을 연속 가중했다. 대표 lambda `.5`의 5%
blend는 `-0.20/-4.24/+0.35 BSS`, residual correlation은 `.999895` 이상이었다.
Hard/uncertainty와 season robustness는 개념적으로 다르지만 둘 다 이미 CLOSED다.

### existing robust validation

- rolling pseudo-future: `<=2019 -> 2020`, ..., `<=2023 -> 2024`
- 2022/2023/2024 fold median, worst fold, paired 2SE 및 R/F safety gate
- 2023-only gain으로 평균을 구제하지 않는 high-margin policy
- temporal recency/window/decay, direct-Brier, season balancing, season GroupDRO
- feature effect shape의 fold 간 안정성 분류와 stable-only additive model

### 이번 후보와 중복도

| 후보 | 판정 | 근거 |
| --- | --- | --- |
| A. Season GroupDRO | **REDUNDANT** | 동일 objective가 eta `.01/.05/.10`, seed 42/43, temporal OOF 및 champion blend까지 실행됨 |
| B. Leave-One-Regime-Out robust risk | **REDUNDANT** | 기존 rolling pseudo-future의 worst/median 선택 정책과 동일한 model-selection 원리; validation target은 학습 objective에 쓰지 않음 |
| C. mean season loss + variance | **REDUNDANT** | 기존 GroupDRO/season-balanced 결과에서 mean, worst, season std를 함께 최적 선택했고 frontier가 이미 관측됨 |
| D. invariant/stable feature learning | **REDUNDANT** | 52 feature의 binned effect 안정성 분류 및 stable-only/full additive 학습이 완료됨 |

C의 정확한 수식 자체를 별도 optimizer로 실행하지는 않았지만 새로운 데이터 축이나
prediction family가 아니다. GroupDRO가 worst loss와 std를 실제로 낮추고도 current
champion residual에 전이되지 않았으므로 lambda sweep은 닫힌 objective를 다시 튜닝하는
행위다.

## B. Temporal failure summary

OOF 배열은 validation season의 공식 train 행 수/순서와 길이가 일치함을 확인했다.
각 prediction은 `0.3 * Cat + 0.7 * team LightGBM`로 재구성했다. 2020/2021 manifest는
현재 파일이 2020 cell만 기록한 stale summary라서 provenance의 완전한 manifest로
간주하지 않았고, 배열 정렬과 recipe는 실제 행 수와 builder contract로 확인했다.

False-confidence는 사전 고정 진단 정의
`(p>=.6 and y=0) or (p<=.4 and y=1)`의 행 비율이다. ECE는 고정 10-bin이다.

| Season | Rows | Brier | Logloss | ECE | Pred mean | Target mean | Residual var | False-conf. | Main failure | Regime-specific? |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2020 | 244,087 | .248235 | .689787 | .023666 | .553945 | .532712 | .247785 | 9.40% | 전체 과예측, F 과예측 | yes |
| 2021 | 247,088 | .245729 | .684508 | .017424 | .515338 | .532762 | .245425 | 2.12% | 전체 저예측, F 저예측 | yes |
| 2022 | 247,472 | .243132 | .679060 | .004860 | .525074 | .528920 | .243117 | 3.98% | R 및 고표본군의 약한 잔여 loss | no large failure |
| 2023 | 245,525 | .252347 | .698317 | .029381 | .510607 | .499957 | .252234 | 6.22% | F target 급락 미적응 | **yes** |
| 2024 | 253,507 | .247655 | .688438 | .004091 | .489717 | .486105 | .247642 | 1.08% | 약한 count/hand 차이 | no large failure |

R/F가 시간 변화의 핵심이다.

| Season | F rows/share | F target | F pred | F Brier | R Brier |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2020 | 23,213 / 9.51% | .587774 | .670952 | .248944 | .248161 |
| 2021 | 25,861 / 10.47% | .703840 | .595778 | .220020 | .248734 |
| 2022 | 30,448 / 12.30% | .708749 | .703170 | .206417 | .248283 |
| 2023 | 25,686 / 10.46% | .472904 | .668803 | .286462 | .248361 |
| 2024 | 30,010 / 11.84% | .459280 | .465484 | .246893 | .247757 |

2023 F는 명확한 shock이지만 2024 모델은 2023 history를 학습해 이미 적응했다. 따라서
2023 F를 더 맞히는 objective는 미래 강건성 근거가 아니다.

### 2022/2024 사전 정의 subgroup geometry

각 cell의 괄호는 season 전체 Brier 대비 excess다. History는 current-row
`asof_*_n`, matchup은 validation season 이전 공식 train에 같은 pitcher-batter pair가
있었던 횟수만 사용했다. 이는 진단용이며 새 feature나 model input으로 쓰지 않았다.

| Axis | 2022 worst eligible cell | 2024 worst eligible cell | 반복 방향? |
| --- | --- | --- | --- |
| R/F | R, 87.70%, `.248283` (`+.005151`) | R, 88.16%, `.247757` (`+.000102`) | 명목상 동일, 2024 효과 소멸 |
| count family | strikes>balls, 28.49%, `.244497` (`+.001365`) | strikes>balls, 28.38%, `.248391` (`+.000737`) | 약하게 동일 |
| pitcher history | 200+, 94.14%, `.243710` (`+.000578`) | 200+, 94.26%, `.247716` (`+.000061`) | 효과 소멸 |
| batter history | 200+, 94.39%, `.244244` (`+.001112`) | 200+, 95.39%, `.247712` (`+.000057`) | 효과 소멸 |
| matchup history | 20+, 24.70%, `.247580` (`+.004448`) | 1-19, 25.30%, `.247910` (`+.000255`) | 불일치 |
| handedness | 1-1, 12.41%, `.243651` (`+.000519`) | 2-1, 35.21%, `.248246` (`+.000591`) | 불일치 |
| inning | 1-3, 33.68%, `.243532` (`+.000400`) | 7+, 32.82%, `.247719` (`+.000064`) | 불일치 |

모든 큰 group은 수만 행이라 표본 크기는 충분하다. 문제는 작은 group variance가 아니라
최악 cell의 정체와 효과 크기가 시즌마다 바뀐다는 점이다. Season x F group도 최소
23,213행으로 학습 가능하지만 target이 `.703840 -> .708749 -> .472904 -> .459280`으로
변하므로 DRO가 이를 하나의 invariant relation으로 만들 수 없다. 향후 실험이 있었다면
사전 minimum group size는 `10,000` rows로 둘 수 있으나, 이번 결론을 바꾸지 않는다.

## C. Robust frontier

아래는 current 21차 champion 대비 Delta BSS다. 서로 다른 문서의 candidate를
같은 신규 scan으로 재선택하지 않고 각 실험의 사전 선택 대표값을 옮겼다.

| Candidate | 2022 | 2023 | 2024 | Robust interpretation |
| --- | ---: | ---: | ---: | --- |
| Champion | 0 | 0 | 0 | frozen endpoint |
| Hard 5% | -39.12 | +173.58 | -10.44 | 2023-only, calibration failure |
| Uncertainty `.5`, 5% | -0.20 | -4.24 | +0.35 | near-identical residual, no frontier gain |
| Hierarchical EB, 5% | -2.81 | +97.92 | -12.41 | mean reversion toward 2023 only |
| LUPI auxiliary, 5% | +0.88 | +25.13 | -2.50 | 2024 negative |
| Role-state, 5% | +0.96 | -13.59 | -0.34 | no robust gain |
| Direct-Brier, 1% | +0.97 | -2.75 | +1.93 | 2023/subset failure, tiny effect |
| Season GroupDRO, 0.5% | +0.43 | -1.47 | +0.99 | 2022/24 positive but below margin; pooled negative |
| Stable additive, 0.25% | -0.76 | +2.90 | +1.07 | 2022 negative |
| Two-strike submitted | +5.75 | +24.02 | +4.34 | old local gate pass, actual LB `-1.81`; CLOSED |

GroupDRO standalone은 동일 MLP ERM 대비 worst Brier를 `.000255970` 낮추고 season
Brier std를 `.00464107 -> .00448453`로 줄였다. 그러나 2022는 `.000126037`
악화했고, learned q range는 모든 fold에서 `.001` 미만으로 거의 uniform이었다.
Champion blend의 residual correlation은 `.997803`; 2023 F gain은
`-.000078121`, safety floor `-1e-5`를 크게 위반했다.

즉 평균-worst risk Pareto movement는 MLP 내부에는 있었지만 champion에 추가할 수 있는
signal이 아니었다. Champion은 관측된 robust frontier의 실용 끝점에 가깝다.

## D. Candidate objectives

REDUNDANT는 즉시 종료하므로 실제 후보를 남기지 않았다.

| Rank | Objective | Novelty | Temporal evidence | +10 plausibility |
| ---: | --- | --- | --- | --- |
| - | season GroupDRO | REDUNDANT | exact prior KILL; `+,-,+` tiny blend | 없음 |
| - | leave-one-regime-out worst-risk selection | REDUNDANT | rolling folds/worst gate가 이미 운영 정책 | 없음 |
| - | mean loss + lambda season variance | REDUNDANT | GroupDRO가 std를 낮춰도 transfer 실패 | 없음 |
| - | stable/invariant feature restriction | REDUNDANT | stable-only/full additive 3/3 positive 없음 | 없음 |

Deep IRM, architecture search, lambda/eta 재탐색, season x count/player 사후 group은 하지
않았다. 특히 2023 F를 본 뒤 F 전용 objective를 만드는 것은 금지했다.

## E. Stable-vs-unstable signal

선행 additive-shape audit은 univariate binned target relation의 Pearson/Spearman,
weighted sign agreement, maximum drift를 사용했다. 단순 tree importance가 아니다.

- outer-fold comparison: `STABLE 44 / UNSTABLE 2 / WEAK 6`
- validation 2022: `40 / 5 / 7`
- validation 2023: `44 / 2 / 6`
- validation 2024: `43 / 3 / 6`
- 여러 fold에서 반복 불안정: `hand_combo`, `base_state`, `batter_team_id`,
  `game_dayofweek`; 일부 fold의 `asof_pitcher_n`, pitchmix n, strike rate
- 누적/최근 pitcher rate, batter rate, count, score, win expectancy 등 핵심 정보는
  대체로 STABLE

Stable signal은 충분히 많지만 champion도 이미 이를 사용한다. Stable-only additive는
full additive보다 standalone Brier가 `.000038030` 나빴고, 가장 가까운 champion blend도
`-1.895e-6/+7.242e-6/+2.678e-6` Brier gain으로 2022가 음수였다. 따라서
"stable relation이 존재"와 "stable restriction이 champion residual을 줄임"은 다르다.

## F. 2022/2024 common direction

공통 방향의 약한 증거는 strikes>balls와 broad R/high-history population뿐이다. 하지만
2024 excess loss는 R `.000102`, high pitcher history `.000061`, high batter history
`.000057`에 불과하다. Hand, inning, matchup의 worst cell은 바뀐다.

Gradient-conflict 저비용 proxy도 낙관적이지 않다.

- season-balanced vs same MLP ERM: 2022 `-.000157`, 2024 `+.000036`
- GroupDRO eta `.05`: 2022 `-.000126`, 2024 `+.000035`
- direct Brier vs BCE seed 42: 2022 negative, 2024 positive; seed 43에서 방향 반전
- hard/EB/LUPI는 2023 개선과 2022/2024 악화가 결합

이는 하나의 robust gradient가 2022와 2024를 크게 함께 개선하기보다 평균화/수축으로
끝날 가능성이 높음을 보인다. GroupDRO가 champion blend에서 둘을 함께 개선한 유일한
근접 사례도 `+0.43/+0.99 BSS`로 효과가 작다.

## G. Upper-bound analysis

2024 prevalence baseline은 약 `.249807`이다. 전체 `+10 BSS`에는 Brier
`0.000024981` 감소가 필요하다. 한 subgroup만 개선한다고 가정하면 필요한 subgroup
Brier 감소는 다음과 같다.

| Affected coverage | Example | Required subgroup Brier gain |
| ---: | --- | ---: |
| 100% | season-wide objective | `.00002498` |
| 88.16% | R | `.00002833` |
| 28.38% | strikes>balls | `.00008801` |
| 11.84% | F | `.00021102` |
| 5% | small robust group | `.00049961` |

R/count cell의 required gain 자체는 수학적으로 가능하지만, robust objective가 실제
champion에 준 2024 전체 gain은 `.000002469` (`+0.99 BSS`)뿐이다. +10에는 약
`10.1x`가 필요하다. eta/weight를 키우면 기존 table에서 2023 loss가 단조 증가했고
subset safety도 이미 실패했다. Stable additive도 2024 `+1.07 BSS` 수준이다.
따라서 필요한 효과는 기존 관측치보다 한 order 크며 **unrealistic**이다.

## H. Rule safety

Objective 자체는 다음 계약을 지키면 규칙상 안전하다.

- training fold 내부 historical season label/loss만 group weight에 사용
- outer validation target은 q, preprocessing, early stopping에 미사용
- inference는 current row + frozen train-derived preprocessing/model만 사용
- test group statistics, frequency, prediction distribution, regime detection 미사용

선행 GroupDRO 구현도 이 계약을 지켰다. 따라서 future leakage는 없고 test-group
dependency도 없으며 row independence를 유지할 수 있다. 이번에는 candidate를 만들지
않았으므로 신규 runtime dynamic test 대상은 없다. Frozen champion의 기존 row
independence audit은 full/shuffle/reverse/subset/single/mutation 모두 max diff `0.0`이다.

## I. Final decision

**REGIME-ROBUST OBJECTIVE REDUNDANT — DO NOT TRAIN**

일반론상 robust learning은 hard-example weighting과 다르지만, 이 프로젝트에서는 바로
그 season GroupDRO와 stable-feature restriction을 이미 실행했다. 결과는 variance 감소가
champion complementarity나 2022/2024 high-margin gain으로 이어지지 않는다는 직접
반증이다. Leave-one-regime-out과 variance penalty는 남은 신규 정보축이 아니라 기존
validation/objective family의 재표현이다.

## J. Next action

**DO NOT TRAIN**

- season GroupDRO, season x R/F, season x count, eta/lambda sweep 재개 금지
- leave-one-regime-out을 새 candidate 이름으로 재실행 금지
- deep IRM 또는 stable-feature architecture 탐색 금지
- production model/ZIP 생성 금지
- 21차 champion 유지


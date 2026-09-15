# 2023 Regime Shift 진단 및 2025 전이 가능성

실행일: 2026-08-22. 공식 train 2020~2024와 cutoff-safe Cat30/team70 rolling OOF만
사용했다. 2025 test는 읽지 않았고 모델·ZIP을 만들지 않았다.

## A. Season summary

| Metric | 2022 | 2023 | 2024 |
|---|---:|---:|---:|
| rows | 247,472 | 245,525 | 253,507 |
| target mean | .528920 | .499957 | .486105 |
| prediction mean | .525074 | .510607 | .489717 |
| prediction std | .077901 | .068266 | .044663 |
| Brier | .243132 | **.252347** | .247655 |
| logloss | .679060 | **.698317** | .688438 |
| ECE | .004860 | **.029381** | .004091 |
| new pitcher row share | 15.74% | 13.80% | 19.86% |
| pitcher low-history share | 3.07% | 2.43% | 3.12% |
| batter low-history share | 3.07% | 2.47% | 2.65% |
| unseen matchup share | 50.86% | 45.90% | 48.84% |

Target mean은 2022→2024의 하락 추세다. 2023의 turnover/cold-start는 오히려 가장
낮다. 특이점은 target variance가 아니라 champion calibration이다. 고정 0.1 bin
전체는 `artifacts/regime_shift/calibration_by_season.csv`에 있다.

## B. 가장 큰 distribution shift Top 10

수치는 numeric KS다. 2023 특이성은 `min(22→23,23→24)-0.5×(22→24)`로 단순
랭킹했으며 모델 선택에는 쓰지 않았다.

| Rank | Feature | 22→23 | 23→24 | 2023 특이성 | 방향 |
|---:|---|---:|---:|---:|---:|
| 1 | asof_pitcher_fastball_rate | .0612 | .0513 | .0354 | -.0101 |
| 2 | asof_pitcher_strike_rate | .0552 | .0539 | .0270 | +.0020 |
| 3 | asof_pitcher_ball_rate | .0518 | .0600 | .0255 | -.0019 |
| 4 | asof_pitcher_offspeed_rate | .0456 | .0529 | .0211 | +.0042 |
| 5 | batter_id | .0978 | .0906 | .0181 | cohort 이동 |
| 6 | asof_batter_n | .1884 | .1466 | .0097 | +792.9 |
| 7 | away_win_expectancy | .0338 | .0287 | .0095 | -2.30 |
| 8 | home_win_expectancy | .0338 | .0287 | .0095 | +2.30 |
| 9 | asof_pitcher_pitchmix_n | .0901 | .0536 | .0085 | +686.1 |
| 10 | asof_pitcher_n | .0901 | .0536 | .0085 | +686.1 |

Count, outs, inning half 등 전술 marginal의 JS는 매우 작았다. 구조적 season-state에서는
`cur_p_rev`가 가장 2023-specific(KS .1025/.0720, 특이성 .0452)했다. 즉 broad
tactical mix보다 선수 cohort와 누적/현재 투구 프로필 변화가 2023을 구분한다.

## C. Champion error 차이 Top 10 설명축

`2023 error - mean(2022,2024 error)` 순이다.

| Rank | Feature/Subgroup | 2022 error | 2023 error | 2024 error |
|---:|---|---:|---:|---:|
| 1 | game_type=F | .206417 | **.286462** | .246893 |
| 2 | pitcher team 23 | 별도 CSV | 2023 excess +.0420 | 별도 CSV |
| 3 | batter history <100 | .221979 | **.272951** | .245758 |
| 4 | pitcher team 13 | .228642 | **.264603** | .246700 |
| 5 | pitcher history <100 | .236320 | **.262782** | .246958 |
| 6 | count 3-0 | - | excess +.01156 | - |
| 7 | count 3-1 | - | excess +.01018 | - |
| 8 | count 2-0 | - | excess +.00811 | - |
| 9 | count 1-1 | - | excess +.00783 | - |
| 10 | count 1-0 | - | excess +.00749 | - |

결정적인 F 분해:

| Season | share | target | champion prediction | calibration error | Brier |
|---:|---:|---:|---:|---:|---:|
| 2022 | 12.30% | **.708749** | .703170 | +.005580 | .206417 |
| 2023 | 10.46% | **.472904** | .668803 | **-.195900** | .286462 |
| 2024 | 11.84% | **.459280** | .465484 | -.006204 | .246893 |

F의 성공률이 2023에 약 23.6%p 급락했다. 2023 OOF는 2022까지의 높은 F rate만
학습했으므로 이를 예측할 수 없었고, 2024 OOF는 2023 전환을 학습해 이미 적응했다.
2023 전체 ECE 급증의 주원인이다. R target은 2022/2023 `.5037/.5031`로 안정적이었다.

## D. Candidate correction 공통점

후보 단독의 F행 Brier gain:

| Candidate | 2022 | 2023 | 2024 |
|---|---:|---:|---:|
| Hard weight | -.05770 | **+.03595** | -.05131 |
| Hierarchical EB | -.02178 | **+.02837** | -.00881 |
| LUPI auxiliary | -.00272 | **+.00634** | -.00681 |
| LUPI distillation | -.00017 | -.00649 | -.00042 |
| Role state | -.00017 | -.00577 | -.00028 |
| Uncertainty | -.00020 | -.00099 | -.00004 |

2023-only로 좋아진 세 후보의 공통 correction subgroup은 `game_type=F`, 저표본 타자,
저표본 투수다. 모두 champion의 과거 high-F 확률을 0.5 방향으로 수축하는 성격이다.
그러나 같은 subgroup에서 2022와 2024는 모두 악화한다. 특히 2024 champion이 이미
F 전환을 학습했으므로 correction의 수명이 2023 한 폴드뿐이라는 강한 증거다.

## E. Regime classifier

- 전체 legal pre-pitch predictor(ID/history/month 포함) AUC: **0.8449**
- ID, month, raw history counts 제외 AUC: **0.8112**

전체 모델 Top feature는 pitcher ID, batter history count, batter ID, pitcher history
count, month였다. 제한 모델은 batter success/middle rate, pitcher reverse/strike/
middle rate, pitch mix가 상위였다. 따라서 2023 입력 분포는 실제로 구분 가능하다.
다만 이 classifier는 양옆 시즌을 모두 본 사후 진단 모델이며, 2025가 어느 쪽인지 test
전체 분포 없이 알 수 없고 선수/누적 history는 시간 자체를 인코딩한다. AUC를 곧바로
배포 가능한 selector 근거로 해석하면 안 된다.

## F. 핵심 원인

1. **F 리그/구간의 단발성 target 레짐 전환**
   - target `.7087→.4729`, champion은 2023에 `.6688` 예측.
   - 2023 ECE와 Brier 악화의 가장 큰 축이며 2024에는 이미 적응했다.
2. **후보들의 공통 평균회귀 효과**
   - Hard/EB/LUPI auxiliary가 2023 F를 개선하지만 양옆 시즌 F를 악화.
   - 새로운 지속 신호보다 전환 첫해의 과거 prior 오류를 우연히 상쇄했다.
3. **저표본 선수가 원인이 아니라 충격 증폭 구간**
   - 2023 turnover/low-history share는 가장 낮지만 해당 행 error는 가장 높다.
   - 선수 구성량보다 F 전환과 cohort별 calibration mismatch가 중요하다.
4. **입력 분포 shift는 존재**
   - restricted AUC .811, pitch profile/성공률 state가 2023을 구분.
   - 그러나 error를 고친 F correction은 2024에 전이되지 않는다.
5. **2023 OOF는 미래 전환을 예측할 수 없는 정상적인 temporal failure**
   - leakage나 OOF 정렬 문제가 아니라 cutoff 설계가 드러낸 regime shock다.
   - 실제 2025 모델은 이미 2023·2024의 낮은 F 레짐을 학습한다.

## G. 최종 판정

**SHIFT EXISTS BUT NOT ACTIONABLE**

2023 shift와 원인은 명확하지만, 2025용 row-level mixture가 필요한 근거는 없다.
가장 중요한 F 충격은 2024 OOF에서 이미 흡수됐고, 2023 correction을 재사용하면 2024가
악화한다. test-wide 분포로 2025 regime을 선택하는 것도 규칙상 금지다.

## H. 다음 행동

**ROBUST VALIDATION POLICY ONLY**

향후 후보는 임의 시즌 가중평균 대신 다음 순서로 판정한다.

1. `min(2022, 2024 gain) > 0`을 최근 안정성 1차 gate로 둔다.
2. 세 폴드 median gain이 양수여야 한다.
3. 2023 gain은 전체 평균을 구제하는 근거로 사용하지 않는다.
4. `game_type=R/F`별 최소 gain을 함께 보고 F 한 셀의 oracle 개선을 전체 신호로 해석하지 않는다.
5. 재학습 후보는 multi-seed와 paired 2SE를 계속 요구한다.

전체 산출물은 `artifacts/regime_shift/`, 재현 코드는
`scripts/diagnose_regime_shift.py`다. 현재 champion ZIP은 변경하지 않았다.

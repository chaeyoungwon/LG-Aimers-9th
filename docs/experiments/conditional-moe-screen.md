# Cat/team conditional mixture-of-experts 사전 스크린

실행일: 2026-08-22

## 결론

**PRE-TRAINING REJECT — NO HIGH-MARGIN CANDIDATE**

사용자가 외부 신규 데이터 없이 능동 탐색을 재개하도록 지시한 뒤, 21차 champion의
고정 `CatBoost 30% + LightGBM team 70%`를 현재 행별 conditional gate로 바꾸는 구조를
첫 신규 축으로 검토했다. 21차 ZIP은 변경하지 않았다.

이 축은 global blend weight tuning과 달리 과거 cutoff-safe OOF에서 현재 행 조건별
expert 상대우위를 학습하는 mixture-of-experts다. 그러나 학습 전에 수행한 temporal
repeatability 감사에서 gate target이 재현되지 않았다. 따라서 Stage 1 모델 학습과 ZIP
생성을 시작하지 않았다.

## 데이터 계약

- 공식 train만 사용
- 저장된 Cat/team rolling OOF: 2020~2024
- 각 시즌 prediction은 해당 시즌보다 앞선 train만으로 학습
- 기준 prediction: `0.3 * Cat + 0.7 * team`
- 상대우위: `(y-team)^2 - (y-cat)^2`; 양수면 Cat 우위
- test, 외부 데이터, validation/test 집계 사용 없음

향후 gate를 만들 경우에도 frozen historical model과 현재 행 feature만 사용하므로 row
independence 구현은 가능하다. 이번 기각 이유는 규칙이 아니라 temporal signal 부재다.

## Expert-level headroom

BSS 환산은 비교용으로 `100000 / 0.2498` scale을 사용했다.

| Season | Cat gain vs 30:70 | team gain vs 30:70 | per-row oracle gain | Cat win rate | corr(Cat, team) | MAD |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020 | -226.31 | +9.60 | +3963.70 | 50.19% | .92637 | .02069 |
| 2021 | +84.59 | -95.59 | +3405.93 | 51.30% | .92865 | .01733 |
| 2022 | -26.25 | -28.16 | +2762.24 | 49.14% | .98178 | .01437 |
| 2023 | +102.52 | -86.53 | +3010.78 | 50.15% | .97660 | .01505 |
| 2024 | -78.38 | -5.18 | +2786.99 | 48.85% | .94609 | .01426 |

Oracle은 각 행의 정답을 본 뒤 expert를 선택하므로 deployable upper bound가 아니다.
중요한 것은 expert의 global 방향이 `team/Cat/mixed/Cat/team`으로 시즌마다 뒤집힌다는
점이다. 2024에서는 Cat을 30% 넣는 것 자체가 team 단독보다 약 `5.18 BSS` 나빴다.

## Current-row condition repeatability

17개 저카디널리티 조건을 독립적으로 조사했다. 각 값별 상대우위 평균을 2022와
2024에서 비교하고, 양쪽 모두 Cat 우위인 값의 전체 row coverage를 측정했다.

| Feature | 2022↔2024 group-adv corr | stable positive coverage 2022/2024 |
| --- | ---: | ---: |
| game_month | +.252 | 0% / 0% |
| day of week | -.845 | 0% / 0% |
| inning | +.024 | 0% / 0% |
| game_type | -1.000 | 0% / 0% |
| balls | -.281 | 0% / 0% |
| strikes | +.425 | 0% / 0% |
| outs | +.794 | 0% / 0% |
| number of runners | -.727 | 0% / 0% |
| base_state | +.073 | 9.95% / 10.53% |
| pitcher team | -.009 | 9.23% / 9.11% |
| batter team | -.232 | 0% / 0% |

top/bottom, runner flag 3개, pitcher hand, batter hand도 stable positive coverage가
0%였다. 더 엄격하게 2020~2024 전 시즌에서 Cat 우위가 유지되는 값의 최소 coverage는
17개 feature 모두 **0%**였다.

단순 count cell 일부에는 2022/2024 양수가 있었지만 이 신호는 이미 실제 LB에서 실패한
two-strike/count-specific expert 축과 중복되므로 되살리지 않는다.

## Gate 판정

Conditional gate가 성공하려면 과거 OOF의 expert 상대우위가 미래에도 반복되어야 한다.
실측은 다음과 반대다.

- global expert 방향이 시즌별 반전
- game type, runner, team, count의 group advantage도 비반복
- 2021·2023 Cat 우위를 학습하면 2024에서 반대 방향
- 전 시즌 stable coverage 0%
- 2024에서 baseline 30% Cat조차 team 단독보다 약함

따라서 selector classifier, soft gate, row-local weight regression, stacking network를
학습하지 않았다. 이들은 동일한 비반복 target을 더 유연하게 과적합할 뿐이며 2024
`+10 BSS`의 사전 근거가 없다.

최종 판정: **conditional Cat/team MoE CLOSED**. Champion은 계속 21차다.

# 2022·2024 공통 취약구간 Residual Slice Mining

실행일: 2026-08-22. Discovery primary는 2022·2024이며 2023은 stress test로만
사용했다. 2025 test는 읽지 않았다. Stable slice 하나를 발견해 expert까지 검증했지만
2024 paired 2SE를 넘지 못해 **NO SUBMISSION CANDIDATE**로 판정했다.

## Slice 정의와 안전성

사전 정의한 축만 사용했다: game type, 투수/타자 as-of history `0/1~20/21~100/
101~500/500+`, cutoff 이전 official train의 투수×타자 matchup count, count family,
handedness, inning phase, base/out, 기존 season-state availability. Matchup lookup은
각 outer validation 이전 train만 사용한다. 다른 test 행, test frequency, percentile,
groupby, rolling은 필요 없다.

Count family는 고정 우선순위다: `three_ball`(balls=3), `two_strike`(strikes=2 and
balls<3), `batter_ahead`, `even`, `pitcher_ahead`. 따라서 최종 slice는 정확히
0-2/1-2/2-2다.

## A. Single slice Top 10

양옆 시즌 excess가 모두 양수이고 share≥2%인 항목만 표시한다. Direction은
`target mean - prediction mean`의 부호가 2022·2024에서 같은지다.

| Rank | Slice | 2022 share | 2022 excess | 2024 share | 2024 excess | Direction |
|---:|---|---:|---:|---:|---:|---|
| 1 | count=two_strike | 24.07% | .001530 | 24.14% | .000974 | same, underpredict |
| 2 | base=empty | 53.33% | .000266 | 51.92% | .000170 | opposite |
| 3 | matchup=established | 42.32% | .003311 | 44.10% | .000112 | opposite |
| 4 | game_type=R | 87.70% | .005151 | 88.16% | .000102 | opposite |
| 5 | count=pitcher_ahead | 12.59% | .000118 | 12.39% | .000085 | same, underpredict |
| 6 | batter state available | 88.98% | .000546 | 90.70% | .000085 | opposite |
| 7 | pitcher history 500+ | 86.94% | .001160 | 87.34% | .000079 | opposite |
| 8 | batter history 500+ | 87.78% | .001819 | 90.51% | .000055 | opposite |
| 9 | pitcher state available | 84.26% | .000143 | 80.14% | .000043 | opposite |

`two_strike`만 effect size와 동일 방향이 모두 분명했다. `pitcher_ahead`는 bootstrap
불안정이었다.

## B. Interaction Top 10

| Rank | Interaction | 2022 share/excess | 2024 share/excess | Parent 개선 |
|---:|---|---:|---:|---|
| 1 | two_strike × batter history 500+ | 21.04% / .003354 | 21.86% / .001040 | No |
| 2 | two_strike × pitcher history 500+ | 21.24% / .002460 | 21.27% / .001008 | No |
| 3 | opposite hand × two_strike | 11.98% / .001455 | 12.37% / .000998 | No |
| 4 | same hand × two_strike | 12.09% / .001603 | 11.77% / .000950 | No |
| 5 | R × batter history 101~500 | 5.98% / .005481 | 3.88% / .000320 | direction fail |
| 6 | late inning × R | 27.55% / .004907 | 27.79% / .000204 | direction fail |
| 7 | R × pitcher history 500+ | 79.06% / .005187 | 79.81% / .000111 | direction fail |
| 8 | middle inning × R | 29.23% / .005304 | 29.37% / .000104 | direction fail |
| 9 | R × pitcher history 101~500 | 6.85% / .004992 | 6.77% / .000102 | direction fail |
| 10 | R × batter history 500+ | 80.31% / .005147 | 83.55% / .000100 | direction fail |

어떤 interaction도 `two_strike` 부모보다 common excess를 실질적으로(+.001) 높이지
못했다. 따라서 더 세분화하지 않았다.

## C. Stable slices

stable slice count = **1**

- slice: `count_family=two_strike` (0-2/1-2/2-2)
- 2022 excess: `.001530`, correction `+.003928`
- 2024 excess: `.000974`, correction `+.003684`
- bootstrap 500회 2022 CI: `[.000900, .002141]`
- bootstrap 500회 2024 CI: `[.000606, .001336]`
- 2023 excess: `.000417`, correction `-.008868` (stress only)

Bootstrap은 고정 slice와 전체 error를 각 시즌에서 재표집했다. Slice 정의를 bootstrap
결과로 바꾸지 않았다.

## D. 핵심 오류 유형

**DISCRIMINATION**

두 시즌 calibration bias는 약 `+.0037~+.0039`로 excess Brier보다 설명력이 작다.
단순 offset 대신 slice-specific model을 선택한 이유다.

## E. Expert 여부

**EXPERT TESTED**

과거 train 중 동일 slice 행만 사용한 단일 LightGBM(기존 70 features, 31 leaves,
236 rounds)을 seed 42/43/44로 학습했다. Slice 밖은 champion과 정확히 같고, slice
안에서만 10/25/50%를 혼합했다. 최적화 grid는 추가하지 않았다.

## F. Expert 결과

대표 25%, 세 seed expert 확률 평균 기준:

| Fold | Champion Brier | Candidate Brier | Δ BSS | paired 2SE BSS |
|---:|---:|---:|---:|---:|
| 2022 | .243132 | .243116 | **+6.50** | 4.24 |
| 2023 | .252347 | .252290 | **+22.75** | 4.31 |
| 2024 | .247655 | .247645 | **+3.78** | **4.31** |

2022는 2SE를 넘지만 2024는 넘지 못했다. 세 seed 각각의 overall, R, F gain은 모두
양수였으나 2024 seed 42 R gain은 `+0.00000004`로 사실상 0이다.

## G. Slice-specific gain

25% expert의 slice ΔBSS:

- 2022: `+26.98`
- 2023: `+95.70`
- 2024: `+15.67`
- outside slice: 모든 폴드 정확히 `0`

R/F 전체 ΔBSS도 양수였다: 2022 `R +6.80 / F +5.02`, 2023
`R +10.57 / F +127.35`, 2024 `R +1.59 / F +20.26`. 2023 F가 큰 값을 만들지만
후보 선택 근거에는 쓰지 않았다.

## H. Robust gate

- min(2022, 2024): `+3.78 BSS`
- median fold: `+6.50 BSS`
- fold mean/std/2SE: `+11.01 / 10.26 / 11.84 BSS`
- seed stability: overall/R/F 모두 양수이나 2024 R seed42는 사실상 0
- R/F stability: 평균 양수
- paired 2SE: **2024 FAIL** (`3.78 < 4.31`)

방향과 multi-seed strong gate는 좋지만 최근 시즌의 효과가 노이즈 바닥보다 작다.
목표 격차 약 42점에 비해 기대 크기도 작아 ZIP 제작/제출 위험을 정당화하지 못한다.

## I. 최종 결정

**NO SUBMISSION CANDIDATE**

Stable residual slice는 존재하지만 expert 제출 gate가 불완전하다. 새 ZIP은 만들지
않았고 현재 champion을 유지한다. 재현은 `scripts/mine_residual_slices.py`와
`scripts/validate_two_strike_expert.py`, 산출물은 `artifacts/residual_slice/`다.

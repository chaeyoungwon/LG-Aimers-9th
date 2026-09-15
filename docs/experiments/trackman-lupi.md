# TrackMan LUPI / Auxiliary Student 신규 정보축 검증

실행일: 2026-08-22

## 실험 목적과 최종 결정

공식 current-pitch TrackMan을 training-only privileged supervision으로 사용했을 때,
test에서는 legal pre-pitch 52개 피처만 받는 student가 21차 CatBoost 30% + LightGBM
team 70% champion의 잔차를 줄이는지 검증했다.

**FAIL — NO SUBMISSION CANDIDATE.** Auxiliary student는 2023에서만 큰 blend 이득을
냈고 2024는 악화했다. Distillation은 pooled 및 2023·2024 blend가 모두 악화했다.
기존 빠른 중단 조건을 충족하므로 seed 44 재학습, architecture/lambda 재탐색 및 ZIP
생성을 하지 않았다.

## A. TrackMan linkage

기존 official-data-only reciprocal pitch anchor를 재사용하고 실제 artifact와 공식 CSV를
다시 대조했다. 새 fuzzy matching은 만들지 않았다.

| 항목 | 값 |
|---|---:|
| 전체 train row | 1,475,092 |
| HIGH-mapped main row | 561,608 |
| main-key unique anchor | 377,287 |
| reciprocal one-to-one match | **374,986** |
| reciprocal 충돌로 제거된 anchor | **2,301** |
| 전체 coverage | **25.4212%** |
| 전체 train pitcher | 792 |
| matched main/TM pitcher | 109 / 109 |

연도별 anchor/전체 행 coverage는 2019 `44,711/18.83%`, 2020 `53,656/21.98%`,
2021 `71,184/28.81%`, 2022 `68,728/27.77%`, 2023 `74,057/30.16%`,
2024 `62,650/24.71%`다. TrackMan chronological rank가 strictly increasing인
adjacent pair는 99.9672%, exact rank gap은 98.6337%였다.

실제 공식 컬럼 coverage:

| 공식 컬럼 | matched non-null | coverage |
|---|---:|---:|
| `pitch_type_group` | 374,986 | 100.0000% |
| `rel_speed` | 374,977 | 99.9976% |
| `spin_rate` | 374,194 | 99.7888% |
| `induced_vert_break` | 374,976 | 99.9973% |
| `horz_break` | 374,897 | 99.9763% |
| `extension` | 374,946 | 99.9893% |
| `rel_height` | 374,977 | 99.9976% |
| `rel_side` | 374,977 | 99.9976% |
| `zone_speed` | 374,977 | 99.9976% |

## B. Auxiliary Student

구조는 legal ID-free 52 features → shared `64→32` MLP → control head + auxiliary
heads다. 모든 train row에서 main BCE를 계산하고 cutoff-safe linked row에서만 fold-train
표준화 MSE를 더했다. auxiliary predictability gate가 선택한 head는 `rel_side`와
`horz_break`, lambda는 `.05`, batch 16,384, 3 epochs다.

| Fold | 21차 Champion BSS | Candidate BSS | Δ BSS |
|---:|---:|---:|---:|
| 2022 | 2420.73 | 2047.68 | **-373.06** |
| 2023 | -938.83 | -750.99 | **+187.83** |
| 2024 | 861.55 | 345.79 | **-515.76** |
| pooled | 911.17 | 674.66 | **-236.51** |

동일 architecture control-only 대비 seed 42 auxiliary Brier gain은
`-3.6231e-5 / +2.5110e-4 / -9.2563e-5`, seed 43은
`-1.8924e-5 / +2.5217e-4 / -4.9220e-5`다. 두 seed 모두 2022·2024가 악화했다.
이 사전 등록된 kill pattern 때문에 seed 44는 실행하지 않았다.

## C. Distillation

Privileged teacher는 legal X + current-pitch 8개 물리값 + `pitch_type_group`, student는
legal X만 사용했다. teacher soft target은 inner 3-fold OOF만 사용했으며 선택 alpha는
`.5`다. Privileged teacher 자체의 legal-only teacher 대비 Brier gain은
2022/2023/2024 `+.00182984 / +.00189353 / +.00180555`로 강했다. 그러나 student로
전달되지 않았다.

| Fold | 21차 Champion BSS | Candidate BSS | Δ BSS |
|---:|---:|---:|---:|
| 2022 | 2420.73 | 2291.64 | **-129.10** |
| 2023 | -938.83 | -1303.61 | **-364.79** |
| 2024 | 861.55 | 686.02 | **-175.53** |
| pooled | 911.17 | 688.96 | **-222.21** |

## D. Residual complementarity

| Candidate | prediction correlation | residual correlation | champion error 상위 10% candidate Brier gain |
|---|---:|---:|---:|
| Auxiliary | 0.882673 | **0.998020** | +0.0231859 |
| Distillation | 0.962598 | **0.999340** | +0.0108936 |

상위 오류행에서는 candidate가 나아 보이지만 residual correlation이 거의 1이고, 이
부분집합은 정답을 본 사후 oracle subset이다. 독립적인 일반화 신호의 근거로 쓰지 않는다.

## E. 21차 Champion blend

표는 Brier를 BSS 단위로 환산한 champion 대비 Δ다.

| Candidate weight | 2022 | 2023 | 2024 | Mean Δ |
|---:|---:|---:|---:|---:|
| Auxiliary 2.5% | +0.70 | +12.77 | -0.94 | +4.17 |
| Auxiliary 5% | +0.88 | +25.13 | -2.50 | +7.84 |
| Auxiliary 10% | -0.30 | +48.59 | -7.45 | +13.62 |
| Auxiliary 15% | -3.53 | +70.41 | -14.85 | +17.34 |
| Distillation 2.5% | +0.21 | -5.15 | -0.44 | -1.79 |
| Distillation 5% | +0.25 | -10.50 | -1.09 | -3.78 |
| Distillation 10% | -0.20 | -21.81 | -2.98 | -8.33 |
| Distillation 15% | -1.36 | -33.94 | -5.69 | -13.66 |

Auxiliary 5%의 fold mean/std/2SE는 `+7.84 / 15.07 / 17.40 BSS`다. pooled paired
2σ는 `1.47 BSS`를 넘지만, 2023 하나가 전체를 끌고 가고 2024가 악화하므로 temporal
strong gate는 FAIL이다. Distillation 5%는 `-3.78 / 5.86 / 6.76 BSS`다.

## Coverage별 분석

Auxiliary 5%의 대표 결과:

| Fold | linked | unlinked | high history | low history |
|---:|---:|---:|---:|---:|
| 2022 | +0.47 | +0.97 | +2.45 | -3.33 |
| 2023 | +42.94 | +19.31 | +32.81 | +57.32 |
| 2024 | **-7.85** | -0.49 | -3.21 | **-14.49** |

Cold-start는 폴드당 63~88행뿐이라 2σ가 144~300 BSS로 판정 불가다. non-cold-start는
전체 결과와 같다. L/R matchup과 12개 count에서도 2023 양수·2024 혼재 패턴이
반복됐다. 예를 들어 2024 `3-1`은 `-21.40`, `0-2`는 `+13.72 BSS`로 방향이
안정적이지 않다.

## F. 통계 판정

- mean = `+7.84 BSS` (Auxiliary 5%, fold arithmetic mean)
- std = `15.07 BSS`
- 2σ/2SE = `17.40 BSS`
- seed stability = seed 42·43 모두 2022/2024 악화
- 판정 = **FAIL**

## G. 규칙 안전성

학습 데이터 흐름은 `official train legal X → shared trunk → control prediction`이며,
TrackMan Z는 cutoff-safe linked train row의 auxiliary/teacher loss에만 들어간다. Outer
validation과 실제 inference 입력에는 legal current-row/as-of X만 존재한다. auxiliary
scaler, vocabulary, teacher target은 fold train에서만 적합했다. validation TrackMan,
future TrackMan, test groupby/aggregation/frequency/rank/calibration은 사용하지 않았다.
따라서 candidate 자체는 행 독립이지만 성능 gate를 통과하지 못했다.

## H. 최종 결정

**NO SUBMISSION CANDIDATE**

재분석 산출물은 `artifacts/backtest/lupi_champion21/`이며,
`scripts/analyze_lupi_champion21.py`가 동결 candidate OOF를 현재 21차 기준으로 다시
계산한다. production 코드·현재 최고 ZIP은 변경하지 않았다.

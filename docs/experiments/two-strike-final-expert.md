# Two-Strike Expert 최종 제한 검증

실행일: 2026-08-22. Slice는 `strikes==2 and balls<3`로 동결했고 후보는 LGB,
CatBoost, 50:50 expert 평균, inner-OOF residual specialist 네 개만 검증했다.
오프라인 결론은 **SUBMISSION CANDIDATE**였으나 실제 LB는 `1056.264418601`로
21차 champion보다 `-1.810011281` 하락했다. 따라서 최종 판정은 **기각**이며
two-strike 탐색은 종료한다. 기존 champion ZIP은 보존했다.

## A. Main comparison

세 seed expert 예측을 평균한 뒤 slice-gated 적용한 ΔBSS다.

| Candidate | Weight | 2022 | 2023 | 2024 | 2024 paired 2SE | Gate |
|---|---:|---:|---:|---:|---:|---|
| LGB | .10 | +3.30 | +9.77 | +2.22 | 1.72 | PASS 2024 |
| LGB | .25 | +6.50 | +22.75 | +3.78 | 4.31 | FAIL |
| LGB | .50 | +7.19 | +39.89 | +1.66 | 8.61 | FAIL |
| Cat | .10 | +2.51 | +10.56 | +2.40 | 1.77 | PASS 2024 |
| Cat | .25 | +3.95 | +24.57 | +4.12 | 4.43 | FAIL |
| Cat | .50 | +.14 | +43.01 | +1.96 | 8.87 | FAIL |
| LGB+Cat 50:50 | .10 | +2.99 | +10.23 | +2.37 | 1.63 | PASS 2024 |
| **LGB+Cat 50:50** | **.25** | **+5.75** | **+24.02** | **+4.34** | **4.08** | **PASS** |
| LGB+Cat 50:50 | .50 | +5.78 | +42.91 | +3.35 | 8.17 | FAIL |
| Residual | .10 | -.30 | +.61 | -4.70 | 4.30 | FAIL |
| Residual | .25 | -3.73 | +.50 | -22.77 | 10.77 | FAIL |
| Residual | .50 | -17.39 | -2.38 | -82.31 | 21.60 | FAIL |

`.10` 후보들도 각자 2024 gain>2SE지만 최종 후보는 2024 gain이 가장 크면서 hard
gate를 통과한 사전 정의 C/.25다. 추가 weight 탐색은 하지 않았다.

## B. Seed detail — C/.25

| Seed | 2022 ΔBSS | 2023 ΔBSS | 2024 ΔBSS |
|---:|---:|---:|---:|
| 42 | +4.78 | +25.58 | +4.13 |
| 43 | +5.55 | +24.30 | +4.28 |
| 44 | +6.16 | +21.64 | +4.06 |

모든 seed에서 2022·2024가 양수다.

## C. Count detail — C/.25

진단용이며 selector는 바꾸지 않았다.

| Count | 2022 gain | 2023 gain | 2024 gain |
|---|---:|---:|---:|
| 0-2 | +13.97 | +145.49 | +26.03 |
| 1-2 | +32.40 | +87.83 | +26.71 |
| 2-2 | +21.56 | +84.66 | +1.28 |

2-2의 2024 효과가 작아도 사후 분리하지 않고 three-count slice 전체를 유지한다.

## D. Expert diversity

세 폴드 pooled slice 기준:

- corr(champion, LGB expert) = `.94724`
- corr(champion, CAT expert) = `.93549`
- corr(LGB expert, CAT expert) = `.95432`
- mean expert residual corr with champion = `.99923`

Cat은 LGB와 완전히 같지 않고, 평균으로 2024 paired gate가 `3.78<4.31`에서
`4.34>4.08`로 개선됐다. Residual specialist는 inner OOF 113,850/173,425/231,787개
two-strike 행을 사용했지만 모든 2022·2024 설정이 실패했다.

Slice 내 C/.25 prediction mean/std는 2022 `.52352/.07029`, 2023
`.51210/.06224`, 2024 `.49326/.04036`; target mean은 `.52643/.50123/.49394`다.
2024 calibration mean은 거의 그대로이며 AUC가 개선되는 discrimination 효과다.

## E. R/F — C/.25

세 seed expert 평균 후보 기준:

- 2022 R = `+6.20 BSS`
- 2022 F = `+0.46 BSS`
- 2024 R = `+1.58 BSS`
- 2024 F = `+23.32 BSS`

2022 F seed42는 Brier gain `-3.43e-6`으로 미세 음수지만 다른 seed와 평균은 양수이며
치명적 붕괴는 아니다. 2023 F 큰 gain은 gate 근거로 사용하지 않았다.

## F. Locality / leakage

- validation outside-slice max_abs_diff = `0.0`
- 실제 후보 ZIP vs champion ZIP, 3,000행 outside-slice max/mean/changed =
  `0.0 / 0.0 / 0`
- 후보 ZIP 행 독립성 single/full/shuffled/reversed/subset/unrelated mutation =
  모두 max `0.0`, mean `0.0`, changed `0`
- slice rule은 현재 행 balls/strikes만 사용
- residual target은 cutoff-safe inner OOF만 사용
- test groupby/frequency/rank/rolling/distribution/calibration 없음

## G. Robust gate

- 2022 positive = PASS (`+5.75`)
- 2024 positive = PASS (`+4.34`)
- 2024 > paired 2SE = PASS (`4.34 > 4.08`)
- all seeds 2022/2024 positive = PASS
- median fold positive = PASS
- R/F stable = PASS
- outside slice exact = PASS
- leakage/row independence = PASS

## H. 최종 결정

**REJECTED BY LEADERBOARD — TWO-STRIKE SEARCH CLOSED**

- 실제 LB: `1056.264418601`
- 21차 champion LB: `1058.074429882`
- 차이: `-1.810011281`

- ZIP: `artifacts/sub_two_strike_lgbcat_w0p250.zip`
- SHA-256: `48add68ddd754f496256483aa41057de9612592a601f2ea18e5f445a1bfeb297`
- CRC: PASS (23 files)
- isolated execution: PASS, return code 0
- row independence max_abs_diff: `0.0`

기존 `sub_tree_reblend_w0p300.zip`을 champion으로 유지한다. 제출에 사용한 후보 ZIP은
현재 로컬 artifacts에 남아 있지 않지만 builder와 당시 SHA-256으로 재현 가능하다.
재생성·재제출하지 않는다. 사전 약속대로 다른 Cat/LGB 설정, 0-2/1-2/2-2 분리,
interaction, 추가 weight는 탐색하지 않는다.

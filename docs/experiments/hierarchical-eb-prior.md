# Hierarchical Empirical-Bayes prior-only 검증

작성: 2026-08-22. 목적은 새 트리 튜닝 없이 공식 train의 과거 정답만으로 만든
계층형 EB 축이 21차 챔피언(CatBoost 30% + LightGBM team 70%)에 보완적인지
검증하는 것이다. 결론은 **기각**이며 제출 ZIP은 만들지 않았다.

## A. 검증 설계와 누수 차단

롤링 cutoff는 `season < 2022 → 2022`, `< 2023 → 2023`, `< 2024 → 2024`다.
전역, 투수, 타자, 투수×타자좌우, 타자×투수좌우, 투수×카운트,
타자×카운트, 투수×타자만 사용했다. 각 posterior는
`(successes + k * parent) / (n + k)`이며 매치업 parent는 투수와 타자
posterior의 평균이다. 최종 prior는 전역값과 7개 posterior를
`n/(n+k)`로 고정 가중 평균한다. validation label로 계수나 가중치를 맞추지 않았다.

`transform()`에는 target을 전달하지 않는다. lookup은 cutoff 이전 공식 train에서만
생성한다. 셔플, 역순, 부분집합, 한 행 삭제, 무관 행 중복에서 기존 행의 모든
posterior 최대 차이는 모두 **0.0**이었다. 따라서 다른 평가 행의 값·존재·순서를
사용하지 않는 행 독립 구조다.

## B. 커버리지

| 폴드 | 투수 | 타자 | 투수×카운트 | 타자×카운트 | 투수×타자 |
|---|---:|---:|---:|---:|---:|
| 2022 | 84.26% | 88.98% | 84.00% | 88.79% | 49.14% |
| 2023 | 86.20% | 90.60% | 86.12% | 90.36% | 54.10% |
| 2024 | 80.14% | 90.70% | 80.08% | 90.57% | 51.16% |

매치업의 관측 행 중앙 표본은 모든 폴드에서 20이었다. 상세 그룹 수와 중앙값은
`artifacts/hierarchical_eb/coverage.csv`에 있다.

## C. prior-only 결과

`k = 20, 100, 500` 중 pooled standalone Brier가 가장 낮은 500을 대표값으로
선택했다. gain은 `champion Brier - candidate Brier`라 양수가 개선이다.

| 폴드 | 챔피언 | EB prior | gain | corr(EB, champion) | corr(gap, residual) |
|---|---:|---:|---:|---:|---:|
| 2022 | .243132 | .247779 | **-.004647** | .6125 | -.0002 |
| 2023 | .252347 | .251119 | **+.001228** | .6360 | +.0986 |
| 2024 | .247655 | .251321 | **-.003666** | .3583 | -.0044 |
| pooled | .247699 | .250080 | **-.002382** | .5793 | +.0305 |

2022와 2024가 크게 악화하므로 prior-only 단독 모델은 실패다.

## D. 고정 소량 블렌드

| weight | 2022 | 2023 | 2024 | pooled |
|---:|---:|---:|---:|---:|
| 2.5% | -0.00000054 | +0.00012481 | -0.00001349 | +0.00003629 |
| 5% | -0.00000701 | +0.00024479 | -0.00003099 | +0.00006766 |
| 10% | -0.00003774 | +0.00047027 | -0.00007802 | +0.00011567 |
| 15% | -0.00009218 | +0.00067644 | -0.00014108 | +0.00014401 |

pooled만 보면 양수지만 모든 가중치가 2022·2024에서 악화한다. 세 폴드 일관성
규칙을 통과한 가중치는 없다.

## E. 오류 상위 및 희소 그룹

챔피언 오류 상위 5/10/20%에서는 EB가 당연히 평균 쪽으로 돌아오며 크게 이긴다.
이는 정답을 본 사후 oracle 구간이라 배포 가능한 gating 근거가 아니다. 사전 식별
가능한 희소 그룹에서는 반대였다. 저표본(1~100) 투수 gain은 2022 `-0.00929`,
2024 `-0.01023`; 저표본 타자는 각각 `-0.01048`, `-0.00804`였다. 매치업 표본이
5개보다 많은 구간도 2022 `-0.00260`, 2024 `-0.00257`이었다. 좌우·카운트별
결과도 2024에서 후보를 구제하는 안정 구간이 없었다. 전체 표는
`artifacts/hierarchical_eb/error_segments.csv`에 있다.

## F. Logistic combiner 판정

실행하지 않았다. 요청된 중단 조건인 “2022/2024 악화”와 “고정 소량 블렌드 전부
일관성 실패”가 동시에 성립했다. 이 상태에서 combiner를 학습하면 2023 레짐에
과적합할 위험만 커진다.

## G. 최종 판정

**기각.** 현재 21차 `artifacts/sub_tree_reblend_w0p300.zip`을 유지한다.
Hierarchical EB, 희소 그룹 gating, logistic stacking 어느 것도 제출 후보로 만들지
않는다. 기존 ZIP은 변경하지 않았다.

## H. 재현

```bash
DYLD_LIBRARY_PATH=/Users/wooh/Documents/dev/LG-Aimers-9th/.venv/lib \
  /Users/wooh/Documents/dev/LG-Aimers-9th/.venv/bin/python \
  scripts/validate_hierarchical_eb.py
DYLD_LIBRARY_PATH=/Users/wooh/Documents/dev/LG-Aimers-9th/.venv/lib \
  /Users/wooh/Documents/dev/LG-Aimers-9th/.venv/bin/python -m pytest \
  tests/test_hierarchical_eb.py -q
```

산출물은 `artifacts/hierarchical_eb/`에 있다. 검사 당시 챔피언 ZIP SHA-256은
`8ce93a197efea18cf89b7b988c59a8b369d683983152b24b74976a96885da7cb`이다.

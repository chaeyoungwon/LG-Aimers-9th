# Pitcher-form asymmetric shape validation

## 결론

**B. residual signal exists but cannot be safely exploited**

Nested fold에서 `beta_neg < 0`, `beta_pos > 0`의 부호는 반복됐고 두 cap 모두 2023·2024·pooled Brier가 양수였다. 그러나 최신 fold의 전체 개선은 매우 작고, 두 후보 모두 2024 full-count에서 큰 손실을 냈다. primary cap `0.005`는 2024 negative-form 구간도 안전성 임계값을 넘겨 악화됐다. 따라서 production-equivalent package를 만들지 않고 form 축을 종료한다.

## Hypothesis

기존 linear pitcher-form correction 이후에도 positive/negative form에 서로 다른 residual slope가 남는지 검증했다. Champion endpoint나 기존 form alpha는 고정하고 다음 두 basis만 사용했다.

```text
f_pos = max(f, 0)
f_neg = min(f, 0)
delta = beta_pos * f_pos + beta_neg * f_neg
p_new = clip(champion_final + clip(delta, -cap, cap), 0, 1)
```

Intercept, champion 재학습, alpha 재튜닝, batter/total form, count, unseen flag 및 그 밖의 residual feature는 사용하지 않았다.

## Validation design

- 2022: 이전 OOF가 없어 `beta_neg = beta_pos = 0`인 identity로 평가
- 2023 validation: 2022 OOF residual만으로 beta fitting
- 2024 validation: 2022+2023 OOF residual만으로 beta fitting
- 목적함수: mean squared residual + `lambda * ||beta||²`
- Ridge `lambda = 1e-4`: validation label이나 Brier를 보지 않고 training basis energy만으로 고정했다. 2022의 negative/positive basis diagonal은 각각 `9.1933e-5`, `3.5952e-5`였으며 lambda는 두 값 이상인 strong shrinkage다.
- 평가 reference: 모든 경우 `champion_final`
- 후보 cap: primary `0.005`, 비교 후보 `0.010`만 평가

## Coefficient stability

| Validation | Fit period | beta_neg | beta_pos | Signs |
| --- | --- | ---: | ---: | --- |
| 2022 | identity | 0 | 0 | neutral |
| 2023 | 2022 | -0.0221457 | +0.1463723 | - / + |
| 2024 | 2022+2023 | -0.1318728 | +0.4516530 | - / + |

부호는 nested folds에서 일치한다. 다만 2024 fit의 절댓값은 positive 쪽 약 3.1배, negative 쪽 약 6.0배로 커져 계수 크기 자체는 안정적이라고 보기 어렵다. `f_neg`가 음수이므로 negative beta도 양의 delta를 만든다. 즉 학습된 두 slope 모두 확률을 올리며, positive form은 증폭하고 negative form은 기존 하향 보정을 일부 되돌리는 형태다.

## Brier

`gain = Brier(champion_final) - Brier(candidate)`이며 양수가 개선이다. Pooled는 채택 판단이 가능한 2023+2024의 row-weighted 평균이고 2022 identity는 제외했다.

| Candidate | 2022 | 2023 | 2024 | Pooled | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| ridge1e-4, cap 0.005 | 0 | +3.25017e-5 | +9.65431e-7 | +1.64813e-5 | reject |
| ridge1e-4, cap 0.010 | 0 | +3.29837e-5 | +2.10053e-6 | +1.72951e-5 | reject |

시즌별·pooled gain과 beta 부호 조건은 통과했지만, 최신 시즌 개선은 사실상 0에 가깝고 아래 주요 subset 안전성 조건을 통과하지 못했다.

## Subset safety

다음은 primary cap `0.005`의 subset Brier gain이다.

| Subset | 2023 | 2024 |
| --- | ---: | ---: |
| game R | +3.62992e-5 | +1.09506e-6 |
| game F | 0 | 0 |
| pitcher seen | +3.87645e-5 | +1.79482e-6 |
| pitcher unseen | +7.73738e-7 | -1.84937e-6 |
| full count | +1.79558e-5 | **-4.71133e-5** |
| non-full count | +3.32423e-5 | +3.30656e-6 |
| negative form | +4.77717e-6 | **-1.13056e-5** |
| near-zero form | +1.38285e-7 | -5.39125e-8 |
| positive form | +9.49526e-5 | +1.21518e-5 |

주요 subset은 최소 5,000 rows, 큰 degradation은 gain `< -1e-5`로 사전에 고정했다. Primary 후보는 2024 full-count 11,771 rows와 negative-form 73,862 rows에서 실패했다. 비교 cap `0.010`도 2024 full-count에서 `-4.62034e-5`로 실패했다. Full-count와 unseen 여부는 correction feature로 쓰지 않았고 진단에만 사용했다.

Extreme-form 구간이 전체 gross positive gain에서 차지한 비율은 primary 후보 기준 2023 `21.71%`, 2024 `8.29%`였다. 개선이 극소수 extreme row에만 집중된 것은 아니지만, 이는 subset degradation을 상쇄하지 못한다.

## Calibration and correction size

| Cap / season | mean(delta) | median | p05 | p95 | Pred. mean before | after | Cap rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.005 / 2023 | +0.0005572 | +0.0001940 | 0 | +0.0024511 | 0.5104872 | 0.5110444 | 0.49% |
| 0.005 / 2024 | +0.0015811 | +0.0007859 | 0 | +0.0050000 | 0.4893748 | 0.4909559 | 13.39% |
| 0.010 / 2023 | +0.0005600 | +0.0001940 | 0 | +0.0024511 | 0.5104872 | 0.5110472 | 0.00% |
| 0.010 / 2024 | +0.0018789 | +0.0007859 | 0 | +0.0075539 | 0.4893748 | 0.4912537 | 1.82% |

Primary 후보의 mean delta는 negative/positive form에서 각각 2023 `+0.0002310`/`+0.0014397`, 2024 `+0.0012227`/`+0.0034536`였다. Intercept는 없지만 두 beta의 부호 조합 때문에 correction이 사실상 전역적인 upward shift로 작동했고, 2024에서 그 크기가 더 커졌다.

## Safety checks

- Nested fit: 각 validation season보다 앞선 OOF만 beta fitting에 사용했다.
- Validation label leakage: validation label을 뒤집어도 해당 fold beta가 bit-exact하게 유지되는 테스트를 추가했다.
- Champion identity: `beta_neg = beta_pos = 0`에서 candidate와 `champion_final`이 `atol=0`으로 완전히 동일하다.
- Row independence: single, full batch, subset, shuffle, reverse 입력에서 동일 row 예측이 `atol=0`으로 일치한다.
- Test isolation: test rows를 읽지 않았다.
- Champion integrity: 실험 전후 ZIP SHA-256은 모두 `ce10c428c9b6bd88ebf713fcc05b91ce22fc555573f3d11f8deb283637484c47`이다.
- Production unchanged: production, submission, champion ZIP을 수정하지 않았고 새 package나 LB 제출을 만들지 않았다.

## Saturation decision

Asymmetric two-slope 후보가 모든 안전성 gate를 통과한 경우에만 saturation basis를 검토하기로 했다. 통과 후보가 없으므로 세 번째 parameter는 추가하거나 평가하지 않았다.

## Reproduction

```bash
.venv/bin/python scripts/validate_pitcher_form_shape.py
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

정량 결과는 `artifacts/pitcher_form_shape/`의 `coefficient_stability.csv`, `fold_results.csv`, `subset_results.csv`, `correction_distribution.csv`, `summary.json`에 저장된다.

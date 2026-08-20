# Temporal OOF reconstruction audit

작성일: 2026-08-20

## 결론

현재 champion의 5개 member 중 `team` LightGBM stack과 `team_nn`은 이 저장소의
학습 코드와 최종 모델이 바이트 단위로 대응한다. `hgb`, `cat`, `nn`은 최종 모델과
추론 메타는 남아 있지만 원 학습 저장소의 entrypoint가 유실됐다. 직렬화된 모델과
메타에서 피처·전체 파라미터·seed를 복구할 수 있으므로 같은 recipe로 재학습은
가능하지만 source-exact라고 부르지는 않는다.

공식 `trackman_history.csv`는 로컬에 있으나 최신 champion member의 `groups`는
`matchup/count/rates` 또는 `team`뿐이다. `tm`/`platoon` lookup을 쓰는 member가
없으므로 이번 OOF 재생성에는 TrackMan이 필요하지 않다.

## Member inventory

| member | training entrypoint | feature builder | model hyperparameters | calibration | lookup | external artifact | 재학습 판정 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `hgb` | 원본 유실; `scripts/build_regime_blend_oof.py`에서 복구 recipe 제공 | champion `build_submit_features`, groups=`matchup,count,rates`, ID 제외 52열, 고정 category levels | sklearn HGB: `max_iter=50`, `min_samples_leaf=3000`, `learning_rate=.1`, `max_leaf_nodes=31`, `random_state=42`, 나머지는 직렬화 모델의 sklearn 1.8.0 파라미터 | stage 결합 후 c0 logit intercept | 없음 | 원 학습 저장소/테스트 유실 | recipe-reconstructed |
| `cat` | 원본 유실; 새 재생성 script에서 복구 recipe 제공 | HGB와 같은 52열, categorical 5열의 결측=`__NA__` | CatBoost: 500 trees, depth 6, lr .05, `l2_leaf_reg=20`, subsample .8, seed 42; native 모델에서 전체 파라미터 복구 | stage 결합 후 c0 logit intercept | 없음 | 원 학습 저장소/테스트 유실 | recipe-reconstructed |
| `nn` | 원본 entrypoint 유실; `src.nn_embed` 구조와 champion meta 사용 | HGB와 같은 row-local 파생 + pitcher/batter ID embedding, champion 고정 category order | emb 8, hidden 128/64, dropout .1, ID dropout .08, Adam lr .001, batch 4096, seed 42, random train-only holdout 10% | stage 결합 후 c0 logit intercept | train-only ID vocab/normalization | 원 entrypoint와 원 torch 2.12 환경 유실 | recipe-reconstructed |
| `team` | `src/train_base.py`, `src/train_monotone.py` | `src.train_base.add_features`, train-only category maps, raw player IDs 포함 57열 | eng31/leaf63/decay85/mono63의 seed·leaf·round·decay·monotone 설정 보존 | inner forward season에서 group별 Platt를 fit한 뒤 고정 0.609056/0.390944 결합 | train-only category maps | 없음 | source-exact |
| `team_nn` | `src/train_nn_candidate.py`, `src/nn_embed.py` | `build_nn_features`, train-only prep/ID vocab | `nn`과 같은 구조·학습 파라미터 | 없음; `team`과 0.9/0.1 probability mix | train-only ID vocab/normalization | 없음 | source-exact recipe |

## Byte-level provenance checks

- champion `team_model.lgbmstack.json`의 eng31/leaf63/decay85/mono63 model string은
  현재 `submission/model/lgbm_*.txt`와 각각 SHA-256까지 동일하다.
- champion `team_nn_model.npz`는
  `candidates/nn10/model/nn_model.npz`와 SHA-256
  `b3f3e38f7d635d0b12e3cf7e1a3ee7d1b7f4812076e2db81865a1704ea6fee4f`로
  동일하다.
- champion의 `ours` NN은 별도 SHA-256
  `035ee5d7391dbcaf1d9c7e0941df9ad7a9c714d6b23c74aa44734db5a4108116`이며,
  category level 순서와 일부 전처리 상수 표현도 team NN과 다르다.

## Stage reconstruction

```text
ours_raw   = 0.4*hgb + 0.4*cat + 0.2*nn
ours_stage = sigmoid(logit(ours_raw) + c0)

team_lgbm  = 0.6090563539773146 * Platt(group_1)
           + 0.3909436460226854 * Platt(group_2)
team_stage = 0.9*team_lgbm + 0.1*team_nn

champion_base = 0.4*ours_stage + 0.6*team_stage
```

`ours`의 c0는 학습 시즌별 전체 target rate에 직선을 fit해 다음 시즌 rate를
예측하고, 그 값과 cutoff 마지막 시즌의 raw prediction 평균을 logit 공간에서
맞추는 방식이다. 최종 train 2019~2024로 계산한 2025 forecast는
`0.47469465355296947`이고 champion meta와 일치한다.

team Platt는 각 outer validation의 직전 inner forward prediction/label로만 fit한다.
예를 들어 2024 prediction의 calibration은 train<=2022 모델이 2023에 낸
prediction과 2023 label로 fit하고, train<=2023 모델의 2024 prediction에 적용한다.

## Endpoint classification

| endpoint | 재현 방식 | temporal 주의점 |
| --- | --- | --- |
| pitcher season form | cutoff 이전 label로 `(n0,s0)` table 생성, 직전 시즌의 공식 as-of row로 `mu` 계산 | target season label/집계 금지 |
| batter season form | cutoff 이전 label로 `(n0,s0)` table 생성 | `alpha`는 기존 고정 endpoint; 2022/2023 자체 튜닝 점수로 해석하지 않음 |
| cold-start expert | cutoff 이전 R row만 학습, ID 2열 제외, cutoff R pitcher set으로 gate | target pitcher 빈도/label 금지 |

## Parity gate

Regime 실험 전에 다음을 만족해야 한다.

1. 모든 member/stage prediction이 finite이고 `[0,1]` 범위다.
2. stage/champion 산술식이 `atol=0`으로 일치한다.
3. 2024 fold에서 team Platt 계수가 보존된 full recipe의 기록과 합리적으로 가깝다.
4. fold별 row 수와 positive rate가 원본 train의 season 집계와 같다.
5. source-exact member와 recipe-reconstructed member를 결과에서 구분한다.

`ours` 원 entrypoint가 없으므로 temporal OOF 전체를 “historical artifact와 bit-exact”라고
주장할 수는 없다. 다만 recovered recipe의 fold parity가 안정적이면 champion 구조에
대한 가장 가까운 재구성 실험으로 사용하고 이 한계를 최종 verdict에 남긴다.

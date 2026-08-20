# Model diversity temporal OOF

실행일: 2026-08-21

## 결론

**C. no useful model diversity signal**

XGBoost hist, TabM, FT-Transformer 모두 현재 ET25 w=.020 champion을 이기거나 안정적으로 보완하지 못했다. 세 learner가 모두 complementarity gate에서 탈락했으므로 사전 정의 blend weight(`.005`, `.010`, `.020`, `.050`)는 평가하지 않았고 production ZIP도 생성하지 않았다.

현재 immutable champion은 그대로 유지한다.

- Artifact: `artifacts/sub_et25_w020.zip`
- LB BSS: `1017.0233029621`
- SHA-256 before/after: `316bf6dab11fb78f06cc74089ea38bdb3b957355f4510db5fcc8a485dc26cecf`

## 실험 계약

- Temporal folds: `<=2021 -> 2022`, `<=2022 -> 2023`, `<=2023 -> 2024`
- 입력: accepted ID-free raw HGB-52 contract 그대로 사용
- 새 feature, calibration, seed ensemble, hyperparameter search 없음
- current champion OOF는 original champion과 frozen ET25를 `0.98:0.02`로 재구성했으며 기존 weight reference와 모든 fold에서 absolute diff `0.0`
- test 데이터는 읽지 않음

Complementarity gate는 실행 전에 다음처럼 고정했다.

- Path A: 한 fold standalone gain `>= 1e-5`, 2023/2024 gain 각각 `>= -5e-4`
- Path B: pooled champion correlation `<= .90`, 2023/2024 champion top-10% error win rate 각각 `>= .60`, 두 rate 차이 `<= .15`, 2023/2024 gain 각각 `>= -5e-4`
- Path A 또는 B를 통과한 learner에만 small blend를 적용

## Temporal OOF 결과

Gain은 `current champion Brier - learner Brier`이므로 양수가 개선이다.

| learner | season | standalone Brier | gain vs champion | corr champion | corr ET25 | champion top-10% error에서 learner win | learner top-10% error에서 champion win |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost hist | 2022 | 0.243758325 | -6.433e-4 | 0.956964 | 0.812437 | 64.97% | 59.05% |
| XGBoost hist | 2023 | 0.253519820 | -4.926e-4 | 0.959887 | 0.798140 | 56.51% | 60.62% |
| XGBoost hist | 2024 | 0.248259744 | -1.950e-4 | 0.846040 | 0.545209 | 33.55% | 97.50% |
| XGBoost hist | pooled | 0.248497526 | -4.415e-4 | 0.937017 | 0.764495 | 57.46% | 70.67% |
| TabM | 2022 | 0.244219318 | -1.104e-3 | 0.971150 | 0.832375 | 37.13% | 91.58% |
| TabM | 2023 | 0.255683376 | -2.656e-3 | 0.960587 | 0.812120 | 32.68% | 82.63% |
| TabM | 2024 | 0.249072391 | -1.008e-3 | 0.726146 | 0.407061 | 23.47% | 99.56% |
| TabM | pooled | 0.249637910 | -1.582e-3 | 0.918325 | 0.743134 | 40.24% | 92.59% |
| FT-Transformer | 2022 | 0.243530071 | -4.150e-4 | 0.959963 | 0.810002 | 66.40% | 58.87% |
| FT-Transformer | 2023 | 0.255267969 | -2.241e-3 | 0.960289 | 0.803612 | 32.20% | 85.13% |
| FT-Transformer | 2024 | 0.248192502 | -1.278e-4 | 0.848855 | 0.544234 | 26.01% | 97.35% |
| FT-Transformer | pooled | 0.248973989 | -9.180e-4 | 0.940732 | 0.765716 | 49.80% | 79.26% |

### XGBoost hist

사용자 지정 recipe 하나만 실행했다: 600 trees, depth 6, learning rate .03, subsample/column sample .8, min child weight 5, lambda 5, seed 42, `tree_method=hist`. Early stopping이나 parameter search는 사용하지 않았다.

모든 fold의 standalone gain이 음수였다. 2024 prediction correlation은 낮아졌지만 champion high-error 보완율도 33.55%로 함께 붕괴했다. Path A/B 모두 실패했다.

### TabM

XGBoost gate 실패 후에만 조건부 실행했다. 하나의 small recipe(`k=8`, 2 blocks, width 128, 8 epochs, seed 42)를 사용했다. 공식 TabM contract에 맞춰 k개 head loss를 각각 최적화하고 classification inference에서는 head별 probability를 평균했다. 공식 구현이 안내하는 default AdamW learning rate `.002`, weight decay `.0003`을 사용했다.

2023/2024 degradation이 catastrophic floor보다 컸고 high-error 보완율도 낮아 Path A/B 모두 실패했다. 구현 근거는 [TabM official repository](https://github.com/yandex-research/tabm/blob/main/README.md)와 [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf)를 따른다.

### FT-Transformer

기존 `src/nn_embed.py`는 player embedding과 2-layer ReLU MLP이고, 후보는 per-feature token, multi-head self-attention, CLS token을 사용하므로 사실상 같은 구조가 아니라고 판정한 뒤 마지막 조건부 단계로 실행했다.

공식 default-like 2-block backbone(128 dimension, 8 heads)을 하나만 사용했다. 2022 high-error rows에서는 66.40%의 국소 보완 신호가 있었지만 2023에서 standalone gain `-2.241e-3`, win rate 32.20%로 붕괴했다. 2024도 standalone과 complementarity가 모두 음수여서 Path A/B 모두 실패했다. backbone은 [rtdl-revisiting-models official repository](https://github.com/yandex-research/rtdl-revisiting-models)의 `FTTransformer.get_default_kwargs`와 default optimizer를 사용했다.

## Blend, subset, production

세 learner 모두 Stage C를 통과하지 못했으므로 Stage D blend를 실행하지 않았다. 따라서 `blend_results.csv`는 의도적으로 header-only이며 worst major subset gain은 N/A다. gate 이전 모델에 weight를 적용해 subset 결과를 고르는 것은 사전 정의 stage policy를 위반한다.

- Selected candidate: 없음
- Production ZIP: 생성하지 않음
- Champion 변경: 없음
- Verdict: C

## 재현

사용 환경 주요 버전:

- Python 3.12
- XGBoost 3.4.1
- TabM 0.0.3
- rtdl-revisiting-models 0.0.2
- PyTorch 2.8.0, MPS (TabM/FT-Transformer)

```bash
.venv/bin/python scripts/validate_model_diversity.py
```

fold prediction cache가 존재하면 재학습하지 않고 결과표와 summary를 다시 검증·생성한다. 재학습이 필요한 경우에만 `--force`를 사용한다.

## 결과물

- `artifacts/model_diversity/xgb_results.csv`
- `artifacts/model_diversity/tabm_results.csv`
- `artifacts/model_diversity/ft_transformer_results.csv`
- `artifacts/model_diversity/complementarity.csv`
- `artifacts/model_diversity/blend_results.csv`
- `artifacts/model_diversity/summary.json`
- `scripts/validate_model_diversity.py`

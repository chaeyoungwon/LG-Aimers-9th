# Champion residual discovery

이 실험은 기존 temporal OOF의 `champion_final` 이후 residual structure를 진단한다.
Test data, TrackMan, submission runtime을 읽거나 수정하지 않는다.

## 실행

먼저 `artifacts/regime_blend_oof/cache/season_2021.npz`부터
`season_2024.npz`까지 생성되어 있어야 한다.

```bash
python3 scripts/run_residual_discovery.py \
  --train /Users/wooh/Documents/dev/open/data/train.csv \
  --oof-dir artifacts/regime_blend_oof \
  --output artifacts/residual_discovery
```

Residual model validation은 `2022 <- 2021`, `2023 <- 2022`,
`2024 <- 2022+2023` 순방향 계약을 사용한다. Quantile, imputation, scaling,
category vocabulary도 같은 과거 OOF에서만 fit한다.

결과 해석과 최종 결정은
`docs/experiments/2026-08-20-champion-residual-discovery.md`에 기록한다.

# Regime-aware blend OOF reconstruction results

작성일: 2026-08-20

## Current champion

- immutable artifact: `artifacts/submit_r10_pitcher_w418194.zip`
- LB BSS: `1016.4442212358`
- base: `0.4 * ours_stage + 0.6 * team_stage`
- endpoint: pitcher form, batter form, regular-season cold-start expert
  (`w=0.4181944103545871`)

## OOF reconstruction

| member | reproducible | source | result |
| --- | --- | --- | --- |
| `hgb` | recipe-level | champion sklearn model/meta | recovered 52-column feature recipe and full HGB parameters |
| `cat` | recipe-level | champion native CatBoost model/meta | recovered feature recipe and full native parameters |
| `nn` | recipe-level | champion NPZ/meta + `src.nn_embed` | recovered architecture/preprocessing; original entrypoint and torch 2.12 runtime missing |
| `team` | source-exact recipe | `src/train_base.py`, `src/train_monotone.py` | four LightGBM model strings match the champion/submission models by SHA-256 |
| `team_nn` | source-exact recipe | `src/train_nn_candidate.py`, `src/nn_embed.py` | final NPZ matches `candidates/nn10` by SHA-256 |
| form/coldstart | source-exact structure | current endpoint scripts and champion constants | every lookup/gate rebuilt from rows at or before the fold cutoff |

Official `trackman_history.csv` was available but not used: none of the five current champion
members declares the `tm` or `platoon` feature group. The reconstruction reads no test rows.

## Fold parity

There is no historical OOF/cache to use as a numeric fold reference. The table therefore reports
the reconstructed fixed `0.4/0.6` base and full champion-equivalent endpoint, not a claimed
bit-exact historical OOF comparison.

| validation fold | rows | positive rate | champion base Brier | champion final Brier | status |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 2021 | 247,088 | 0.532762 | 0.246921238 | 0.246662486 | pass |
| 2022 | 247,472 | 0.528920 | 0.243275229 | 0.243116332 | pass |
| 2023 | 245,525 | 0.499957 | 0.253273720 | 0.253041573 | pass |
| 2024 | 253,507 | 0.486105 | 0.248304517 | 0.248069141 | pass |

All member and stage predictions are finite, contain no NaNs, and remain in `[0,1]`. Stage
arithmetic is checked exactly. As a source-recipe diagnostic, fitting the 2024 team raw prediction
to the historical holdout gives a maximum Platt coefficient difference of `1.797e-4`, below the
predefined `5e-4` tolerance. The actual 2024 OOF prediction does not use 2024 labels; its
calibration is fitted from the previous forward fold.

## Regime experiment

Positive values are Brier improvements over the reconstructed fixed `0.4/0.6` champion base.
For each family, the row shown is the descriptively best pre-registered `(rho, tau)` setting.

| method | 2022 | 2023 | 2024 | pooled | worst | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| global `rho=1e-5` | -1.359e-4 | +2.679e-5 | +3.580e-4 | +8.536e-5 | -1.359e-4 | reject |
| game_type `rho=1e-5,tau=50000` | -1.044e-4 | +2.952e-5 | +4.237e-5 | -1.051e-5 | -1.044e-4 | reject |
| hand_combo `rho=0,tau=2000` | -1.457e-4 | +1.962e-5 | +3.580e-4 | +7.974e-5 | -1.457e-4 | reject |
| count_state `rho=1e-5,tau=10000` | -1.254e-4 | +2.389e-5 | +3.550e-4 | +8.685e-5 | -1.254e-4 | reject |
| game_type_count_bucket `rho=1e-5,tau=50000` | -1.058e-4 | +2.984e-5 | +1.990e-4 | +4.231e-5 | -1.058e-4 | reject |

The most favorable pooled result is the count-state blend, but it breaches the predefined
`1e-5` worst-fold loss limit by more than an order of magnitude. Applying the unchanged endpoint
does not repair the direction conflict:

| count-state endpoint | 2022 | 2023 | 2024 | pooled |
| --- | ---: | ---: | ---: | ---: |
| blend only | -1.254e-4 | +2.389e-5 | +3.550e-4 | +8.685e-5 |
| + existing form | -7.907e-5 | +1.888e-5 | +2.842e-4 | +7.651e-5 |
| + form and coldstart | -8.025e-5 | +2.057e-5 | +2.478e-4 | +6.430e-5 |

The global optimum is also unstable. Its fitted `ours_stage/team_stage` weights for validation
2022, 2023, and 2024 are respectively `0.102/0.898`, `0.455/0.545`, and `1.000/0.000`.
Count-state cell optima similarly change direction across seasons. Full `n`, raw local optimum,
shrunk weight, global weight, and outer-fold cell gain are saved in
`artifacts/regime_blend_results/regime_weight_diagnostics.{json,md}`.

## Best candidate

- method: `regime=count_state,rho=1e-5,tau=10000`
- mechanism: 12 count cells, local constrained Brier optimum shrunk toward the inner-fold global
  optimum with `lambda=n/(n+10000)`
- latest fold: `+3.550e-4` base Brier gain; `+2.478e-4` after form/coldstart composition
- pooled: `+8.685e-5` base gain; `+6.430e-5` after form/coldstart composition
- stability: reject; 2022 worsens and both global and local weights move materially by fold

This is the least-bad research result, not a packaging candidate.

## Missing artifacts and limitations

- historical temporal OOF/cache: missing, so no fold-level bit parity target exists
- original `hgb`, `cat`, and ours `nn` training entrypoints/tests: missing
- original ours-NN torch 2.12 environment: missing; reconstruction used torch 2.8

Consequently, `hgb`/`cat`/`nn` are recipe reconstructions from final serialized metadata rather
than historical source-exact folds. No unavailable member was replaced with a different model.
The exact reconstruction environment used here is pinned in
`experiments/regime_blend/requirements-oof.txt`.

## Safety and recommendation

- `submission/` was not modified and no new submission ZIP was created.
- champion SHA-256 remains
  `ce10c428c9b6bd88ebf713fcc05b91ce22fc555573f3d11f8deb283637484c47`.
- every fitted weight, lookup, calibration, form statistic, and cold-start gate uses only the inner
  temporal cutoff; validation labels are score-only.
- all generated regime predictions remain row-independent because runtime regimes use only the
  current row and train-derived fixed weights/lookups.

**Recommendation: B. no stable gain — do not submit.**

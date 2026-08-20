# Baseball domain-state feature experiment

Date: 2026-08-20

## Verdict

**C. no useful domain feature signal**

Three leakage-safe, derivable feature families were evaluated independently.
All three produced some positive 2022/2023 signal, but every family worsened
2024 OOF. None passed the required positive-2023, positive-2024, and pooled
`>= 1e-5` gate.

Consequently, no feature ablation, champion complementarity analysis, small
blend, full-train model, or production ZIP was created.

The immutable champion remains:

- LB BSS: `1017.0233029621`
- Artifact: `artifacts/sub_et25_w020.zip`
- SHA-256 before/after: `316bf6dab11fb78f06cc74089ea38bdb3b957355f4510db5fcc8a485dc26cecf`

## Research hypotheses

External baseball research was used only to define possible state variables;
no published effect size or direction was inserted into predictions.

- A live-game collegiate study reported associations between pitch count,
  fatigue, and pitching kinematics, motivating workload hypotheses: [The
  Impact of Fatigue on the Kinematics of Collegiate Baseball
  Pitchers](https://pubmed.ncbi.nlm.nih.gov/26535338/).
- A time-through-the-order study discusses pitcher decline and batter
  familiarity while warning that continuous pitcher evolution and discrete
  familiarity effects are hard to separate, motivating continuous matchup
  history rather than a hard-coded third-time cutoff: [A Bayesian analysis of
  the time through the order penalty in
  baseball](https://arxiv.org/abs/2210.06724).
- A controlled biomechanics comparison found no meaningful windup/stretch
  differences in most studied measures, supporting only a weak runner-state
  prior: [Biomechanical Comparison of the Fastball from Wind-up and the
  Fastball from Stretch](https://doi.org/10.1177/0363546507308938).

These studies do not establish effects for KBO `control_success`; temporal OOF
is the sole predictive selection criterion.

## Schema and safety audit

The main train table has season, anonymous pitcher/batter IDs, inning,
top/bottom, pre-pitch count, base/runner state, game type, H1/H2 hand codes,
and official pre-pitch pitcher/batter cumulative histories.

It does not have a game ID, exact date, timestamp, or within-game pitch order.
`row_id` is only a sample identifier and was not treated as chronology.
Therefore the highest-priority workload and familiarity hypotheses cannot be
constructed safely:

| Domain state | Status | Reason |
|---|---|---|
| Game/inning workload | UNSAFE | No game ID, date, or verified pitch order |
| Previous/recent-game workload | UNSAFE | Games and chronology cannot be reconstructed |
| Pitcher-batter familiarity | UNSAFE | No prior-only same-game/pair sequence |
| Game progression | UNSAFE | Inning exists, but pitcher game pitch index does not |
| Pitcher cumulative reliability | AVAILABLE | Official `asof_pitcher_*` inputs |
| Batter cumulative reliability | AVAILABLE | Official `asof_batter_*` inputs |
| Count/runner pressure | DERIVABLE | Current-row pre-pitch state only |
| H1/H2 interactions | DERIVABLE | Official mapping to R/L is unavailable |
| Weather | UNAVAILABLE | No date, stadium, temperature, or weather |
| Mechanics consistency | UNAVAILABLE | No deterministic main/TrackMan player crosswalk |

The main `pitcher_id` and `trackman_history.pitcher_trackman_id` sets have zero
direct overlap. No name inference, fuzzy matching, external identity data, or
manual crosswalk was attempted. TrackMan release, velocity, spin, and movement
features were therefore closed without OOF modeling.

The full 26-item audit is in `schema_audit.csv`; the 36 candidate/duplicate
features and their status are in `feature_inventory.csv`.

## Evaluated feature families

The baseline was the cached source-exact `team_raw_mono63` LightGBM temporal
OOF. Each candidate used the identical 220-round, seed-42, leaf-63 monotone
recipe and added exactly one family.

### Pressure state

- categorical count pressure: 0-0, hitter ahead, pitcher ahead, two strike,
  three ball, full count, balanced
- categorical runner pressure: empty, runners on, RISP proxy, bases loaded
- count pressure × runner pressure

### Pitcher state × pressure

- `asof_pitcher_n` experience bucket using fold-training quartiles
- recent-five-game-minus-career form bucket using fold-training tertiles
- experience bucket × count pressure
- form bucket × count pressure

The continuous `log1p(asof_pitcher_n)` and recent-career gap already exist in
the source-exact baseline. This family tests only a new nonlinear categorical
state representation and its pressure interactions.

### Handedness interaction

- H1/H2 pitcher-batter combination
- hand combination × count pressure
- hand combination × form bucket

No R/L semantic label was inferred. Every fold's quantiles and categorical
maps were fitted from its training rows only.

## Temporal OOF results

Positive gain means lower Brier loss than the same source-exact mono63 baseline.

| Family | 2022 gain | 2023 gain | 2024 gain | Pooled gain | Decision |
|---|---:|---:|---:|---:|---|
| pressure state | +2.392766e-5 | +1.300906e-4 | -3.009884e-5 | +4.049766e-5 | reject |
| pitcher state × pressure | +3.211675e-5 | +5.317879e-6 | -1.093292e-5 | +8.683272e-6 | reject |
| handedness interaction | +6.895895e-5 | +4.457904e-5 | -2.820646e-5 | +2.794378e-5 | reject |

Pressure state has the largest pooled gain and a substantial 2023 result, but
its 2024 loss violates the explicit cross-season gate. Handedness shows the
same pattern. Pitcher state is smaller than the strong pooled threshold and
also fails 2024. This consistent sign reversal indicates temporal instability,
not a deployable baseball-state representation.

Feature importance and SHAP were not used to rescue any candidate. Because no
full family passed, internal one-feature-removal ablations were correctly gated
off.

## Complementarity and blending

Champion complementarity was reserved for a strong independent family. Since
none passed, `complementarity.csv` contains its documented empty schema and the
fixed 1%, 2%, 5%, and 10% blends were not evaluated.

This avoids using a positive pooled average to hide a negative latest-season
result. No leaderboard information influenced family or blend selection.

## Leakage and row independence

- Forward folds are train through 2021→2022, through 2022→2023, and through
  2023→2024.
- Validation targets are used only for Brier scoring.
- Flipping every 2024 target changed zero domain-feature cells.
- Removing validation rows left the retained rows' features identical.
- Training for the 2024 fold ends at 2023.
- Quantile buckets and category maps use fold-training rows only.
- The feature builder gives identical retained-row output for single, full,
  reverse, and subset inputs.
- No test rows, test distribution, test player frequency, or leaderboard result
  was read.

For memory safety, missing fold predictions are trained in isolated worker
processes and cached. The parent analysis process only loads completed fold
predictions, preventing native LightGBM allocation from accumulating across
many sequential fits.

## Conclusion

The official data can represent pressure, reliability, form, and handedness
states, but its existing row-level features already expose most of their raw
information to a tree learner. The new categorical interactions do not remain
beneficial in 2024. More physically meaningful workload, familiarity, or
mechanics states cannot be reconstructed without unsafe chronological or player
identity assumptions.

The domain-feature axis is therefore closed at C for the currently available
official schema.

## Reproduction and artifacts

```bash
.venv/bin/python scripts/validate_domain_features.py
```

Outputs under `artifacts/domain_features/`:

- `schema_audit.csv`
- `feature_inventory.csv`
- `family_oof_results.csv`
- `workload_results.csv`
- `pressure_results.csv`
- `familiarity_results.csv`
- `complementarity.csv`
- `summary.json`

Contract tests are in `tests/test_domain_features.py`.

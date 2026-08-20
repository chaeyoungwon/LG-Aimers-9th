# Temporal recency experiment

Date: 2026-08-20

## Verdict

**C. no temporal recency signal**

Global season recency did not produce a stable temporal OOF improvement. No
scheme passed the required `2023 > 0`, `2024 > 0`, and `pooled > 0` gate, and
none approached the strong-signal threshold of pooled gain `>= 2e-5`.
Consequently, complementarity/blend evaluation, context diagnostics, a full
champion rebuild, and production packaging were not performed.

The immutable leaderboard champion remains:

- LB BSS: `1017.0233029621`
- File: `artifacts/sub_et25_w020.zip`
- SHA-256 before and after: `316bf6dab11fb78f06cc74089ea38bdb3b957355f4510db5fcc8a485dc26cecf`
- ET25 weight: `0.0200` (unchanged)

No test rows, test distributions, leaderboard feedback, or external data were
used in this experiment.

## Stage 0: temporal-weighting audit

| Model | Training seasons | Season weights | `sample_weight` | Recent only |
|---|---|---|---:|---:|
| HGB | All seasons available by cutoff | Uniform | No | No |
| CatBoost | All seasons available by cutoff | Uniform | No | No |
| ours NN | All seasons available by cutoff | Uniform | No | No |
| team LightGBM | All seasons available by cutoff | eng31/leaf63/mono63 uniform; one distinct decay85 member | Yes, decay85 only | No |
| team NN | All seasons available by cutoff | Uniform | No | No |
| cold-start expert | All regular-season rows available by cutoff | Uniform | No | No |
| ExtraTrees ET25 | All seasons available by cutoff | Uniform | No | No |

The team LightGBM ensemble already has one season-decayed component, but it is
a distinct leaf63/seed recipe and has limited ensemble mass: 15.8753% in group
1 and 4.5001% in group 2. The experiment therefore isolated the uniform
`team_raw_mono63` LightGBM recipe rather than duplicating that member.

## Experimental contract

The selected learner was the source-exact, reproducible
`team_raw_mono63` LightGBM model with monotone constraints and 220 boosting
rounds. Its cached uniform OOF prediction was the common baseline. The only
changed variable was the set of training seasons or their season-age weights.

Forward folds were fixed as:

- train through 2021, validate on 2022
- train through 2022, validate on 2023
- train through 2023, validate on 2024

For validation year `V` and training year `T`, exponential weight was exactly
`decay ** ((V - T) - 1)`. Candidate decays were fixed at 1.00, 0.85, 0.70,
and 0.50. Windows were fixed at all history, last four, last three, and last
two seasons. Positive gain means lower Brier loss than the uniform mono63
baseline.

Uniform baseline Brier losses were 0.243455890452 (2022), 0.253225580214
(2023), 0.247898094633 (2024), and 0.248177677521 pooled.

## Exponential-decay OOF results

| Scheme | 2022 gain | 2023 gain | 2024 gain | Pooled gain | Worst fold |
|---|---:|---:|---:|---:|---|
| decay 0.85 | -2.663274e-5 | -2.196556e-5 | -3.523592e-5 | -2.801928e-5 | 2024 |
| decay 0.70 | -3.529915e-5 | -9.295033e-5 | -8.103139e-5 | -6.979093e-5 | 2023 |
| decay 0.50 | -1.424883e-4 | -2.425367e-4 | -1.404368e-4 | -1.746975e-4 | 2023 |

Every non-uniform exponential candidate worsened every validation season.
Stronger decay produced larger pooled degradation.

## Recent-window OOF results

| Scheme | 2022 gain | 2023 gain | 2024 gain | Pooled gain | Worst fold |
|---|---:|---:|---:|---:|---|
| last 4 seasons | 0.000000e+0 | 0.000000e+0 | -6.185528e-5 | -2.100558e-5 | 2024 |
| last 3 seasons | 0.000000e+0 | +3.839916e-5 | -1.564994e-4 | -4.051651e-5 | 2024 |
| last 2 seasons | -5.943688e-5 | -2.007726e-4 | -1.897431e-4 | -1.501732e-4 | 2023 |

Zeros for last four seasons in 2022/2023 and last three seasons in 2022 mean
that the window contained all seasons available to those folds. The last-three
candidate's isolated 2023 improvement was outweighed by a much larger 2024
loss, so it is not a stable temporal signal.

## Calibration drift

| Fold | Actual rate | Uniform mean | Uniform gap | Decay 0.85 mean | Decay 0.85 gap |
|---|---:|---:|---:|---:|---:|
| 2022 | 0.528920 | 0.531835 | +0.002915 | 0.531594 | +0.002673 |
| 2023 | 0.499957 | 0.519448 | +0.019490 | 0.519109 | +0.019152 |
| 2024 | 0.486105 | 0.494935 | +0.008830 | 0.495043 | +0.008938 |

The outcome base rate fell in 2023 and 2024, but recency weighting did not
reliably track it. Decay 0.85 slightly reduced the calibration gap in 2022 and
2023, then enlarged it in 2024, while Brier loss worsened in every fold.

## Train-only drift diagnostics

Target-rate transitions were:

| Transition | Previous rate | New rate | Change |
|---|---:|---:|---:|
| 2020 to 2021 | 0.532712 | 0.532762 | +0.000050 |
| 2021 to 2022 | 0.532762 | 0.528920 | -0.003841 |
| 2022 to 2023 | 0.528920 | 0.499957 | -0.028963 |
| 2023 to 2024 | 0.499957 | 0.486105 | -0.013852 |

For 2023 to 2024, pitcher ID frequency Jensen-Shannon divergence was 0.203063
and 24.3480% of 2024 rows used a pitcher absent from 2023. Batter ID divergence
was 0.138238 with a 12.7353% row-level unseen rate. Some historical-rate
features also shifted materially. These diagnostics establish real train-only
drift, but the OOF results show that a single global season weighting is not an
effective correction for it. The deterministic `season` column's apparent
numeric shift is not treated as substantive feature drift.

## Stage gates and leakage audit

No candidate had positive 2023, 2024, and pooled gains. Therefore:

- `complementarity.csv` and `blend_results.csv` contain their documented empty
  schemas; no post-hoc candidate was searched.
- No seen/low-history/high-history context analysis was run because its
  prerequisite recency expert did not exist.
- The fixed 0.02/0.05/0.10/0.15 champion blends were not evaluated.
- The current ET25 component was not changed or removed.
- No full champion recency rebuild or production ZIP was created.

The leakage test used the 2024 fold. Validation rows were absent from training;
flipping all validation targets left training row selection, season weights,
and preprocessing category maps unchanged. The maximum sample-weight change
was exactly 0.0.

## Reproduction and artifacts

Run:

```bash
.venv/bin/python scripts/validate_temporal_recency.py
```

Outputs are under `artifacts/temporal_recency/`:

- `model_audit.csv`
- `recency_grid.csv`
- `window_results.csv`
- `drift_results.csv`
- `complementarity.csv`
- `blend_results.csv`
- `summary.json`

The implementation is covered by `tests/test_temporal_recency.py` for the
fixed candidate grids, exact age-minus-one weighting, temporal windowing,
future/validation-row rejection, the strong-signal gate, and the member audit.

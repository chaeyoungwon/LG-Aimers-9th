# 2024 rule-regime boundary analysis

Date: 2026-08-20

## Verdict

**B. WEAK / LOCALIZED SHIFT**

The 2023 to 2024 boundary contains a small count-dependent conditional shift,
but it is not a strong or production-transferable rule-regime signal. Only the
`count_state` family passed the preregistered localized-B gate. No family passed
the strong-A magnitude, residual-alignment, and persistence gate together.

Stage B was therefore stopped. No correction expert, champion rebuild, test
inference, or production ZIP was created.

The immutable champion remains:

- LB BSS: `1017.0233029621`
- Artifact: `artifacts/sub_et25_w020.zip`
- SHA-256 before/after: `316bf6dab11fb78f06cc74089ea38bdb3b957355f4510db5fcc8a485dc26cecf`
- ET25 weight: `0.0200`, unchanged

## Data and target contract

The official `train.csv` exposes `season`, but no `year`, `game_date`, or game
identifier. Exact game counts cannot be reconstructed without guessing. The
reported unit is therefore a pitch row, and `game_count` is explicitly null in
`season_summary.csv`.

| Season | Rows | Success rate | Pitchers | Batters |
|---:|---:|---:|---:|---:|
| 2019 | 237,413 | 0.564670 | 355 | 400 |
| 2020 | 244,087 | 0.532712 | 356 | 371 |
| 2021 | 247,088 | 0.532762 | 386 | 398 |
| 2022 | 247,472 | 0.528920 | 390 | 403 |
| 2023 | 245,525 | 0.499957 | 382 | 397 |
| 2024 | 253,507 | 0.486105 | 391 | 424 |

The official description defines `control_success=1` as control success under
an operational criterion and `0` as failure. It does not disclose a direct
mapping from called ball/strike outcomes to this target. `balls_before` and
`strikes_before` are pre-pitch context, while the current pitch's judgment and
result are deliberately unavailable.

Consequently, a possible causal path is only a hypothesis: a changed rules or
league environment could alter called outcomes, pitch strategy, future count
composition, or an operational control label if that label uses such outcomes.
The supplied data cannot establish which path exists, nor identify ABS as the
cause. The analysis only tests whether the 2024 timing boundary coincides with
a distinctive train-data conditional shift.

Handedness is reported as `H1`/`H2`. The official description calls these
left/right type codes but does not map code 1 or 2 to R/L, so no direction was
guessed.

## Analysis contract

The fixed boundaries were 2021→2022, 2022→2023, and 2023→2024. The first two
are placebos for the preregistered 2024 hypothesis. Every cell reports sample
sizes, frequency, conditional rates, raw delta, standard error, Wald 95%
interval, and a shrunken delta.

Cell deltas are shrunk toward the boundary-wide target-rate shift with:

```text
n_eff = harmonic_mean(n_previous, n_current)
factor = n_eff / (n_eff + 1000)
shrunk_delta = global_delta + factor * (raw_delta - global_delta)
```

Cells below 500 rows in either season are marked small and excluded from family
gates. Structural ranking requires at least 1,000 rows in both seasons.

The localized-B family gate requires at least four eligible cells, weighted
absolute conditional deviation `>= 0.004`, placebo distinctness `>= 1.25`, at
least 25% raw-significant cells, and champion-residual direction alignment
`>= 0.60`. The strong-A gate raises magnitude to `0.010`, distinctness to
`1.50`, significant fraction to `0.50`, alignment to `0.70`, and additionally
requires persistence correlation `>= 0.30` with sign agreement `>= 0.60`.

## Global target shift

| Boundary | Previous rate | New rate | Delta |
|---|---:|---:|---:|
| 2021→2022 | 0.532762 | 0.528920 | -0.003841 |
| 2022→2023 | 0.528920 | 0.499957 | -0.028963 |
| 2023→2024 | 0.499957 | 0.486105 | -0.013852 |

The largest global fall occurred at the 2022→2023 placebo boundary, not at
2024. The 2024 rate change alone is therefore not evidence of a unique rule
regime.

## Count × season

All twelve legal pre-pitch counts had lower rates in 2024, but the amount of
decline varied.

| Count | n 2023 | n 2024 | Rate 2023 | Rate 2024 | Raw delta | 95% CI | Shrunk delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-0 | 63,138 | 65,317 | 0.502851 | 0.486290 | -0.016561 | [-0.022029, -0.011093] | -0.016519 |
| 0-1 | 30,327 | 31,402 | 0.512052 | 0.501051 | -0.011001 | [-0.018890, -0.003112] | -0.011091 |
| 0-2 | 14,526 | 16,155 | 0.501790 | 0.492293 | -0.009496 | [-0.020701, +0.001708] | -0.009764 |
| 1-0 | 25,591 | 26,438 | 0.496581 | 0.479915 | -0.016666 | [-0.025256, -0.008075] | -0.016561 |
| 1-1 | 25,293 | 25,313 | 0.505080 | 0.486232 | -0.018848 | [-0.027559, -0.010137] | -0.018658 |
| 1-2 | 23,456 | 24,399 | 0.504306 | 0.497807 | -0.006499 | [-0.015460, +0.002463] | -0.006794 |
| 2-0 | 8,774 | 9,392 | 0.492136 | 0.473275 | -0.018861 | [-0.033400, -0.004321] | -0.018363 |
| 2-1 | 13,479 | 13,446 | 0.496921 | 0.474119 | -0.022802 | [-0.034739, -0.010866] | -0.022184 |
| 2-2 | 20,380 | 20,640 | 0.497301 | 0.490649 | -0.006652 | [-0.016329, +0.003025] | -0.006987 |
| 3-0 | 2,858 | 3,214 | 0.487754 | 0.481954 | -0.005800 | [-0.030984, +0.019385] | -0.007800 |
| 3-1 | 5,807 | 6,020 | 0.480282 | 0.459136 | -0.021146 | [-0.039135, -0.003157] | -0.020091 |
| 3-2 | 11,896 | 11,771 | 0.465619 | 0.456971 | -0.008648 | [-0.021350, +0.004054] | -0.009054 |

The aggregated special contexts were:

| Context | n 2023 | n 2024 | Raw delta | 95% CI | 2024 champion residual `y-p` |
|---|---:|---:|---:|---:|---:|
| Full count 3-2 | 11,896 | 11,771 | -0.008648 | [-0.021350, +0.004054] | -0.018346 |
| All 3-ball | 20,561 | 21,005 | -0.011423 | [-0.021016, -0.001830] | -0.014507 |
| All 2-strike | 70,258 | 72,965 | -0.007230 | [-0.012409, -0.002051] | -0.000491 |

The count family is the sole localized-B signal:

- placebo-relative conditional distinctness: `1.8914`
- weighted absolute deviation after removing the global shift: `0.004598`
- raw-significant cells: `7/12`
- 2024 champion residual direction alignment: `0.6247`
- 2022→2023 to 2023→2024 shrunk-delta correlation: `0.4531`
- sign agreement: `1.0000`

This is coherent but small. It misses the strong-A magnitude threshold by more
than half and also misses the 0.70 residual-alignment threshold. The dominant
signal remains a broad downward base-rate movement rather than a large change
specific to a few count states.

## Game type and full count

| Context | n 2023 | n 2024 | Rate delta | 95% CI | 2024 champion residual `y-p` |
|---|---:|---:|---:|---:|---:|
| F | 25,686 | 30,010 | -0.013623 | [-0.021934, -0.005312] | -0.021289 |
| R | 219,839 | 223,497 | -0.013411 | [-0.016355, -0.010468] | -0.001103 |
| F × 3-2 | 1,202 | 1,368 | -0.034252 | [-0.072842, +0.004338] | -0.032667 |
| R × 3-2 | 10,694 | 10,403 | -0.005461 | [-0.018917, +0.007995] | -0.016463 |

F × 3-2 is the most tempting local narrative: the raw decline and champion
overprediction point in the same direction. However, its interval includes
zero, and the entire `game_type × count` family's 2024 conditional heterogeneity
is only `0.1196` times its maximum placebo value. This family fluctuated much
more at 2022→2023 and has essentially zero latest persistence correlation
(`-0.0052` after shrinkage). It is treated as generic season noise, not a rule
expert target.

## Other context families

No other core interaction passed even the localized gate.

| Family | 2024 abs conditional shift | Ratio vs placebo | Residual alignment | Latest persistence corr. |
|---|---:|---:|---:|---:|
| hand matchup | 0.005386 | 0.814 | 0.490 | +0.060 |
| base state × count | 0.009848 | 1.178 | 0.759 | -0.436 |
| inning bucket × count | 0.006587 | 0.953 | 0.621 | -0.408 |
| pitcher reliability × count | 0.008104 | 0.359 | 0.718 | -0.368 |
| pitcher form × count | 0.010537 | 0.224 | 0.620 | -0.376 |

`outs_before`, `outs_before × count`, `base_state`, `inning_bucket`, and
`top_bottom` are also present in the full artifacts. None produced a qualifying
family decision.

The reliability analysis illustrates why distribution and conditional shifts
must be separated. Established-pitcher row share fell by 0.0675 while unseen
pitcher share rose by 0.0627. Yet 2023→2024 conditional rate deltas were
-0.01494 for established, -0.01310 for low-history, and only -0.00679 for
unseen pitchers. The composition shift is large, but it does not support a
special 2024 low-history correction.

Form composition also changed: negative-form share fell by 0.1000, neutral
rose by 0.0622, and positive rose by 0.0378. Conditional deltas were -0.02442
for positive, -0.01149 for negative, and -0.00897 for neutral. The corresponding
form × count family was far less distinctive than its placebo (`0.2245`) and
had negative persistence, so it cannot justify carrying a 2024 form correction
into 2025.

## Champion residuals and exploratory ranking

Current-champion OOF is reconstructed exactly as:

```text
0.98 * original_champion + 0.02 * accepted_ET25_prefix
```

The accepted ET25 prefix OOF contract exists for 2022–2024. A new 2021 forest
was deliberately not trained, so current-champion residual comparisons begin
in 2022; conditional placebo analysis itself covers all three requested
boundaries.

The leading sufficiently sized 2024-aligned cells include positive-form × 1-1,
R × 1-1, empty-base × 0-0, early-inning × 1-1, and established-pitcher × 1-1.
They are retained as diagnostics only. The ranking combines shrunk magnitude,
cell-level placebo ratio, sample size, champion-residual direction, and parent
context direction. It is not a model-selection rule and was never applied to
test rows.

## Safety and stage decision

- Only `train.csv` and frozen temporal OOF artifacts were read.
- No test distribution, player frequency, count frequency, or test rows were
  accessed.
- Static contexts are row-local. Flipping 2024 targets leaves their values
  unchanged.
- Reliability and pitcher form use cutoff-safe OOF fields. The signed form is
  recovered as `pitcher_form_pred - champion_base`, not from validation labels.
- Small cells are marked and excluded from gates rather than trusted at face
  value.
- No 2024 target mean or cell delta was turned into a test correction.
- No Stage B learner, 2024 specialist, candidate ZIP, or leaderboard proposal
  was produced.

The data support a weak localized count interaction, but not the proposition
that 2024 introduced a large, distinct, and persistent rule regime that should
be extrapolated into 2025.

## Reproduction and artifacts

```bash
.venv/bin/python scripts/analyze_rule_regime.py
```

Outputs under `artifacts/rule_regime/`:

- `season_summary.csv`
- `count_shift.csv`
- `matchup_shift.csv`
- `game_type_shift.csv`
- `context_frequency_shift.csv`
- `conditional_shift.csv`
- `placebo_boundaries.csv`
- `champion_residual_shift.csv`
- `persistence.csv`
- `summary.json`

Unit tests are in `tests/test_rule_regime.py`.

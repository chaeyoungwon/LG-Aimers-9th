# Context-trained specialist experiment

Date: 2026-08-20

## Verdict

**C. no specialist signal**

The first-priority 2-strike specialist failed the raw temporal gate. It improved
its own subset in 2023, but worsened 2022, 2024, and pooled OOF. Per the fixed
priority contract, the experiment stopped at Stage 1; 3-ball, full-count,
game-type, and pitcher-history specialists were not trained.

No subset blend, multi-specialist system, calibration, full-train model, or
production ZIP was created.

The immutable champion remains:

- LB BSS: `1017.0233029621`
- Artifact: `artifacts/sub_et25_w020.zip`
- SHA-256 before/after: `316bf6dab11fb78f06cc74089ea38bdb3b957355f4510db5fcc8a485dc26cecf`

## Experimental contract

The specialist differs from the previous regime-blend work because its model
was trained only on rows belonging to its target context. For the first stage:

```text
training subset: strikes_before == 2
validation subset: strikes_before == 2
routing predicate: current row's strikes_before == 2
```

The learner reused the source-exact `mono63` LightGBM settings: 220 rounds,
seed 42, 63 leaves, minimum 1,200 rows per leaf, and the existing monotone
constraint directions. Features were the exact champion HGB/CatBoost ID-free
52-column row-local contract. Raw pitcher and batter IDs were excluded, no new
features were invented, and categorical levels were fitted from each fold's
specialist training rows only.

Temporal folds were fixed at train through 2021→validate 2022, through
2022→2023, and through 2023→2024. No test rows or test distribution were read.

## Raw specialist results

Positive gain means lower Brier loss than the current ET25 w=.020 champion on
the same 2-strike validation rows.

| Fold | Specialist train rows | Validation rows | Champion Brier | Specialist Brier | Subset gain | Overall hard-route diagnostic |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 204,515 | 71,381 | 0.244416 | 0.244993 | -5.770599e-4 | -1.664476e-4 |
| 2023 | 275,896 | 70,258 | 0.252848 | 0.252759 | +8.949339e-5 | +2.560890e-5 |
| 2024 | 346,154 | 72,965 | 0.248480 | 0.248577 | -9.778416e-5 | -2.814447e-5 |
| pooled | — | 214,604 | 0.248558 | 0.248754 | -1.958878e-4 | -5.631357e-5 |

All folds comfortably passed the 50,000-row training threshold. Failure was
therefore not caused by inadequate sample size. The raw gate failed because
2024 and pooled gains were negative. The full-dataset hard-route values above
are diagnostics only; hard replacement was not considered for adoption.

The result also misses the target pooled overall gain of `+2e-5` by a wide
margin: the diagnostic is negative `-5.631e-5`.

## Complementarity

| Fold | Prediction correlation | Specialist wins | Champion top-10% error rows won |
|---|---:|---:|---:|
| 2022 | 0.944993 | 0.475855 | 0.748564 |
| 2023 | 0.943494 | 0.490905 | 0.694563 |
| 2024 | 0.822670 | 0.515672 | 0.341099 |
| pooled | 0.921703 | 0.494320 | 0.654536 |

The specialist sometimes repaired large champion errors in 2022/2023, but the
behavior did not generalize to 2024: its top-error win rate fell to 34.11%, and
its aggregate Brier became worse. Pooled prediction correlation of 0.922 also
indicates limited learner diversity despite the specialized training
distribution.

## Stage gate

The preregistered raw gate required:

- positive 2023 subset gain
- positive 2024 subset gain
- positive pooled subset gain
- at least 50,000 specialist training rows in every fold

Only the 2023 and sample-size conditions passed. Because the first-priority
candidate failed, the following were deliberately not executed:

- 2-strike subset blends at 0.10, 0.25, and 0.50
- 3-ball specialist
- full-count specialist
- R/F specialists
- seen/unseen-low-history specialists
- major-subset safety selection and multi-specialist composition

`blend_results.csv` and `subset_results.csv` contain readable empty schemas to
record that these gates were not reached. Non-target routing is unit-tested as
bit-exact identity with the champion, but there is no accepted candidate that
requires a production row-independence audit.

## Safety

- Current champion predictions are reconstructed directly as 98% original
  champion plus the accepted 2% ET25 OOF member.
- The specialist sees only cutoff-safe training rows satisfying its fixed
  row-local predicate.
- Validation labels are used only for scoring.
- No calibration was added.
- No previous global/regime, residual, form, hierarchy, recency, rule-regime,
  TrackMan, or leaderboard-weight axis was reopened.
- No test distribution or leaderboard result selected a specialist or weight.
- The champion ZIP checksum is unchanged and no submission ZIP was created.

## Reproduction and artifacts

```bash
.venv/bin/python scripts/validate_context_specialist.py
```

Outputs under `artifacts/context_specialist/`:

- `specialist_results.csv`
- `subset_results.csv`
- `blend_results.csv`
- `complementarity.csv`
- `summary.json`

Prediction caches under the same artifact directory allow deterministic audit
reruns without retraining. Contract tests are in
`tests/test_context_specialist.py`.

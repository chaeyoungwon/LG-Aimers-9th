"""Validate a two-slope shape correction on the existing pitcher-form endpoint.

Only ``min(f, 0)`` and ``max(f, 0)`` are fitted, where ``f`` is the existing
pitcher form adjustment.  There is no intercept and the champion coefficient
is fixed at one.  All coefficients are learned from strictly earlier OOF.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CHAMPION_ZIP = ROOT / "artifacts" / "submit_r10_pitcher_w418194.zip"
EXPECTED_CHAMPION_SHA256 = (
    "ce10c428c9b6bd88ebf713fcc05b91ce22fc555573f3d11f8deb283637484c47"
)
SEASONS = (2022, 2023, 2024)
FIT_SEASONS = {2022: (), 2023: (2022,), 2024: (2022, 2023)}
RIDGE_LAMBDA = 1e-4
CAPS = (0.005, 0.010)
PRIMARY_CAP = 0.005
NEAR_ZERO_THRESHOLD = 0.001
MAJOR_SUBSET_MIN_ROWS = 5000
MAJOR_SUBSET_MAX_LOSS = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_season(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        required = {
            "y",
            "champion_base",
            "pitcher_form_pred",
            "champion_final",
            "x_game_type",
            "x_balls_before",
            "x_strikes_before",
            "x_pitcher_seen",
        }
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"{path}: missing arrays {missing}")
        payload = {key: loaded[key] for key in required}
    n = len(payload["y"])
    if any(len(values) != n for values in payload.values()):
        raise ValueError(f"{path}: inconsistent array lengths")
    payload["y"] = np.asarray(payload["y"], dtype="float64")
    payload["champion_final"] = np.asarray(
        payload["champion_final"], dtype="float64"
    )
    payload["pitcher_form_adjustment"] = (
        np.asarray(payload["pitcher_form_pred"], dtype="float64")
        - np.asarray(payload["champion_base"], dtype="float64")
    )
    return payload


def form_basis(form_adjustment: np.ndarray) -> np.ndarray:
    form = np.asarray(form_adjustment, dtype="float64")
    return np.column_stack([np.minimum(form, 0.0), np.maximum(form, 0.0)])


def fit_asymmetric_beta(
    form_adjustment: np.ndarray,
    residual: np.ndarray,
    ridge_lambda: float = RIDGE_LAMBDA,
) -> np.ndarray:
    """Fit [beta_neg, beta_pos] with no intercept under mean-Brier ridge."""
    basis = form_basis(form_adjustment)
    target = np.asarray(residual, dtype="float64")
    if basis.shape[0] != len(target) or len(target) == 0:
        raise ValueError("form/residual must be same-length non-empty arrays")
    if not np.isfinite(ridge_lambda) or ridge_lambda < 0.0:
        raise ValueError("ridge_lambda must be finite and non-negative")
    gram = basis.T @ basis / len(basis)
    linear = basis.T @ target / len(basis)
    return np.linalg.solve(gram + ridge_lambda * np.eye(2), linear)


def apply_asymmetric_correction(
    champion: np.ndarray,
    form_adjustment: np.ndarray,
    beta: np.ndarray,
    cap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return candidate, capped delta, and uncapped delta using row-local math."""
    prediction = np.asarray(champion, dtype="float64")
    basis = form_basis(form_adjustment)
    coefficients = np.asarray(beta, dtype="float64")
    if coefficients.shape != (2,):
        raise ValueError("beta must have shape (2,)")
    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError("cap must be finite and positive")
    raw_delta = basis @ coefficients
    delta = np.clip(raw_delta, -cap, cap)
    candidate = np.clip(prediction + delta, 0.0, 1.0)
    return candidate, delta, raw_delta


def brier(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.mean(
            (np.asarray(target, dtype="float64") - np.asarray(prediction, dtype="float64"))
            ** 2
        )
    )


def candidate_name(cap: float) -> str:
    return f"asym_pitcher_form_ridge1e-4_cap{int(round(cap * 1000)):03d}"


def nested_beta(
    validation_season: int, seasons: Mapping[int, Mapping[str, np.ndarray]]
) -> tuple[np.ndarray, dict]:
    fit_seasons = FIT_SEASONS[validation_season]
    if not fit_seasons:
        return np.zeros(2, dtype="float64"), {
            "fit_period": "identity",
            "fit_rows": 0,
            "active_rows": 0,
            "basis_energy_neg": 0.0,
            "basis_energy_pos": 0.0,
        }
    form = np.concatenate(
        [seasons[season]["pitcher_form_adjustment"] for season in fit_seasons]
    )
    residual = np.concatenate(
        [
            seasons[season]["y"] - seasons[season]["champion_final"]
            for season in fit_seasons
        ]
    )
    basis = form_basis(form)
    beta = fit_asymmetric_beta(form, residual)
    return beta, {
        "fit_period": "+".join(map(str, fit_seasons)),
        "fit_rows": len(form),
        "active_rows": int(np.count_nonzero(form)),
        "basis_energy_neg": float(np.mean(basis[:, 0] ** 2)),
        "basis_energy_pos": float(np.mean(basis[:, 1] ** 2)),
    }


def exact_row_independence_check(
    champion: np.ndarray,
    form: np.ndarray,
    beta: np.ndarray,
    cap: float,
) -> bool:
    full = apply_asymmetric_correction(champion, form, beta, cap)[0]
    n = len(full)
    indices = np.asarray(sorted(set([0, n // 3, n // 2, n - 1])), dtype="int64")
    single = np.asarray(
        [
            apply_asymmetric_correction(
                champion[index : index + 1], form[index : index + 1], beta, cap
            )[0][0]
            for index in indices
        ]
    )
    if not np.array_equal(single, full[indices]):
        return False
    subset = apply_asymmetric_correction(champion[indices], form[indices], beta, cap)[0]
    if not np.array_equal(subset, full[indices]):
        return False
    reversed_prediction = apply_asymmetric_correction(
        champion[::-1], form[::-1], beta, cap
    )[0][::-1]
    if not np.array_equal(reversed_prediction, full):
        return False
    rng = np.random.default_rng(42)
    order = rng.permutation(n)
    shuffled = apply_asymmetric_correction(champion[order], form[order], beta, cap)[0]
    restored = np.empty_like(shuffled)
    restored[order] = shuffled
    return bool(np.array_equal(restored, full))


def identity_check(seasons: Mapping[int, Mapping[str, np.ndarray]]) -> bool:
    zero = np.zeros(2, dtype="float64")
    for payload in seasons.values():
        champion = payload["champion_final"]
        candidate, delta, raw = apply_asymmetric_correction(
            champion, payload["pitcher_form_adjustment"], zero, PRIMARY_CAP
        )
        if not (
            np.array_equal(candidate, champion)
            and np.array_equal(delta, np.zeros_like(delta))
            and np.array_equal(raw, np.zeros_like(raw))
        ):
            return False
    return True


def form_scale_threshold(
    validation_season: int, seasons: Mapping[int, Mapping[str, np.ndarray]]
) -> float:
    fit_seasons = FIT_SEASONS[validation_season]
    if fit_seasons:
        values = np.concatenate(
            [
                np.abs(seasons[season]["pitcher_form_adjustment"])
                for season in fit_seasons
            ]
        )
    else:
        values = np.abs(seasons[validation_season]["pitcher_form_adjustment"])
    active = values[values > 0.0]
    return 0.0 if len(active) == 0 else float(np.quantile(active, 0.95))


def subset_masks(
    payload: Mapping[str, np.ndarray], extreme_threshold: float
) -> list[tuple[str, str, np.ndarray]]:
    form = payload["pitcher_form_adjustment"]
    game_type = np.asarray(payload["x_game_type"]).astype("U8")
    seen = np.asarray(payload["x_pitcher_seen"], dtype=bool)
    full_count = (
        np.asarray(payload["x_balls_before"], dtype="int8") == 3
    ) & (np.asarray(payload["x_strikes_before"], dtype="int8") == 2)
    masks: list[tuple[str, str, np.ndarray]] = []
    for value in ("R", "F"):
        masks.append(("game_type", value, game_type == value))
    masks.extend(
        [
            ("pitcher_seen", "seen", seen),
            ("pitcher_seen", "unseen", ~seen),
            ("full_count", "full", full_count),
            ("full_count", "non_full", ~full_count),
            ("form_sign", "negative", form < -NEAR_ZERO_THRESHOLD),
            (
                "form_sign",
                "near_zero",
                np.abs(form) <= NEAR_ZERO_THRESHOLD,
            ),
            ("form_sign", "positive", form > NEAR_ZERO_THRESHOLD),
            (
                "form_extreme",
                "extreme",
                np.abs(form) >= extreme_threshold,
            ),
            (
                "form_extreme",
                "non_extreme",
                np.abs(form) < extreme_threshold,
            ),
        ]
    )
    return masks


def fold_and_subset_results(
    seasons: Mapping[int, Mapping[str, np.ndarray]],
    betas: Mapping[int, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    fold_rows = []
    subset_rows = []
    distribution_rows = []
    safety = {}
    for cap in CAPS:
        name = candidate_name(cap)
        safety[name] = {
            "row_independent": True,
            "major_subset_losses": [],
            "extreme_positive_gain_share": {},
        }
        for season in SEASONS:
            payload = seasons[season]
            y = payload["y"]
            champion = payload["champion_final"]
            form = payload["pitcher_form_adjustment"]
            candidate, delta, raw_delta = apply_asymmetric_correction(
                champion, form, betas[season], cap
            )
            reference_brier = brier(y, champion)
            candidate_brier = brier(y, candidate)
            gain = reference_brier - candidate_brier
            safety[name]["row_independent"] = safety[name]["row_independent"] and (
                exact_row_independence_check(champion, form, betas[season], cap)
            )
            fold_rows.append(
                {
                    "candidate": name,
                    "validation_season": season,
                    "n": len(y),
                    "fit_period": "identity"
                    if not FIT_SEASONS[season]
                    else "+".join(map(str, FIT_SEASONS[season])),
                    "ridge_lambda": RIDGE_LAMBDA,
                    "cap": cap,
                    "beta_neg": float(betas[season][0]),
                    "beta_pos": float(betas[season][1]),
                    "champion_brier": reference_brier,
                    "candidate_brier": candidate_brier,
                    "brier_gain": gain,
                }
            )
            extreme_threshold = form_scale_threshold(season, seasons)
            per_row_gain = (y - champion) ** 2 - (y - candidate) ** 2
            positive_gain = np.maximum(per_row_gain, 0.0)
            extreme = np.abs(form) >= extreme_threshold
            gross_positive = float(positive_gain.sum())
            share = (
                0.0
                if gross_positive <= 0.0
                else float(positive_gain[extreme].sum() / gross_positive)
            )
            safety[name]["extreme_positive_gain_share"][str(season)] = share
            for subset_type, subset_value, mask in subset_masks(
                payload, extreme_threshold
            ):
                if not mask.any():
                    continue
                subset_gain = brier(y[mask], champion[mask]) - brier(
                    y[mask], candidate[mask]
                )
                subset_rows.append(
                    {
                        "candidate": name,
                        "validation_season": season,
                        "subset_type": subset_type,
                        "subset_value": subset_value,
                        "n": int(mask.sum()),
                        "fraction": float(mask.mean()),
                        "champion_brier": brier(y[mask], champion[mask]),
                        "candidate_brier": brier(y[mask], candidate[mask]),
                        "brier_gain": subset_gain,
                        "extreme_threshold": extreme_threshold,
                    }
                )
                if (
                    season in (2023, 2024)
                    and mask.sum() >= MAJOR_SUBSET_MIN_ROWS
                    and subset_gain < -MAJOR_SUBSET_MAX_LOSS
                ):
                    safety[name]["major_subset_losses"].append(
                        {
                            "season": season,
                            "subset_type": subset_type,
                            "subset_value": subset_value,
                            "n": int(mask.sum()),
                            "brier_gain": subset_gain,
                        }
                    )
            sign_masks = {
                "all": np.ones(len(form), dtype=bool),
                "negative": form < -NEAR_ZERO_THRESHOLD,
                "near_zero": np.abs(form) <= NEAR_ZERO_THRESHOLD,
                "positive": form > NEAR_ZERO_THRESHOLD,
            }
            for form_group, mask in sign_masks.items():
                if not mask.any():
                    continue
                values = delta[mask]
                distribution_rows.append(
                    {
                        "candidate": name,
                        "validation_season": season,
                        "form_group": form_group,
                        "n": int(mask.sum()),
                        "mean_delta": float(values.mean()),
                        "median_delta": float(np.median(values)),
                        "p05_delta": float(np.quantile(values, 0.05)),
                        "p95_delta": float(np.quantile(values, 0.95)),
                        "min_delta": float(values.min()),
                        "max_delta": float(values.max()),
                        "mean_abs_delta": float(np.abs(values).mean()),
                        "capped_rate": float(np.mean(np.abs(raw_delta[mask]) > cap)),
                        "prediction_mean_before": float(champion[mask].mean()),
                        "prediction_mean_after": float(candidate[mask].mean()),
                    }
                )
    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(subset_rows),
        pd.DataFrame(distribution_rows),
        safety,
    )


def summarize_candidates(
    fold_results: pd.DataFrame,
    coefficient_stable: bool,
    safety: Mapping[str, Mapping[str, object]],
) -> list[dict]:
    rows = []
    for candidate, body in fold_results.groupby("candidate"):
        by_season = {
            int(row.validation_season): row for row in body.itertuples()
        }
        gains = {season: float(by_season[season].brier_gain) for season in SEASONS}
        pooled_all = float(np.average(body["brier_gain"], weights=body["n"]))
        evaluable = body[body["validation_season"].isin([2023, 2024])]
        pooled = float(
            np.average(evaluable["brier_gain"], weights=evaluable["n"])
        )
        candidate_safety = safety[candidate]
        no_major_loss = not candidate_safety["major_subset_losses"]
        not_extreme_only = all(
            float(value) <= 0.60
            for season, value in candidate_safety["extreme_positive_gain_share"].items()
            if int(season) in (2023, 2024)
        )
        passed = (
            gains[2023] > 0.0
            and gains[2024] > 0.0
            and pooled > 0.0
            and coefficient_stable
            and no_major_loss
            and bool(candidate_safety["row_independent"])
            and not_extreme_only
        )
        rows.append(
            {
                "candidate": candidate,
                "brier_gain_2022": gains[2022],
                "brier_gain_2023": gains[2023],
                "brier_gain_2024": gains[2024],
                "pooled_brier_gain": pooled,
                "pooled_all_seasons_brier_gain": pooled_all,
                "coefficient_sign_stable": coefficient_stable,
                "row_independent": bool(candidate_safety["row_independent"]),
                "major_subset_loss_count": len(
                    candidate_safety["major_subset_losses"]
                ),
                "not_extreme_only": not_extreme_only,
                "verdict": "pass" if passed else "reject",
            }
        )
    return sorted(rows, key=lambda row: row["pooled_brier_gain"], reverse=True)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oof-dir",
        type=Path,
        default=ROOT / "artifacts" / "regime_blend_oof",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "pitcher_form_shape",
    )
    args = parser.parse_args(argv)
    if not CHAMPION_ZIP.is_file():
        parser.error(f"champion ZIP not found: {CHAMPION_ZIP}")
    checksum_before = sha256(CHAMPION_ZIP)
    if checksum_before != EXPECTED_CHAMPION_SHA256:
        raise SystemExit(f"champion checksum mismatch: {checksum_before}")

    seasons = {}
    for season in SEASONS:
        path = args.oof_dir / "cache" / f"season_{season}.npz"
        if not path.is_file():
            parser.error(f"OOF cache not found: {path}")
        seasons[season] = load_season(path)

    if not identity_check(seasons):
        raise SystemExit("zero-beta champion identity check failed")

    betas = {}
    coefficient_rows = []
    for validation_season in SEASONS:
        beta, metadata = nested_beta(validation_season, seasons)
        betas[validation_season] = beta
        coefficient_rows.append(
            {
                "validation_season": validation_season,
                **metadata,
                "ridge_lambda": RIDGE_LAMBDA,
                "beta_neg": float(beta[0]),
                "beta_pos": float(beta[1]),
            }
        )
    fitted = [betas[season] for season in (2023, 2024)]
    coefficient_stable = bool(
        np.sign(fitted[0][0]) == np.sign(fitted[1][0])
        and np.sign(fitted[0][1]) == np.sign(fitted[1][1])
        and np.all(np.asarray(fitted) != 0.0)
    )
    print("coefficient stability (reported before Brier evaluation):")
    for row in coefficient_rows:
        print(
            f"  validation={row['validation_season']} fit={row['fit_period']} "
            f"beta_neg={row['beta_neg']:.12g} beta_pos={row['beta_pos']:.12g}"
        )
    print(f"  nested-fold signs stable: {coefficient_stable}")

    fold_results, subset_results, distributions, safety = fold_and_subset_results(
        seasons, betas
    )
    candidate_summaries = summarize_candidates(
        fold_results, coefficient_stable, safety
    )
    checksum_after = sha256(CHAMPION_ZIP)
    if checksum_after != checksum_before:
        raise SystemExit("champion checksum changed during experiment")

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coefficient_rows).to_csv(
        args.output / "coefficient_stability.csv", index=False
    )
    fold_results.to_csv(args.output / "fold_results.csv", index=False)
    subset_results.to_csv(args.output / "subset_results.csv", index=False)
    distributions.to_csv(
        args.output / "correction_distribution.csv", index=False
    )
    passing = [row for row in candidate_summaries if row["verdict"] == "pass"]
    saturation_triggered = bool(
        passing
        and any(
            float(safety[row["candidate"]]["extreme_positive_gain_share"][str(season)])
            > 0.60
            for row in passing
            for season in (2023, 2024)
        )
    )
    summary = {
        "hypothesis": (
            "positive and negative pitcher-form adjustments may require different residual slopes"
        ),
        "ridge_lambda": RIDGE_LAMBDA,
        "ridge_selection": (
            "fixed from training-basis energy only; lambda is at least as large as each 2022 basis diagonal"
        ),
        "caps": list(CAPS),
        "primary_cap": PRIMARY_CAP,
        "identity_atol_zero": True,
        "coefficient_sign_stable": coefficient_stable,
        "coefficients": coefficient_rows,
        "candidates": candidate_summaries,
        "passing_candidates": passing,
        "safety": safety,
        "saturation_basis_evaluated": False,
        "saturation_basis_reason": (
            "not evaluated: no two-slope candidate passed every safety gate"
            if not passing
            else "not evaluated: the two-slope candidate does not require an extreme-only basis"
            if not saturation_triggered
            else "eligible but intentionally deferred; keep this round at two parameters"
        ),
        "validation_labels_used_for_fit": False,
        "test_rows_read": False,
        "champion_sha256_before": checksum_before,
        "champion_sha256_after": checksum_after,
        "final_verdict": (
            "A. stable nonlinear form signal confirmed"
            if passing
            else "B. residual signal exists but cannot be safely exploited"
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved pitcher-form shape artifacts to {args.output}")
    print(f"beta sign stable: {coefficient_stable}")
    print(f"passing candidates: {len(passing)}")
    print(f"verdict: {summary['final_verdict']}")
    print(f"champion checksum unchanged: {checksum_before == checksum_after}")


if __name__ == "__main__":
    main()

"""Discover stable, leakage-safe residual structure after the R10 champion.

The script consumes reconstructed forward OOF caches plus official train rows.
It never reads test data and never writes submission files.  Validation labels
are used only for diagnostics and scoring; every residual model is fitted on
strictly earlier OOF seasons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parents[1]
CHAMPION_ZIP = ROOT / "artifacts" / "submit_r10_pitcher_w418194.zip"
EXPECTED_CHAMPION_SHA256 = (
    "ce10c428c9b6bd88ebf713fcc05b91ce22fc555573f3d11f8deb283637484c47"
)
TARGET = "control_success"
DISCOVERY_SEASONS = (2022, 2023, 2024)
SOURCE_SEASONS = (2021, 2022, 2023, 2024)
MODEL_TRAIN_SEASONS = {
    2022: (2021,),
    2023: (2022,),
    2024: (2022, 2023),
}
CORRECTION_SETTINGS = (
    (0.10, 0.005),
    (0.25, 0.010),
    (0.50, 0.020),
)
MODEL_NUMERIC_FEATURES = (
    "game_month",
    "game_dayofweek",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "pitcher_seen",
    "cold_start",
    "hgb",
    "cat",
    "nn",
    "team",
    "team_nn",
    "ours_stage",
    "team_stage",
    "champion_base",
    "champion_final",
    "member_std",
    "member_range",
    "hgb_minus_cat",
    "hgb_minus_nn",
    "ours_minus_team",
    "abs_hgb_minus_cat",
    "abs_hgb_minus_nn",
    "abs_ours_minus_team",
    "form_adjustment",
    "pitcher_form_adjustment",
    "batter_form_adjustment",
    "cold_expert_gap",
    "abs_cold_expert_gap",
)
MODEL_CATEGORICAL_FEATURES = (
    "game_type",
    "top_bottom",
    "base_state",
    "count_state",
    "pitcher_hand",
    "batter_hand",
    "same_hand",
    "inning_bucket",
)
UNIVARIATE_NUMERIC_FEATURES = tuple(
    name
    for name in MODEL_NUMERIC_FEATURES
    if name
    not in {
        "pitcher_seen",
        "cold_start",
        "runner_on_1b",
        "runner_on_2b",
        "runner_on_3b",
    }
)
DISAGREEMENT_FEATURES = (
    "member_std",
    "member_range",
    "abs_ours_minus_team",
    "abs_hgb_minus_cat",
    "abs_hgb_minus_nn",
    "abs_cold_expert_gap",
)
CATEGORICAL_DIAGNOSTIC_FEATURES = (
    "game_type",
    "count_state",
    "pitcher_hand",
    "batter_hand",
    "same_hand",
    "base_state",
    "inning_bucket",
    "num_runners_on",
    "pitcher_seen",
    "cold_start",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def build_discovery_frame(train_season: pd.DataFrame, oof: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Join row-local train features to same-order OOF predictions with hard parity checks."""
    if len(train_season) != len(oof["y"]):
        raise ValueError("train/OOF row count mismatch")
    target = train_season[TARGET].to_numpy(dtype="float64")
    if not np.array_equal(target, np.asarray(oof["y"], dtype="float64")):
        raise ValueError("train/OOF target order mismatch")
    frame = train_season.copy().reset_index(drop=True)
    for name in ("hgb", "cat", "nn", "team", "team_nn", "ours_stage", "team_stage"):
        frame[name] = np.asarray(oof[name], dtype="float64")
    for name in (
        "champion_base",
        "pitcher_form_pred",
        "batter_form_pred",
        "coldstart_expert_pred",
        "champion_final",
    ):
        frame[name] = np.asarray(oof[name], dtype="float64")
    frame["pitcher_seen"] = np.asarray(oof["x_pitcher_seen"], dtype="int8")
    frame["cold_start"] = np.asarray(oof["coldstart_mask"], dtype="int8")
    frame["count_state"] = (
        frame["balls_before"].astype("int8") * 3
        + frame["strikes_before"].astype("int8")
    ).astype("int8")
    frame["same_hand"] = (frame["pitcher_hand"] == frame["batter_hand"]).astype("int8")
    frame["inning_bucket"] = np.select(
        [frame["inning"] <= 3, frame["inning"] <= 6],
        ["1-3", "4-6"],
        default="7+",
    )

    members = frame[["hgb", "cat", "nn", "team", "team_nn"]].to_numpy()
    frame["member_std"] = members.std(axis=1)
    frame["member_range"] = members.max(axis=1) - members.min(axis=1)
    frame["hgb_minus_cat"] = frame["hgb"] - frame["cat"]
    frame["hgb_minus_nn"] = frame["hgb"] - frame["nn"]
    frame["ours_minus_team"] = frame["ours_stage"] - frame["team_stage"]
    frame["abs_hgb_minus_cat"] = frame["hgb_minus_cat"].abs()
    frame["abs_hgb_minus_nn"] = frame["hgb_minus_nn"].abs()
    frame["abs_ours_minus_team"] = frame["ours_minus_team"].abs()
    frame["pitcher_form_adjustment"] = frame["pitcher_form_pred"] - frame["champion_base"]
    frame["batter_form_adjustment"] = frame["batter_form_pred"] - frame["pitcher_form_pred"]
    frame["form_adjustment"] = frame["batter_form_pred"] - frame["champion_base"]
    frame["cold_expert_gap"] = frame["coldstart_expert_pred"] - frame["batter_form_pred"]
    frame["abs_cold_expert_gap"] = frame["cold_expert_gap"].abs()
    frame["signed_residual"] = target - frame["champion_final"]
    frame["squared_error"] = frame["signed_residual"] ** 2
    return frame


def quantile_edges(values: np.ndarray, bins: int = 5) -> np.ndarray:
    finite = np.asarray(values, dtype="float64")
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.asarray([-np.inf, np.inf])
    interior = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    return np.r_[-np.inf, interior, np.inf]


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    numeric = np.asarray(values, dtype="float64")
    labels = np.full(len(numeric), "missing", dtype="U16")
    finite = np.isfinite(numeric)
    indices = np.searchsorted(edges[1:-1], numeric[finite], side="right")
    labels[finite] = np.asarray([f"q{index + 1}" for index in indices], dtype="U16")
    return labels


def grouped_metrics(frame: pd.DataFrame, labels: np.ndarray) -> list[dict]:
    rows = []
    target = frame[TARGET].to_numpy(dtype="float64")
    prediction = frame["champion_final"].to_numpy(dtype="float64")
    residual = target - prediction
    squared = residual**2
    for label in sorted(np.unique(labels)):
        mask = labels == label
        count = int(mask.sum())
        residual_stderr = (
            0.0
            if count <= 1
            else float(residual[mask].std(ddof=1) / math.sqrt(count))
        )
        brier_stderr = (
            0.0
            if count <= 1
            else float(squared[mask].std(ddof=1) / math.sqrt(count))
        )
        rows.append(
            {
                "bin": str(label),
                "n": count,
                "prediction_mean": float(prediction[mask].mean()),
                "positive_rate": float(target[mask].mean()),
                "calibration_gap": float(residual[mask].mean()),
                "signed_residual_mean": float(residual[mask].mean()),
                "champion_brier": float(squared[mask].mean()),
                "squared_error_mean": float(squared[mask].mean()),
                "signed_residual_stderr": residual_stderr,
                "brier_stderr": brier_stderr,
            }
        )
    return rows


def univariate_diagnostics(
    frames: Mapping[int, pd.DataFrame], features: Sequence[str]
) -> pd.DataFrame:
    rows = []
    for season in DISCOVERY_SEASONS:
        validation = frames[season]
        history = pd.concat(
            [frames[source] for source in MODEL_TRAIN_SEASONS[season]], ignore_index=True
        )
        for feature in features:
            edges = quantile_edges(history[feature].to_numpy())
            labels = assign_bins(validation[feature].to_numpy(), edges)
            metrics = grouped_metrics(validation, labels)
            for row in metrics:
                index = None if row["bin"] == "missing" else int(row["bin"][1:]) - 1
                rows.append(
                    {
                        "validation_season": season,
                        "feature": feature,
                        "bin": row["bin"],
                        "bin_low": None if index is None else float(edges[index]),
                        "bin_high": None if index is None else float(edges[index + 1]),
                        **{key: value for key, value in row.items() if key != "bin"},
                    }
                )
    return pd.DataFrame(rows)


def categorical_diagnostics(frames: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for season in DISCOVERY_SEASONS:
        frame = frames[season]
        overall_brier = float(frame["squared_error"].mean())
        overall_residual = float(frame["signed_residual"].mean())
        for feature in CATEGORICAL_DIAGNOSTIC_FEATURES:
            labels = frame[feature].fillna("__NA__").astype(str).to_numpy()
            for row in grouped_metrics(frame, labels):
                mask = labels == row["bin"]
                complement = ~mask
                complement_brier = float(frame.loc[complement, "squared_error"].mean())
                complement_residual = float(
                    frame.loc[complement, "signed_residual"].mean()
                )
                complement_brier_se = float(
                    frame.loc[complement, "squared_error"].std(ddof=1)
                    / math.sqrt(complement.sum())
                )
                complement_residual_se = float(
                    frame.loc[complement, "signed_residual"].std(ddof=1)
                    / math.sqrt(complement.sum())
                )
                rows.append(
                    {
                        "validation_season": season,
                        "feature": feature,
                        "category": row.pop("bin"),
                        "brier_delta_vs_fold": row["champion_brier"] - overall_brier,
                        "residual_delta_vs_fold": row["signed_residual_mean"] - overall_residual,
                        "brier_delta_vs_complement": row["champion_brier"]
                        - complement_brier,
                        "residual_delta_vs_complement": row["signed_residual_mean"]
                        - complement_residual,
                        "brier_delta_stderr": math.sqrt(
                            row["brier_stderr"] ** 2 + complement_brier_se**2
                        ),
                        "residual_delta_stderr": math.sqrt(
                            row["signed_residual_stderr"] ** 2
                            + complement_residual_se**2
                        ),
                        **row,
                    }
                )
    return pd.DataFrame(rows)


def disagreement_diagnostics(frames: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    return univariate_diagnostics(frames, DISAGREEMENT_FEATURES)


def calibration_diagnostics(frames: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    fixed_edges = np.linspace(0.0, 1.0, 21)
    for season in DISCOVERY_SEASONS:
        frame = frames[season]
        history = pd.concat(
            [frames[source] for source in MODEL_TRAIN_SEASONS[season]], ignore_index=True
        )
        disagreement_low, disagreement_high = np.quantile(
            history["member_std"], [0.25, 0.75]
        )
        reliability = np.select(
            [frame["asof_pitcher_n"] < 1000, frame["asof_pitcher_n"] < 5000],
            ["low", "medium"],
            default="high",
        )
        subsets: list[tuple[str, str, np.ndarray]] = [("overall", "all", np.ones(len(frame), bool))]
        for value in ("R", "F"):
            subsets.append(("game_type", value, frame["game_type"].eq(value).to_numpy()))
        for value, flag in (("cold", 1), ("noncold", 0)):
            subsets.append(("cold_start", value, frame["cold_start"].eq(flag).to_numpy()))
        for value, flag in (("seen", 1), ("unseen", 0)):
            subsets.append(("pitcher_seen", value, frame["pitcher_seen"].eq(flag).to_numpy()))
        for value in ("low", "medium", "high"):
            subsets.append(("pitcher_reliability", value, reliability == value))
        subsets.extend(
            [
                (
                    "member_disagreement",
                    "low",
                    frame["member_std"].to_numpy() <= disagreement_low,
                ),
                (
                    "member_disagreement",
                    "high",
                    frame["member_std"].to_numpy() >= disagreement_high,
                ),
            ]
        )
        prediction = frame["champion_final"].to_numpy(dtype="float64")
        target = frame[TARGET].to_numpy(dtype="float64")
        prediction_bin = np.clip(np.searchsorted(fixed_edges, prediction, side="right") - 1, 0, 19)
        for subset_type, subset_value, subset_mask in subsets:
            for index in np.unique(prediction_bin[subset_mask]):
                mask = subset_mask & (prediction_bin == index)
                if not mask.any():
                    continue
                mean_prediction = float(prediction[mask].mean())
                positive_rate = float(target[mask].mean())
                rows.append(
                    {
                        "validation_season": season,
                        "subset_type": subset_type,
                        "subset_value": subset_value,
                        "prediction_bin": f"{fixed_edges[index]:.2f}-{fixed_edges[index + 1]:.2f}",
                        "n": int(mask.sum()),
                        "mean_prediction": mean_prediction,
                        "positive_rate": positive_rate,
                        "calibration_gap": positive_rate - mean_prediction,
                        "champion_brier": float(np.mean((prediction[mask] - target[mask]) ** 2)),
                        "disagreement_low_threshold": float(disagreement_low),
                        "disagreement_high_threshold": float(disagreement_high),
                    }
                )
    return pd.DataFrame(rows)


@dataclass
class MatrixEncoder:
    numeric_names: tuple[str, ...]
    categorical_names: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    missing_indices: np.ndarray
    categories: dict[str, tuple[str, ...]]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "MatrixEncoder":
        numeric_names = tuple(MODEL_NUMERIC_FEATURES)
        categorical_names = tuple(MODEL_CATEGORICAL_FEATURES)
        numeric = frame.loc[:, numeric_names].to_numpy(dtype="float64")
        with np.errstate(all="ignore"):
            medians = np.nanmedian(numeric, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        filled = np.where(np.isfinite(numeric), numeric, medians)
        means = filled.mean(axis=0)
        scales = filled.std(axis=0)
        scales = np.where(scales > 1e-12, scales, 1.0)
        missing_indices = np.flatnonzero((~np.isfinite(numeric)).any(axis=0))
        categories = {
            name: tuple(sorted(frame[name].fillna("__NA__").astype(str).unique()))
            for name in categorical_names
        }
        return cls(
            numeric_names,
            categorical_names,
            medians,
            means,
            scales,
            missing_indices,
            categories,
        )

    @property
    def feature_names(self) -> list[str]:
        names = list(self.numeric_names)
        names += [f"{self.numeric_names[index]}__missing" for index in self.missing_indices]
        for column in self.categorical_names:
            names += [f"{column}={category}" for category in self.categories[column]]
        return names

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = frame.loc[:, self.numeric_names].to_numpy(dtype="float64")
        missing = ~np.isfinite(numeric)
        filled = np.where(missing, self.medians, numeric)
        blocks = [np.clip((filled - self.means) / self.scales, -10.0, 10.0)]
        if len(self.missing_indices):
            blocks.append(missing[:, self.missing_indices].astype("float64"))
        for column in self.categorical_names:
            values = frame[column].fillna("__NA__").astype(str).to_numpy()
            blocks.append(
                np.column_stack([values == category for category in self.categories[column]])
            )
        return np.column_stack(blocks).astype("float32", copy=False)


def make_models() -> dict[str, object]:
    return {
        "ridge": Ridge(alpha=100.0, fit_intercept=True, solver="lsqr", tol=1e-6),
        "huber": HuberRegressor(
            epsilon=1.35,
            alpha=1e-3,
            max_iter=300,
            tol=1e-5,
            fit_intercept=True,
        ),
        "tree_depth2": DecisionTreeRegressor(
            max_depth=2,
            min_samples_leaf=20000,
            random_state=42,
        ),
    }


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def model_iterations(model: object) -> int:
    value = np.asarray(getattr(model, "n_iter_", 1)).reshape(-1)
    return int(value[0]) if len(value) else 1


def residual_models(
    frames: Mapping[int, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[int, str], np.ndarray]]:
    rows = []
    coefficients = []
    residual_predictions: dict[tuple[int, str], np.ndarray] = {}
    for season in DISCOVERY_SEASONS:
        train_seasons = MODEL_TRAIN_SEASONS[season]
        train = pd.concat([frames[value] for value in train_seasons], ignore_index=True)
        validation = frames[season]
        encoder = MatrixEncoder.fit(train)
        x_train = encoder.transform(train)
        x_validation = encoder.transform(validation)
        y_train = train["signed_residual"].to_numpy(dtype="float64")
        y_validation = validation["signed_residual"].to_numpy(dtype="float64")
        for model_name, model in make_models().items():
            model.fit(x_train, y_train)
            prediction = np.asarray(model.predict(x_validation), dtype="float64")
            residual_predictions[(season, model_name)] = prediction
            rows.append(
                {
                    "record_type": "predictability",
                    "model": model_name,
                    "validation_season": season,
                    "train_seasons": "+".join(map(str, train_seasons)),
                    "n": len(validation),
                    "residual_correlation": safe_corr(prediction, y_validation),
                    "residual_r2": float(r2_score(y_validation, prediction)),
                    "residual_mae": float(mean_absolute_error(y_validation, prediction)),
                    "predicted_residual_mean": float(prediction.mean()),
                    "predicted_residual_std": float(prediction.std()),
                    "model_iterations": model_iterations(model),
                    "model_converged": bool(
                        not isinstance(model, HuberRegressor)
                        or int(model.n_iter_) < int(model.max_iter)
                    ),
                    "gamma": None,
                    "cap": None,
                    "champion_brier": float(validation["squared_error"].mean()),
                    "candidate_brier": None,
                    "brier_gain": None,
                    "correction_mean_abs": None,
                    "correction_max_abs": None,
                    "correction_nonzero_rate": None,
                }
            )
            if hasattr(model, "coef_"):
                for feature, coefficient in zip(encoder.feature_names, model.coef_):
                    coefficients.append(
                        {
                            "model": model_name,
                            "validation_season": season,
                            "train_seasons": "+".join(map(str, train_seasons)),
                            "feature": feature,
                            "coefficient": float(coefficient),
                        }
                    )
            elif hasattr(model, "feature_importances_"):
                for feature, importance in zip(encoder.feature_names, model.feature_importances_):
                    coefficients.append(
                        {
                            "model": model_name,
                            "validation_season": season,
                            "train_seasons": "+".join(map(str, train_seasons)),
                            "feature": feature,
                            "coefficient": float(importance),
                        }
                    )
            champion = validation["champion_final"].to_numpy(dtype="float64")
            target = validation[TARGET].to_numpy(dtype="float64")
            for gamma, cap in CORRECTION_SETTINGS:
                correction = np.clip(gamma * prediction, -cap, cap)
                candidate = np.clip(champion + correction, 0.0, 1.0)
                champion_brier = float(np.mean((champion - target) ** 2))
                candidate_brier = float(np.mean((candidate - target) ** 2))
                rows.append(
                    {
                        "record_type": "candidate",
                        "model": model_name,
                        "validation_season": season,
                        "train_seasons": "+".join(map(str, train_seasons)),
                        "n": len(validation),
                        "residual_correlation": safe_corr(prediction, y_validation),
                        "residual_r2": float(r2_score(y_validation, prediction)),
                        "residual_mae": float(mean_absolute_error(y_validation, prediction)),
                        "predicted_residual_mean": float(prediction.mean()),
                        "predicted_residual_std": float(prediction.std()),
                        "model_iterations": model_iterations(model),
                        "model_converged": bool(
                            not isinstance(model, HuberRegressor)
                            or int(model.n_iter_) < int(model.max_iter)
                        ),
                        "gamma": gamma,
                        "cap": cap,
                        "champion_brier": champion_brier,
                        "candidate_brier": candidate_brier,
                        "brier_gain": champion_brier - candidate_brier,
                        "correction_mean_abs": float(np.mean(np.abs(correction))),
                        "correction_max_abs": float(np.max(np.abs(correction))),
                        "correction_nonzero_rate": float(np.mean(correction != 0.0)),
                    }
                )
        del x_train, x_validation
    return pd.DataFrame(rows), pd.DataFrame(coefficients), residual_predictions


def numeric_stability(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, body in table.groupby("feature"):
        season_deltas = {}
        residual_deltas = {}
        error_stderrs = {}
        residual_stderrs = {}
        for season, fold in body.groupby("validation_season"):
            usable = fold[fold["bin"].str.startswith("q")].copy()
            usable["index"] = usable["bin"].str[1:].astype(int)
            low = usable.loc[usable["index"].idxmin()]
            high = usable.loc[usable["index"].idxmax()]
            season_deltas[int(season)] = float(high["champion_brier"] - low["champion_brier"])
            residual_deltas[int(season)] = float(
                high["signed_residual_mean"] - low["signed_residual_mean"]
            )
            error_stderrs[int(season)] = math.sqrt(
                float(high["brier_stderr"]) ** 2 + float(low["brier_stderr"]) ** 2
            )
            residual_stderrs[int(season)] = math.sqrt(
                float(high["signed_residual_stderr"]) ** 2
                + float(low["signed_residual_stderr"]) ** 2
            )
        error_values = [season_deltas.get(season, np.nan) for season in DISCOVERY_SEASONS]
        residual_values = [residual_deltas.get(season, np.nan) for season in DISCOVERY_SEASONS]
        stable_error = all(np.isfinite(error_values)) and (
            all(value > 0 for value in error_values) or all(value < 0 for value in error_values)
        )
        stable_residual = all(np.isfinite(residual_values)) and (
            all(value > 0 for value in residual_values)
            or all(value < 0 for value in residual_values)
        )
        error_ratio = float(max(np.abs(error_values)) / max(min(np.abs(error_values)), 1e-12))
        residual_ratio = float(
            max(np.abs(residual_values)) / max(min(np.abs(residual_values)), 1e-12)
        )
        significant_error = all(
            abs(season_deltas[season]) >= 1.96 * error_stderrs[season]
            for season in DISCOVERY_SEASONS
        )
        significant_residual = all(
            abs(residual_deltas[season]) >= 1.96 * residual_stderrs[season]
            for season in DISCOVERY_SEASONS
        )
        rows.append(
            {
                "feature": feature,
                **{f"error_gradient_{season}": season_deltas.get(season) for season in DISCOVERY_SEASONS},
                **{
                    f"signed_residual_gradient_{season}": residual_deltas.get(season)
                    for season in DISCOVERY_SEASONS
                },
                "stable_error_direction": stable_error,
                "stable_signed_residual_direction": stable_residual,
                "significant_error_all_folds": significant_error,
                "significant_signed_residual_all_folds": significant_residual,
                "error_strength_ratio": error_ratio,
                "signed_residual_strength_ratio": residual_ratio,
                "robust_error_gradient": stable_error
                and significant_error
                and error_ratio <= 5.0,
                "robust_signed_residual_gradient": stable_residual
                and significant_residual
                and residual_ratio <= 5.0,
                "mean_abs_error_gradient": float(np.nanmean(np.abs(error_values))),
                "mean_abs_signed_residual_gradient": float(
                    np.nanmean(np.abs(residual_values))
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["robust_error_gradient", "mean_abs_error_gradient"], ascending=[False, False]
    )


def categorical_stability(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feature, category), body in table.groupby(["feature", "category"]):
        if set(body["validation_season"]) != set(DISCOVERY_SEASONS):
            continue
        ordered = body.sort_values("validation_season")
        if (ordered["n"] < 500).any():
            continue
        brier = ordered["brier_delta_vs_complement"].to_numpy(dtype="float64")
        residual = ordered["residual_delta_vs_complement"].to_numpy(dtype="float64")
        brier_se = ordered["brier_delta_stderr"].to_numpy(dtype="float64")
        residual_se = ordered["residual_delta_stderr"].to_numpy(dtype="float64")
        stable_brier = bool(np.all(brier > 0.0) or np.all(brier < 0.0))
        stable_residual = bool(np.all(residual > 0.0) or np.all(residual < 0.0))
        brier_significant = bool(np.all(np.abs(brier) >= 1.96 * brier_se))
        residual_significant = bool(np.all(np.abs(residual) >= 1.96 * residual_se))
        brier_ratio = float(np.max(np.abs(brier)) / max(np.min(np.abs(brier)), 1e-12))
        residual_ratio = float(
            np.max(np.abs(residual)) / max(np.min(np.abs(residual)), 1e-12)
        )
        rows.append(
            {
                "feature": feature,
                "category": category,
                **{
                    f"n_{season}": int(ordered.loc[ordered.validation_season.eq(season), "n"].iloc[0])
                    for season in DISCOVERY_SEASONS
                },
                **{
                    f"brier_delta_{season}": float(
                        ordered.loc[
                            ordered.validation_season.eq(season),
                            "brier_delta_vs_complement",
                        ].iloc[0]
                    )
                    for season in DISCOVERY_SEASONS
                },
                **{
                    f"residual_delta_{season}": float(
                        ordered.loc[
                            ordered.validation_season.eq(season),
                            "residual_delta_vs_complement",
                        ].iloc[0]
                    )
                    for season in DISCOVERY_SEASONS
                },
                "stable_brier_direction": stable_brier,
                "stable_residual_direction": stable_residual,
                "significant_brier_all_folds": brier_significant,
                "significant_residual_all_folds": residual_significant,
                "brier_strength_ratio": brier_ratio,
                "residual_strength_ratio": residual_ratio,
                "robust_brier_pattern": stable_brier
                and brier_significant
                and brier_ratio <= 5.0,
                "robust_residual_pattern": stable_residual
                and residual_significant
                and residual_ratio <= 5.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["robust_residual_pattern", "robust_brier_pattern"], ascending=[False, False]
    )


def coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    linear = coefficients[coefficients["model"].isin(["ridge", "huber"])]
    rows = []
    for (model, feature), body in linear.groupby(["model", "feature"]):
        values = {
            int(row.validation_season): float(row.coefficient)
            for row in body.itertuples()
        }
        ordered = [values.get(season, 0.0) for season in DISCOVERY_SEASONS]
        nonzero = [value for value in ordered if abs(value) > 1e-8]
        stable_sign = len(nonzero) == len(DISCOVERY_SEASONS) and (
            all(value > 0 for value in nonzero) or all(value < 0 for value in nonzero)
        )
        rows.append(
            {
                "model": model,
                "feature": feature,
                **{f"coefficient_{season}": values.get(season, 0.0) for season in DISCOVERY_SEASONS},
                "stable_sign": stable_sign,
                "mean_abs_coefficient": float(np.mean(np.abs(ordered))),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["stable_sign", "mean_abs_coefficient"], ascending=[False, False]
    )


def candidate_summary(results: pd.DataFrame) -> pd.DataFrame:
    candidates = results[results["record_type"] == "candidate"].copy()
    rows = []
    for (model, gamma, cap), body in candidates.groupby(["model", "gamma", "cap"]):
        by_season = {int(row.validation_season): row for row in body.itertuples()}
        gains = [float(by_season[season].brier_gain) for season in DISCOVERY_SEASONS]
        counts = [int(by_season[season].n) for season in DISCOVERY_SEASONS]
        pooled = float(np.average(gains, weights=counts))
        passed = (
            gains[2] > 0.0
            and pooled > 0.0
            and gains[1] > 0.0
            and gains[0] >= -1e-5
        )
        rows.append(
            {
                "model": model,
                "gamma": gamma,
                "cap": cap,
                **{f"brier_gain_{season}": gain for season, gain in zip(DISCOVERY_SEASONS, gains)},
                "pooled_brier_gain": pooled,
                "worst_brier_gain": min(gains),
                "mean_abs_correction": float(
                    np.average(body["correction_mean_abs"], weights=body["n"])
                ),
                "verdict": "pass" if passed else "reject",
            }
        )
    return pd.DataFrame(rows).sort_values("pooled_brier_gain", ascending=False)


def predictability_summary(results: pd.DataFrame) -> pd.DataFrame:
    body = results[results["record_type"] == "predictability"]
    rows = []
    for model, group in body.groupby("model"):
        by_season = {int(row.validation_season): row for row in group.itertuples()}
        counts = [int(by_season[season].n) for season in DISCOVERY_SEASONS]
        rows.append(
            {
                "model": model,
                **{
                    f"residual_corr_{season}": float(by_season[season].residual_correlation)
                    for season in DISCOVERY_SEASONS
                },
                **{
                    f"residual_r2_{season}": float(by_season[season].residual_r2)
                    for season in DISCOVERY_SEASONS
                },
                "pooled_residual_corr": float(
                    np.average(
                        [by_season[season].residual_correlation for season in DISCOVERY_SEASONS],
                        weights=counts,
                    )
                ),
                "pooled_residual_r2": float(
                    np.average(
                        [by_season[season].residual_r2 for season in DISCOVERY_SEASONS],
                        weights=counts,
                    )
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("pooled_residual_corr", ascending=False)


def stable_features_markdown(
    numeric: pd.DataFrame,
    categorical: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> str:
    stable_numeric = numeric[
        numeric["robust_error_gradient"] | numeric["robust_signed_residual_gradient"]
    ].head(20)
    stable_categorical = categorical[
        categorical["robust_brier_pattern"] | categorical["robust_residual_pattern"]
    ].head(30)
    stable_coefficients = coefficients[coefficients["stable_sign"]].groupby("feature").filter(
        lambda body: set(body["model"]) == {"ridge", "huber"}
    )
    stable_coefficients = stable_coefficients.sort_values(
        "mean_abs_coefficient", ascending=False
    ).head(30)
    lines = [
        "# Stable residual features",
        "",
        "This is a discovery diagnostic, not a correction specification.",
        "",
        "## Robust univariate gradients",
        "",
        "Positive means the high quantile has larger Brier error than the low quantile.",
        "",
        "| feature | error 2022 | error 2023 | error 2024 | residual 2022 | residual 2023 | residual 2024 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in stable_numeric.itertuples():
        lines.append(
            f"| {row.feature} | {row.error_gradient_2022:+.3e} "
            f"| {row.error_gradient_2023:+.3e} | {row.error_gradient_2024:+.3e} "
            f"| {row.signed_residual_gradient_2022:+.3e} "
            f"| {row.signed_residual_gradient_2023:+.3e} "
            f"| {row.signed_residual_gradient_2024:+.3e} |"
        )
    lines.extend(
        [
            "",
            "## Robust categorical patterns",
            "",
            "Deltas compare the category with its same-fold complement.",
            "",
            "| feature | category | n 2022/2023/2024 | Brier delta 2022/2023/2024 | residual delta 2022/2023/2024 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in stable_categorical.itertuples():
        lines.append(
            f"| {row.feature} | {row.category} "
            f"| {row.n_2022:,}/{row.n_2023:,}/{row.n_2024:,} "
            f"| {row.brier_delta_2022:+.3e}/{row.brier_delta_2023:+.3e}/{row.brier_delta_2024:+.3e} "
            f"| {row.residual_delta_2022:+.3e}/{row.residual_delta_2023:+.3e}/{row.residual_delta_2024:+.3e} |"
        )
    lines.extend(
        [
            "",
            "## Linear diagnostic coefficients with stable signs",
            "",
            "Coefficients are on train-standardized features and are only comparable within a model.",
            "",
            "| model | feature | 2022 | 2023 | 2024 | mean absolute coefficient |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stable_coefficients.itertuples():
        lines.append(
            f"| {row.model} | {row.feature} | {row.coefficient_2022:+.3e} "
            f"| {row.coefficient_2023:+.3e} | {row.coefficient_2024:+.3e} "
            f"| {row.mean_abs_coefficient:.3e} |"
        )
    return "\n".join(lines) + "\n"


def write_summary_json(
    output: Path,
    univariate_stability: pd.DataFrame,
    category_stability: pd.DataFrame,
    disagreement_stability: pd.DataFrame,
    predictability: pd.DataFrame,
    candidates: pd.DataFrame,
    champion_before: str,
    champion_after: str,
) -> dict:
    stable = univariate_stability[
        univariate_stability["robust_error_gradient"]
        | univariate_stability["robust_signed_residual_gradient"]
    ]
    stable_categories = category_stability[
        category_stability["robust_brier_pattern"]
        | category_stability["robust_residual_pattern"]
    ]
    stable_disagreement = disagreement_stability[
        disagreement_stability["robust_error_gradient"]
        | disagreement_stability["robust_signed_residual_gradient"]
    ]
    passing = candidates[candidates["verdict"] == "pass"]
    payload = {
        "champion_lb_bss": 1016.4442212358,
        "champion_sha256_before": champion_before,
        "champion_sha256_after": champion_after,
        "test_rows_read": False,
        "validation_labels_used_for_fit": False,
        "model_train_seasons": {
            str(key): list(value) for key, value in MODEL_TRAIN_SEASONS.items()
        },
        "stable_univariate_features": stable.head(20).to_dict(orient="records"),
        "stable_categorical_features": stable_categories.head(30).to_dict(
            orient="records"
        ),
        "stable_disagreement_features": stable_disagreement.to_dict(orient="records"),
        "predictability": predictability.to_dict(orient="records"),
        "candidate_corrections": candidates.to_dict(orient="records"),
        "passing_candidates": passing.to_dict(orient="records"),
        "decision": (
            "A. stable residual signals found; no residual model passed and production remains unchanged"
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def read_train(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {TARGET, "season", "row_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"train is missing columns: {missing}")
    return frame


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("/Users/wooh/Documents/dev/open/data/train.csv"),
    )
    parser.add_argument(
        "--oof-dir",
        type=Path,
        default=ROOT / "artifacts" / "regime_blend_oof",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "residual_discovery",
    )
    args = parser.parse_args(argv)
    if not args.train.is_file():
        parser.error(f"train not found: {args.train}")
    if not CHAMPION_ZIP.is_file():
        parser.error(f"champion not found: {CHAMPION_ZIP}")
    champion_before = sha256(CHAMPION_ZIP)
    if champion_before != EXPECTED_CHAMPION_SHA256:
        raise SystemExit(f"champion checksum mismatch before experiment: {champion_before}")

    print(f"load official train: {args.train}", flush=True)
    train = read_train(args.train)
    frames = {}
    for season in SOURCE_SEASONS:
        path = args.oof_dir / "cache" / f"season_{season}.npz"
        if not path.is_file():
            parser.error(f"OOF cache not found: {path}")
        oof = load_npz(path)
        train_season = train[train["season"].eq(season)]
        frames[season] = build_discovery_frame(train_season, oof)
        print(f"aligned season {season}: {len(frames[season]):,} rows", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    print("univariate residual diagnostics", flush=True)
    univariate = univariate_diagnostics(frames, UNIVARIATE_NUMERIC_FEATURES)
    categories = categorical_diagnostics(frames)
    disagreement = disagreement_diagnostics(frames)
    calibration = calibration_diagnostics(frames)
    univariate.to_csv(args.output / "residual_univariate.csv", index=False)
    categories.to_csv(args.output / "residual_categories.csv", index=False)
    disagreement.to_csv(args.output / "disagreement_diagnostics.csv", index=False)
    calibration.to_csv(args.output / "calibration_diagnostics.csv", index=False)

    print("nested temporal residual models", flush=True)
    model_results, coefficients, _ = residual_models(frames)
    model_results.to_csv(args.output / "residual_model_results.csv", index=False)
    coefficients.to_csv(args.output / "residual_model_coefficients.csv", index=False)
    univariate_stability = numeric_stability(univariate)
    disagreement_stability = numeric_stability(disagreement)
    category_stability = categorical_stability(categories)
    coefficient_summary = coefficient_stability(coefficients)
    candidate_results = candidate_summary(model_results)
    predictability = predictability_summary(model_results)
    univariate_stability.to_csv(args.output / "residual_feature_stability.csv", index=False)
    disagreement_stability.to_csv(
        args.output / "disagreement_stability.csv", index=False
    )
    category_stability.to_csv(
        args.output / "residual_category_stability.csv", index=False
    )
    coefficient_summary.to_csv(
        args.output / "residual_coefficient_stability.csv", index=False
    )
    candidate_results.to_csv(args.output / "candidate_corrections.csv", index=False)
    predictability.to_csv(args.output / "residual_predictability.csv", index=False)
    (args.output / "stable_residual_features.md").write_text(
        stable_features_markdown(
            univariate_stability, category_stability, coefficient_summary
        ),
        encoding="utf-8",
    )

    champion_after = sha256(CHAMPION_ZIP)
    summary = write_summary_json(
        args.output,
        univariate_stability,
        category_stability,
        disagreement_stability,
        predictability,
        candidate_results,
        champion_before,
        champion_after,
    )
    print(f"saved residual discovery artifacts to {args.output}", flush=True)
    print(f"passing correction candidates: {len(summary['passing_candidates'])}", flush=True)
    print(f"champion checksum unchanged: {champion_before == champion_after}", flush=True)


if __name__ == "__main__":
    main()

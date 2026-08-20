"""Analyze whether 2024 is a distinct conditional target regime.

This is a train-only Stage A diagnostic.  It never reads evaluation rows and
never builds a production package.  The 2024 boundary is a preregistered
domain hypothesis; effect direction and magnitude come only from train data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = Path("/Users/wooh/Documents/dev/open/data/train.csv")
OUTPUT_DIR = ROOT / "artifacts" / "rule_regime"
OOF_DIR = ROOT / "artifacts" / "regime_blend_oof" / "cache"
ET25_OOF_DIR = ROOT / "artifacts" / "extratrees_size_ablation" / "cache"

CURRENT_CHAMPION = ROOT / "artifacts" / "sub_et25_w020.zip"
CURRENT_CHAMPION_SHA256 = (
    "316bf6dab11fb78f06cc74089ea38bdb3b957355f4510db5fcc8a485dc26cecf"
)
CURRENT_CHAMPION_LB_BSS = 1017.0233029621
CURRENT_ET_WEIGHT = 0.0200

TARGET = "control_success"
BOUNDARIES = ((2021, 2022), (2022, 2023), (2023, 2024))
ANALYSIS_SEASONS = (2021, 2022, 2023, 2024)
CHAMPION_OOF_SEASONS = (2022, 2023, 2024)
MIN_CELL_N = 500
RANK_MIN_CELL_N = 1_000
SHRINKAGE_TAU = 1_000.0

BASE_COLUMNS = (
    "row_id",
    "season",
    "game_month",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "asof_pitcher_n",
    TARGET,
)

CONTEXT_COLUMNS = {
    "count_state": "count_state",
    "pitcher_hand": "pitcher_hand_label",
    "hand_matchup": "hand_matchup",
    "game_type": "game_type_label",
    "game_type_x_count": "game_type_x_count",
    "base_state": "base_state_label",
    "inning_bucket": "inning_bucket",
    "outs_before": "outs_before_label",
    "top_bottom": "top_bottom_label",
    "base_state_x_count": "base_state_x_count",
    "inning_bucket_x_count": "inning_bucket_x_count",
    "outs_before_x_count": "outs_before_x_count",
    "pitcher_reliability": "pitcher_reliability",
    "pitcher_reliability_x_count": "pitcher_reliability_x_count",
    "pitcher_form": "pitcher_form",
    "pitcher_form_x_count": "pitcher_form_x_count",
}

CORE_INTERACTION_FAMILIES = (
    "count_state",
    "hand_matchup",
    "game_type_x_count",
    "base_state_x_count",
    "inning_bucket_x_count",
    "outs_before_x_count",
    "pitcher_reliability_x_count",
    "pitcher_form_x_count",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="float64"))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(dict(payload)), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def safe_corr(left: Iterable[float], right: Iterable[float]) -> float:
    x = np.asarray(list(left), dtype="float64")
    y = np.asarray(list(right), dtype="float64")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(left: Iterable[float], right: Iterable[float]) -> float:
    x = pd.Series(list(left), dtype="float64")
    y = pd.Series(list(right), dtype="float64")
    valid = x.notna() & y.notna()
    if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return float("nan")
    return float(x[valid].corr(y[valid], method="spearman"))


def harmonic_sample_size(n_from: np.ndarray, n_to: np.ndarray) -> np.ndarray:
    left = np.asarray(n_from, dtype="float64")
    right = np.asarray(n_to, dtype="float64")
    return 2.0 / (1.0 / left + 1.0 / right)


def shrink_delta(
    raw_delta: np.ndarray,
    n_from: np.ndarray,
    n_to: np.ndarray,
    parent_delta: float,
    tau: float = SHRINKAGE_TAU,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shrink cell deltas toward the boundary-wide rate shift."""
    n_effective = harmonic_sample_size(n_from, n_to)
    factor = n_effective / (n_effective + float(tau))
    shrunk = float(parent_delta) + factor * (
        np.asarray(raw_delta, dtype="float64") - float(parent_delta)
    )
    return shrunk, factor, n_effective


def current_champion_prediction(
    original_champion: np.ndarray,
    et25_prediction: np.ndarray,
    weight: float = CURRENT_ET_WEIGHT,
) -> np.ndarray:
    return (1.0 - float(weight)) * np.asarray(
        original_champion, dtype="float64"
    ) + float(weight) * np.asarray(et25_prediction, dtype="float64")


def pitcher_form_offset(source: Mapping[str, np.ndarray]) -> np.ndarray:
    """Recover the signed pitcher-form layer from its stored endpoint."""
    return np.asarray(source["pitcher_form_pred"], dtype="float64") - np.asarray(
        source["champion_base"], dtype="float64"
    )


def add_static_contexts(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["count_state"] = (
        out["balls_before"].astype(str) + "-" + out["strikes_before"].astype(str)
    )
    out["pitcher_hand_label"] = "H" + out["pitcher_hand"].astype(str)
    out["batter_hand_label"] = "H" + out["batter_hand"].astype(str)
    out["hand_matchup"] = (
        out["pitcher_hand_label"] + "_vs_" + out["batter_hand_label"]
    )
    out["game_type_label"] = out["game_type"].astype(str)
    out["base_state_label"] = out["base_state"].astype(str)
    out["top_bottom_label"] = out["top_bottom"].astype(str)
    out["outs_before_label"] = "outs_" + out["outs_before"].astype(str)
    out["inning_bucket"] = pd.cut(
        out["inning"],
        bins=[0, 3, 6, 9, np.inf],
        labels=["early_1_3", "middle_4_6", "late_7_9", "extras_10_plus"],
        right=True,
    ).astype("object")
    out["game_type_x_count"] = out["game_type_label"] + "|" + out["count_state"]
    out["base_state_x_count"] = out["base_state_label"] + "|" + out["count_state"]
    out["inning_bucket_x_count"] = (
        out["inning_bucket"].astype(str) + "|" + out["count_state"]
    )
    out["outs_before_x_count"] = out["outs_before_label"] + "|" + out["count_state"]
    return out


def attach_oof_contexts(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["pitcher_seen"] = False
    out["pitcher_form_value"] = np.nan
    for season in ANALYSIS_SEASONS:
        mask = out["season"].eq(season).to_numpy()
        rows = out.loc[mask]
        source = load_npz(OOF_DIR / f"season_{season}.npz")
        target = rows[TARGET].to_numpy(dtype="float64")
        if not np.array_equal(source["y"].astype("float64"), target):
            raise ValueError(f"OOF target order mismatch for {season}")
        history = rows["asof_pitcher_n"].to_numpy(dtype="int64")
        if not np.array_equal(source["x_asof_pitcher_n"].astype("int64"), history):
            raise ValueError(f"OOF history order mismatch for {season}")
        out.loc[mask, "pitcher_seen"] = source["x_pitcher_seen"].astype(bool)
        out.loc[mask, "pitcher_form_value"] = pitcher_form_offset(source)

    seen = out["pitcher_seen"].to_numpy(dtype=bool)
    history = out["asof_pitcher_n"].to_numpy(dtype="float64")
    out["pitcher_reliability"] = np.select(
        [~seen, seen & (history < RANK_MIN_CELL_N)],
        ["unseen", "low_history"],
        default="established",
    )
    form = out["pitcher_form_value"].to_numpy(dtype="float64")
    out["pitcher_form"] = np.select(
        [form > 1e-12, form < -1e-12],
        ["positive", "negative"],
        default="neutral",
    )
    out["pitcher_reliability_x_count"] = (
        out["pitcher_reliability"] + "|" + out["count_state"]
    )
    out["pitcher_form_x_count"] = out["pitcher_form"] + "|" + out["count_state"]
    return out


def season_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby("season", sort=True)
        .agg(
            rows=(TARGET, "size"),
            control_success_rate=(TARGET, "mean"),
            pitcher_count=("pitcher_id", "nunique"),
            batter_count=("batter_id", "nunique"),
        )
        .reset_index()
    )
    summary["game_count"] = np.nan
    summary["game_count_status"] = (
        "unavailable: train.csv has no game_id or game_date column"
    )
    summary["regime_label"] = np.where(
        summary["season"].eq(2024), "2024", "PRE_2024"
    )
    return summary


def grouped_season_stats(
    frame: pd.DataFrame, family: str, context_column: str
) -> pd.DataFrame:
    if family.startswith("pitcher_reliability") or family.startswith("pitcher_form"):
        rows = frame[frame["season"].isin(ANALYSIS_SEASONS)].copy()
    else:
        rows = frame.copy()
    rows = rows[rows[context_column].notna()]
    rows["context"] = rows[context_column].astype(str)
    season_total = rows.groupby("season")[TARGET].size().rename("season_rows")
    season_rate = rows.groupby("season")[TARGET].mean().rename("season_rate")
    result = (
        rows.groupby(["season", "context"], sort=True)[TARGET]
        .agg(n="size", rate="mean")
        .reset_index()
        .join(season_total, on="season")
        .join(season_rate, on="season")
    )
    result.insert(0, "family", family)
    result["frequency"] = result["n"] / result["season_rows"]
    result["brier_cell_constant"] = result["rate"] * (1.0 - result["rate"])
    result["brier_season_baseline"] = (
        result["rate"] * (1.0 - result["season_rate"]) ** 2
        + (1.0 - result["rate"]) * result["season_rate"] ** 2
    )
    return result


def special_count_stats(frame: pd.DataFrame) -> pd.DataFrame:
    masks = {
        "full_count_3-2": frame["balls_before"].eq(3) & frame["strikes_before"].eq(2),
        "all_3-ball": frame["balls_before"].eq(3),
        "all_2-strike": frame["strikes_before"].eq(2),
    }
    rows = []
    for season in sorted(frame["season"].unique()):
        season_rows = frame[frame["season"].eq(season)]
        season_rate = float(season_rows[TARGET].mean())
        for context, full_mask in masks.items():
            selected = frame[frame["season"].eq(season) & full_mask]
            rate = float(selected[TARGET].mean())
            rows.append(
                {
                    "family": "count_special",
                    "season": season,
                    "context": context,
                    "n": len(selected),
                    "rate": rate,
                    "season_rows": len(season_rows),
                    "season_rate": season_rate,
                    "frequency": len(selected) / len(season_rows),
                    "brier_cell_constant": rate * (1.0 - rate),
                    "brier_season_baseline": rate * (1.0 - season_rate) ** 2
                    + (1.0 - rate) * season_rate**2,
                }
            )
    return pd.DataFrame(rows)


def boundary_shifts(season_stats: pd.DataFrame, global_rates: Mapping[int, float]) -> pd.DataFrame:
    outputs = []
    for family, family_rows in season_stats.groupby("family", sort=True):
        for from_season, to_season in BOUNDARIES:
            previous = family_rows[family_rows["season"].eq(from_season)].copy()
            current = family_rows[family_rows["season"].eq(to_season)].copy()
            merged = previous.merge(
                current,
                on="context",
                how="inner",
                suffixes=("_from", "_to"),
                validate="one_to_one",
            )
            if merged.empty:
                continue
            raw_delta = merged["rate_to"] - merged["rate_from"]
            standard_error = np.sqrt(
                merged["rate_from"] * (1.0 - merged["rate_from"]) / merged["n_from"]
                + merged["rate_to"] * (1.0 - merged["rate_to"]) / merged["n_to"]
            )
            parent_delta = float(global_rates[to_season] - global_rates[from_season])
            shrunk, factor, n_effective = shrink_delta(
                raw_delta.to_numpy(),
                merged["n_from"].to_numpy(),
                merged["n_to"].to_numpy(),
                parent_delta,
            )
            output = pd.DataFrame(
                {
                    "family": family,
                    "context": merged["context"],
                    "from_season": from_season,
                    "to_season": to_season,
                    "boundary": f"{from_season}_to_{to_season}",
                    "n_from": merged["n_from"],
                    "n_to": merged["n_to"],
                    "frequency_from": merged["frequency_from"],
                    "frequency_to": merged["frequency_to"],
                    "frequency_delta": merged["frequency_to"]
                    - merged["frequency_from"],
                    "rate_from": merged["rate_from"],
                    "rate_to": merged["rate_to"],
                    "raw_delta": raw_delta,
                    "standard_error": standard_error,
                    "ci95_low": raw_delta - 1.96 * standard_error,
                    "ci95_high": raw_delta + 1.96 * standard_error,
                    "global_delta": parent_delta,
                    "conditional_delta_vs_global": raw_delta - parent_delta,
                    "n_effective": n_effective,
                    "shrinkage_factor": factor,
                    "shrunk_delta": shrunk,
                }
            )
            output["eligible"] = (
                (output["n_from"] >= MIN_CELL_N) & (output["n_to"] >= MIN_CELL_N)
            )
            output["small_cell"] = ~output["eligible"]
            output["raw_significant_95"] = (output["ci95_low"] > 0.0) | (
                output["ci95_high"] < 0.0
            )
            outputs.append(output)
    if not outputs:
        raise ValueError("no boundary shifts were generated")
    return pd.concat(outputs, ignore_index=True)


def frequency_shift_table(conditional: pd.DataFrame) -> pd.DataFrame:
    output = conditional[
        [
            "family",
            "context",
            "from_season",
            "to_season",
            "boundary",
            "n_from",
            "n_to",
            "frequency_from",
            "frequency_to",
            "frequency_delta",
            "eligible",
            "small_cell",
        ]
    ].copy()
    output["frequency_ratio"] = np.divide(
        output["frequency_to"],
        output["frequency_from"],
        out=np.full(len(output), np.nan),
        where=output["frequency_from"].to_numpy() > 0,
    )
    return output


def load_current_champion_oof(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    predictions = {}
    for season in CHAMPION_OOF_SEASONS:
        rows = frame[frame["season"].eq(season)]
        target = rows[TARGET].to_numpy(dtype="float64")
        source = load_npz(OOF_DIR / f"season_{season}.npz")
        prefix = load_npz(ET25_OOF_DIR / f"season_{season}_prefixes.npz")
        if not np.array_equal(source["y"].astype("float64"), target):
            raise ValueError(f"champion OOF target order mismatch for {season}")
        expected_hash = str(prefix["target_sha256"].item())
        if array_sha256(target) != expected_hash:
            raise ValueError(f"ET25 OOF target checksum mismatch for {season}")
        predictions[season] = current_champion_prediction(
            source["champion_final"], prefix["prediction_25"]
        )
    return predictions


def champion_residual_table(
    frame: pd.DataFrame, predictions: Mapping[int, np.ndarray]
) -> pd.DataFrame:
    outputs = []
    for season, prediction in predictions.items():
        rows = frame[frame["season"].eq(season)].copy()
        if len(rows) != len(prediction):
            raise ValueError(f"champion prediction length mismatch for {season}")
        rows["champion_prediction"] = np.asarray(prediction, dtype="float64")
        rows["champion_residual"] = rows[TARGET] - rows["champion_prediction"]
        rows["champion_squared_error"] = rows["champion_residual"] ** 2
        for family, column in CONTEXT_COLUMNS.items():
            part = rows[rows[column].notna()].copy()
            part["context"] = part[column].astype(str)
            grouped = (
                part.groupby("context", sort=True)
                .agg(
                    n=(TARGET, "size"),
                    actual_rate=(TARGET, "mean"),
                    mean_prediction=("champion_prediction", "mean"),
                    mean_y_minus_prediction=("champion_residual", "mean"),
                    brier=("champion_squared_error", "mean"),
                )
                .reset_index()
            )
            grouped.insert(0, "family", family)
            grouped.insert(0, "season", season)
            grouped["calibration_gap_prediction_minus_actual"] = (
                grouped["mean_prediction"] - grouped["actual_rate"]
            )
            outputs.append(grouped)
        # The overlapping special count contexts need direct masks.
        for context, mask in {
            "full_count_3-2": rows["balls_before"].eq(3)
            & rows["strikes_before"].eq(2),
            "all_3-ball": rows["balls_before"].eq(3),
            "all_2-strike": rows["strikes_before"].eq(2),
        }.items():
            selected = rows[mask]
            outputs.append(
                pd.DataFrame(
                    [
                        {
                            "season": season,
                            "family": "count_special",
                            "context": context,
                            "n": len(selected),
                            "actual_rate": selected[TARGET].mean(),
                            "mean_prediction": selected["champion_prediction"].mean(),
                            "mean_y_minus_prediction": selected[
                                "champion_residual"
                            ].mean(),
                            "brier": selected["champion_squared_error"].mean(),
                            "calibration_gap_prediction_minus_actual": selected[
                                "champion_prediction"
                            ].mean()
                            - selected[TARGET].mean(),
                        }
                    ]
                )
            )
    return pd.concat(outputs, ignore_index=True)


def placebo_summary(
    conditional: pd.DataFrame, residual: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    residual_2024 = residual[residual["season"].eq(2024)][
        ["family", "context", "mean_y_minus_prediction"]
    ]
    for (family, boundary), group in conditional.groupby(
        ["family", "boundary"], sort=True
    ):
        eligible = group[group["eligible"]].copy()
        if eligible.empty:
            continue
        weights = eligible["n_effective"].to_numpy(dtype="float64")
        conditional_delta = eligible["conditional_delta_vs_global"].to_numpy(
            dtype="float64"
        )
        raw_delta = eligible["raw_delta"].to_numpy(dtype="float64")
        aligned = float("nan")
        if boundary == "2023_to_2024":
            joined = eligible.merge(
                residual_2024,
                on=["family", "context"],
                how="inner",
                validate="one_to_one",
            )
            if not joined.empty:
                alignment = np.sign(joined["raw_delta"]) == np.sign(
                    joined["mean_y_minus_prediction"]
                )
                aligned = float(
                    np.average(alignment.astype("float64"), weights=joined["n_effective"])
                )
        rows.append(
            {
                "family": family,
                "boundary": boundary,
                "from_season": int(eligible["from_season"].iloc[0]),
                "to_season": int(eligible["to_season"].iloc[0]),
                "n_eligible_cells": len(eligible),
                "weighted_mean_abs_raw_delta": float(
                    np.average(np.abs(raw_delta), weights=weights)
                ),
                "weighted_mean_abs_conditional_delta": float(
                    np.average(np.abs(conditional_delta), weights=weights)
                ),
                "weighted_rms_raw_delta": float(
                    np.sqrt(np.average(raw_delta**2, weights=weights))
                ),
                "significant_cell_fraction": float(
                    eligible["raw_significant_95"].mean()
                ),
                "champion_residual_direction_alignment": aligned,
            }
        )
    result = pd.DataFrame(rows)
    result["distinct_ratio_vs_max_placebo"] = np.nan
    for family, family_rows in result.groupby("family"):
        placebo = family_rows[family_rows["to_season"].lt(2024)]
        current_index = family_rows[family_rows["to_season"].eq(2024)].index
        if placebo.empty or len(current_index) != 1:
            continue
        denominator = max(
            float(placebo["weighted_mean_abs_conditional_delta"].max()), 1e-12
        )
        result.loc[current_index, "distinct_ratio_vs_max_placebo"] = (
            result.loc[current_index, "weighted_mean_abs_conditional_delta"]
            / denominator
        )
    return result


def persistence_table(conditional: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = (
        ("2021_to_2022", "2022_to_2023"),
        ("2022_to_2023", "2023_to_2024"),
    )
    for family, family_rows in conditional.groupby("family", sort=True):
        for earlier, later in pairs:
            left = family_rows[
                family_rows["boundary"].eq(earlier) & family_rows["eligible"]
            ][["context", "raw_delta", "shrunk_delta"]]
            right = family_rows[
                family_rows["boundary"].eq(later) & family_rows["eligible"]
            ][["context", "raw_delta", "shrunk_delta"]]
            joined = left.merge(
                right,
                on="context",
                suffixes=("_earlier", "_later"),
                how="inner",
                validate="one_to_one",
            )
            rows.append(
                {
                    "family": family,
                    "earlier_boundary": earlier,
                    "later_boundary": later,
                    "n_common_cells": len(joined),
                    "pearson_raw_delta": safe_corr(
                        joined["raw_delta_earlier"], joined["raw_delta_later"]
                    ),
                    "spearman_raw_delta": safe_spearman(
                        joined["raw_delta_earlier"], joined["raw_delta_later"]
                    ),
                    "pearson_shrunk_delta": safe_corr(
                        joined["shrunk_delta_earlier"], joined["shrunk_delta_later"]
                    ),
                    "sign_agreement": float(
                        (
                            np.sign(joined["shrunk_delta_earlier"])
                            == np.sign(joined["shrunk_delta_later"])
                        ).mean()
                    )
                    if len(joined)
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def add_parent_consistency(current: pd.DataFrame) -> pd.DataFrame:
    out = current.copy()
    lookup = {
        (row.family, row.context): row.raw_delta
        for row in current.itertuples(index=False)
    }
    consistent = []
    for row in out.itertuples(index=False):
        parents: list[tuple[str, str]] = []
        if row.family.endswith("_x_count"):
            prefix, count = row.context.rsplit("|", 1)
            parent_family = row.family.removesuffix("_x_count")
            parents.extend([(parent_family, prefix), ("count_state", count)])
        elif row.family == "hand_matchup":
            parents.append(("pitcher_hand", row.context.split("_vs_")[0]))
        parent_deltas = [lookup[key] for key in parents if key in lookup]
        consistent.append(
            bool(parent_deltas)
            and all(np.sign(value) == np.sign(row.raw_delta) for value in parent_deltas)
        )
    out["related_context_direction_consistent"] = consistent
    return out


def rank_structural_contexts(
    conditional: pd.DataFrame, residual: pd.DataFrame
) -> pd.DataFrame:
    current = conditional[
        conditional["boundary"].eq("2023_to_2024")
        & (conditional["n_from"] >= RANK_MIN_CELL_N)
        & (conditional["n_to"] >= RANK_MIN_CELL_N)
    ].copy()
    current = add_parent_consistency(current)
    current_residual = residual[residual["season"].eq(2024)][
        ["family", "context", "mean_y_minus_prediction", "brier"]
    ]
    current = current.merge(
        current_residual,
        on=["family", "context"],
        how="left",
        validate="one_to_one",
    )
    previous = conditional[conditional["to_season"].lt(2024)].copy()
    placebo = (
        previous.assign(abs_delta=previous["raw_delta"].abs())
        .groupby(["family", "context"])["abs_delta"]
        .max()
        .rename("max_abs_placebo_delta")
        .reset_index()
    )
    current = current.merge(
        placebo, on=["family", "context"], how="left", validate="one_to_one"
    )
    current["distinct_ratio_vs_cell_placebo"] = current["raw_delta"].abs() / np.maximum(
        current["max_abs_placebo_delta"].fillna(0.0), 1e-6
    )
    current["champion_residual_direction_aligned"] = np.sign(
        current["raw_delta"]
    ) == np.sign(current["mean_y_minus_prediction"])
    size_factor = np.minimum(np.sqrt(current["n_effective"] / RANK_MIN_CELL_N), 5.0)
    current["ranking_score"] = (
        current["shrunk_delta"].abs()
        * np.minimum(current["distinct_ratio_vs_cell_placebo"], 10.0)
        * size_factor
        * np.where(current["champion_residual_direction_aligned"], 1.0, 0.25)
        * np.where(current["related_context_direction_consistent"], 1.0, 0.5)
    )
    current = current.sort_values(
        ["ranking_score", "n_effective"], ascending=[False, False]
    ).reset_index(drop=True)
    current.insert(0, "structural_rank", np.arange(1, len(current) + 1))
    return current


def stage_a_decision(
    placebo: pd.DataFrame, persistence: pd.DataFrame
) -> tuple[str, list[dict], list[dict], list[dict]]:
    current = placebo[
        placebo["boundary"].eq("2023_to_2024")
        & placebo["family"].isin(CORE_INTERACTION_FAMILIES)
    ]
    latest_persistence = persistence[
        persistence["later_boundary"].eq("2023_to_2024")
    ].set_index("family")
    strong_structural = []
    localized = []
    persistent_strong = []
    for row in current.itertuples(index=False):
        is_localized = bool(
            row.n_eligible_cells >= 4
            and row.weighted_mean_abs_conditional_delta >= 0.004
            and row.distinct_ratio_vs_max_placebo >= 1.25
            and row.significant_cell_fraction >= 0.25
            and row.champion_residual_direction_alignment >= 0.60
        )
        if not is_localized:
            continue
        record = {
            "family": row.family,
            "distinct_ratio_vs_max_placebo": row.distinct_ratio_vs_max_placebo,
            "weighted_mean_abs_conditional_delta": row.weighted_mean_abs_conditional_delta,
            "significant_cell_fraction": row.significant_cell_fraction,
            "champion_residual_direction_alignment": row.champion_residual_direction_alignment,
        }
        localized.append(record)
        is_strong = bool(
            row.weighted_mean_abs_conditional_delta >= 0.010
            and row.distinct_ratio_vs_max_placebo >= 1.50
            and row.significant_cell_fraction >= 0.50
            and row.champion_residual_direction_alignment >= 0.70
        )
        if not is_strong:
            continue
        strong_structural.append(record)
        if row.family in latest_persistence.index:
            persistence_row = latest_persistence.loc[row.family]
            if (
                persistence_row["n_common_cells"] >= 4
                and persistence_row["pearson_shrunk_delta"] >= 0.30
                and persistence_row["sign_agreement"] >= 0.60
            ):
                persistent_strong.append(
                    record
                    | {
                        "pearson_shrunk_delta": persistence_row[
                            "pearson_shrunk_delta"
                        ],
                        "sign_agreement": persistence_row["sign_agreement"],
                    }
                )
    if persistent_strong:
        return (
            "A. STRUCTURAL 2024 REGIME SHIFT FOUND",
            strong_structural,
            localized,
            persistent_strong,
        )
    if localized:
        return "B. WEAK / LOCALIZED SHIFT", strong_structural, localized, persistent_strong
    return "C. NO DISTINCT RULE-REGIME SIGNAL", [], [], []


def leakage_audit(frame: pd.DataFrame) -> dict:
    before = add_static_contexts(frame[frame["season"].isin(ANALYSIS_SEASONS)])
    flipped = frame.copy()
    flipped.loc[flipped["season"].eq(2024), TARGET] = 1 - flipped.loc[
        flipped["season"].eq(2024), TARGET
    ]
    after = add_static_contexts(flipped[flipped["season"].isin(ANALYSIS_SEASONS)])
    columns = list(CONTEXT_COLUMNS.values())
    same = all(before[column].equals(after[column]) for column in columns)
    return {
        "validation_target_flip_changes_static_contexts": not same,
        "static_contexts_same_after_validation_target_flip": same,
        "test_data_read": False,
        "passed": same,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    champion_sha_before = sha256_path(CURRENT_CHAMPION)
    if champion_sha_before != CURRENT_CHAMPION_SHA256:
        raise ValueError("immutable champion checksum mismatch before analysis")

    header = pd.read_csv(args.train, nrows=0).columns.tolist()
    time_columns = [name for name in ("season", "year", "game_date", "game_id") if name in header]
    missing = sorted(set(BASE_COLUMNS) - set(header))
    if missing:
        raise ValueError(f"train schema missing required columns: {missing}")
    print(f"[schema] time columns={time_columns}; test data not read", flush=True)
    frame = pd.read_csv(args.train, usecols=list(BASE_COLUMNS))
    frame = add_static_contexts(frame)
    frame = attach_oof_contexts(frame)

    season_table = season_summary(frame)
    global_rates = season_table.set_index("season")["control_success_rate"].to_dict()
    season_stats = [
        grouped_season_stats(frame, family, column)
        for family, column in CONTEXT_COLUMNS.items()
    ]
    season_stats.append(special_count_stats(frame))
    all_stats = pd.concat(season_stats, ignore_index=True)
    conditional = boundary_shifts(all_stats, global_rates)
    frequency = frequency_shift_table(conditional)

    champion_predictions = load_current_champion_oof(frame)
    residual = champion_residual_table(frame, champion_predictions)
    placebo = placebo_summary(conditional, residual)
    persistence = persistence_table(conditional)
    ranking = rank_structural_contexts(conditional, residual)
    (
        verdict,
        structural_families,
        localized_families,
        persistent_families,
    ) = stage_a_decision(placebo, persistence)

    count_stats = all_stats[all_stats["family"].isin(["count_state", "count_special"])].copy()
    count_boundary = conditional[
        conditional["family"].isin(["count_state", "count_special"])
    ][
        [
            "family",
            "context",
            "to_season",
            "n_from",
            "n_to",
            "raw_delta",
            "standard_error",
            "ci95_low",
            "ci95_high",
            "shrunk_delta",
            "small_cell",
        ]
    ].rename(columns={"to_season": "season"})
    count_stats = count_stats.merge(
        count_boundary,
        on=["family", "context", "season"],
        how="left",
        validate="one_to_one",
    )
    matchup = conditional[
        conditional["family"].isin(["pitcher_hand", "hand_matchup"])
    ].copy()
    game_type = conditional[
        conditional["family"].isin(["game_type", "game_type_x_count"])
    ].copy()

    args.output.mkdir(parents=True, exist_ok=True)
    season_table.to_csv(args.output / "season_summary.csv", index=False)
    count_stats.to_csv(args.output / "count_shift.csv", index=False)
    matchup.to_csv(args.output / "matchup_shift.csv", index=False)
    game_type.to_csv(args.output / "game_type_shift.csv", index=False)
    frequency.to_csv(args.output / "context_frequency_shift.csv", index=False)
    conditional.to_csv(args.output / "conditional_shift.csv", index=False)
    placebo.to_csv(args.output / "placebo_boundaries.csv", index=False)
    residual.to_csv(args.output / "champion_residual_shift.csv", index=False)
    persistence.to_csv(args.output / "persistence.csv", index=False)

    champion_sha_after = sha256_path(CURRENT_CHAMPION)
    if champion_sha_after != champion_sha_before:
        raise ValueError("immutable champion changed during analysis")
    leakage = leakage_audit(frame)
    top_columns = [
        "structural_rank",
        "family",
        "context",
        "n_from",
        "n_to",
        "rate_from",
        "rate_to",
        "raw_delta",
        "shrunk_delta",
        "standard_error",
        "distinct_ratio_vs_cell_placebo",
        "mean_y_minus_prediction",
        "brier",
        "champion_residual_direction_aligned",
        "related_context_direction_consistent",
        "ranking_score",
    ]
    summary = {
        "verdict": verdict,
        "stage_b_executed": False,
        "stage_b_reason": (
            "Stage A did not produce a persistent structural family"
            if verdict != "A. STRUCTURAL 2024 REGIME SHIFT FOUND"
            else "Stage A passed; Stage B requires a separate model-design round"
        ),
        "current_immutable_champion": {
            "lb_bss": CURRENT_CHAMPION_LB_BSS,
            "artifact": str(CURRENT_CHAMPION),
            "sha256_before": champion_sha_before,
            "sha256_after": champion_sha_after,
            "unchanged": champion_sha_before == champion_sha_after,
        },
        "train_contract": {
            "rows": len(frame),
            "seasons": sorted(frame["season"].unique().tolist()),
            "time_columns_present": time_columns,
            "exact_game_count_available": False,
            "reason": "train.csv exposes neither game_id nor game_date",
            "test_data_used": False,
        },
        "target_contract": {
            "documented_meaning": "1=control success, 0=control failure under an operational criterion",
            "current_pitch_post_outcome_available": False,
            "direct_called_strike_or_ball_mapping_disclosed": False,
            "causal_interpretation": (
                "A 2024 rule/environment change is only a timing hypothesis. The supplied "
                "definition does not establish that ABS directly determines control_success."
            ),
        },
        "analysis_contract": {
            "boundary_hypothesis": "PRE_2024 versus 2024",
            "placebo_boundaries": ["2021_to_2022", "2022_to_2023"],
            "minimum_cell_n": MIN_CELL_N,
            "ranking_minimum_cell_n": RANK_MIN_CELL_N,
            "shrinkage": (
                "delta shrunk toward boundary-wide target shift with harmonic-n / "
                f"(harmonic-n + {SHRINKAGE_TAU:g})"
            ),
            "hand_labels": (
                "H1/H2 are preserved because the official description does not map codes to R/L"
            ),
            "current_champion_residual_oof_seasons": list(CHAMPION_OOF_SEASONS),
            "current_champion_residual_limit": (
                "the accepted ET25 prefix OOF contract starts in 2022; no new 2021 forest was trained"
            ),
            "localized_b_gate": {
                "minimum_eligible_cells": 4,
                "minimum_weighted_abs_conditional_delta": 0.004,
                "minimum_distinct_ratio_vs_placebo": 1.25,
                "minimum_significant_cell_fraction": 0.25,
                "minimum_champion_residual_alignment": 0.60,
            },
            "strong_a_gate": {
                "minimum_weighted_abs_conditional_delta": 0.010,
                "minimum_distinct_ratio_vs_placebo": 1.50,
                "minimum_significant_cell_fraction": 0.50,
                "minimum_champion_residual_alignment": 0.70,
                "minimum_persistence_correlation": 0.30,
                "minimum_persistence_sign_agreement": 0.60,
            },
        },
        "global_target_shifts": [
            {
                "from_season": left,
                "to_season": right,
                "rate_from": global_rates[left],
                "rate_to": global_rates[right],
                "delta": global_rates[right] - global_rates[left],
            }
            for left, right in BOUNDARIES
        ],
        "structural_families": structural_families,
        "localized_families": localized_families,
        "persistent_structural_families": persistent_families,
        "top_structural_contexts": ranking[top_columns].head(20).to_dict("records"),
        "leakage_audit": leakage,
        "production": {
            "created": False,
            "reason": "Stage A analysis only; production ZIP creation is forbidden",
            "filename_length_hard_gate": 30,
        },
    }
    write_json(args.output / "summary.json", summary)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "structural_families": [x["family"] for x in structural_families],
                "localized_families": [x["family"] for x in localized_families],
                "persistent_structural_families": [
                    x["family"] for x in persistent_families
                ],
                "champion_unchanged": champion_sha_before == champion_sha_after,
                "production_created": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

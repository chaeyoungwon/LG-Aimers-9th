"""Read-only champion recoverable-headroom audit from existing OOF predictions.

This script never fits a model and never reads test data.  It only aligns official
train rows with already-saved temporal OOF predictions and writes aggregate audit
tables under artifacts/headroom_audit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / "LG-Aimers-9th"
TRAIN = ROOT.parent / "open/data/train.csv"
OUT = ROOT / "artifacts/headroom_audit"
OOF = ROOT / "artifacts/backtest/champion_oof"
SEASONS = (2022, 2023, 2024)
PRIMARY = (2022, 2024)
MIN_CELL = 5_000
QUALITY_CANDIDATES = (
    "champion",
    "cat",
    "team",
    "uncertainty_0p5",
    "role_state",
    "two_strike",
)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


def bss_gain(y: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> float:
    denominator = float(y.mean() * (1.0 - y.mean()))
    return 100_000.0 * (brier(y, base) - brier(y, candidate)) / denominator


def load_npy(path: Path, n: int) -> np.ndarray:
    value = np.load(path).astype("float64")
    if value.shape != (n,) or not np.isfinite(value).all():
        raise ValueError(f"invalid prediction {path}: {value.shape}")
    return value


def load_npz_prediction(path: Path, key: str, n: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        value = np.asarray(data[key], dtype="float64")
    if value.shape != (n,) or not np.isfinite(value).all():
        raise ValueError(f"invalid prediction {path}:{key}: {value.shape}")
    return value


def current_prediction(season: int, n: int) -> np.ndarray:
    return 0.3 * load_npy(OOF / f"{season}_cat.npy", n) + 0.7 * load_npy(
        OOF / f"{season}_team.npy", n
    )


def two_strike_prediction(frame: pd.DataFrame, season: int, champion: np.ndarray) -> np.ndarray:
    active = (frame.strikes_before.to_numpy() == 2) & (frame.balls_before.to_numpy() < 3)
    lgb = np.mean(
        [
            load_npy(ROOT / f"artifacts/residual_slice/two_strike_expert_{season}_s{seed}.npy", int(active.sum()))
            for seed in (42, 43, 44)
        ],
        axis=0,
    )
    cat = np.mean(
        [
            load_npy(ROOT / f"artifacts/two_strike_final/{season}_cat_s{seed}.npy", int(active.sum()))
            for seed in (42, 43, 44)
        ],
        axis=0,
    )
    result = champion.copy()
    result[active] = 0.75 * champion[active] + 0.25 * (0.5 * lgb + 0.5 * cat)
    return result


def candidate_predictions(frame: pd.DataFrame, season: int) -> dict[str, np.ndarray]:
    n = len(frame)
    candidates: dict[str, np.ndarray] = {
        "champion": current_prediction(season, n),
        "cat": load_npy(OOF / f"{season}_cat.npy", n),
        "team": load_npy(OOF / f"{season}_team.npy", n),
        "hgb": load_npy(OOF / f"{season}_hgb.npy", n),
        "nn": load_npy(OOF / f"{season}_nn.npy", n),
        "team_nn": load_npy(OOF / f"{season}_team_nn.npy", n),
    }
    diversity = ARCHIVE / "artifacts/model_diversity/cache"
    for name, prefix in (("ft_transformer", "ft_transformer"), ("tabm", "tabm"), ("xgb_hist", "xgb_hist")):
        candidates[name] = load_npz_prediction(diversity / f"{prefix}_{season}.npz", "prediction", n)

    hard = ROOT / "artifacts/backtest/hard_example"
    candidates["hard_weight"] = np.mean(
        [load_npy(hard / f"{season}_hard_s{seed}.npy", n) for seed in (42, 43)], axis=0
    )
    candidates["uncertainty_0p5"] = np.mean(
        [load_npy(hard / f"{season}_uncertainty_0p5_s{seed}.npy", n) for seed in (42, 43)], axis=0
    )
    role = ROOT / "artifacts/backtest/role_state"
    candidates["role_state"] = np.mean(
        [load_npy(role / f"{season}_candidate_s{seed}.npy", n) for seed in (0, 1)], axis=0
    )
    candidates["groupdro"] = load_npz_prediction(
        ARCHIVE / f"artifacts/season_groupdro/cache/season_{season}_groupdro_eta005_seed42.npz",
        "prediction",
        n,
    )
    candidates["lupi_aux"] = load_npz_prediction(
        ARCHIVE / f"artifacts/lupi_aux_oof/cache/season_{season}.npz", "multi_seed42", n
    )
    candidates["lupi_distill"] = load_npz_prediction(
        ARCHIVE / f"artifacts/lupi_distill_oof/cache/season_{season}.npz", "student_alpha_050", n
    )
    candidates["two_strike"] = two_strike_prediction(frame, season, candidates["champion"])
    return candidates


def add_frozen_support(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["count_state"] = result.balls_before * 3 + result.strikes_before
    result["count_family"] = np.select(
        [result.strikes_before > result.balls_before, result.strikes_before == result.balls_before],
        ["S>B", "S=B"],
        default="S<B",
    )
    result["pitcher_history"] = pd.cut(
        result.asof_pitcher_n, [-1, 99, 499, np.inf], labels=["<100", "100-499", "500+"]
    ).astype(str)
    result["batter_history"] = pd.cut(
        result.asof_batter_n, [-1, 99, 499, np.inf], labels=["<100", "100-499", "500+"]
    ).astype(str)
    result["hand_combo_audit"] = result.pitcher_hand.astype(str) + "-" + result.batter_hand.astype(str)
    result["inning_phase"] = pd.cut(
        result.inning, [0, 3, 6, np.inf], labels=["1-3", "4-6", "7+"]
    ).astype(str)
    result["base_out_state"] = result.base_state.astype(str) + "|" + result.outs_before.astype(str)
    return result


def add_matchup_support(full: pd.DataFrame, validation: pd.DataFrame, season: int) -> pd.DataFrame:
    history = full[full.season < season].copy()
    history["count_state"] = history.balls_before * 3 + history.strikes_before
    output = validation.copy()
    pair = history.groupby(["pitcher_id", "batter_id"]).size()
    keys = pd.MultiIndex.from_frame(output[["pitcher_id", "batter_id"]])
    count = pair.reindex(keys, fill_value=0).to_numpy()
    output["matchup_history"] = pd.cut(
        count, [-1, 0, 19, np.inf], labels=["unseen", "1-19", "20+"]
    ).astype(str)
    return output


def error_summary(season: int, frame: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> list[dict]:
    error = (y - p) ** 2
    quantiles = {q: float(np.quantile(error, q)) for q in (0.2, 0.8, 0.9, 0.95, 0.99)}
    masks = {
        "top_1pct": error >= quantiles[0.99],
        "top_5pct": error >= quantiles[0.95],
        "top_10pct": error >= quantiles[0.9],
        "top_20pct": error >= quantiles[0.8],
        "middle_60pct": (error > quantiles[0.2]) & (error < quantiles[0.8]),
        "bottom_20pct": error <= quantiles[0.2],
        "all": np.ones(len(y), dtype=bool),
    }
    rows = []
    for name, mask in masks.items():
        rows.append(
            {
                "season": season,
                "error_band": name,
                "rows": int(mask.sum()),
                "row_share": float(mask.mean()),
                "mean_brier": float(error[mask].mean()),
                "brier_contribution": float(error[mask].sum() / error.sum()),
                "target_mean": float(y[mask].mean()),
                "prediction_mean": float(p[mask].mean()),
                "calibration_bias_y_minus_p": float((y[mask] - p[mask]).mean()),
                "mean_absolute_error": float(np.abs(y[mask] - p[mask]).mean()),
            }
        )
    return rows


def candidate_metrics(
    season: int, y: np.ndarray, predictions: dict[str, np.ndarray]
) -> tuple[list[dict], np.ndarray, np.ndarray, list[str]]:
    base = predictions["champion"]
    base_error = (y - base) ** 2
    names = list(predictions)
    matrix = np.column_stack([predictions[name] for name in names])
    rows = []
    for name in names:
        pred = predictions[name]
        gain = base_error - (y - pred) ** 2
        row = {
            "season": season,
            "candidate": name,
            "rows": len(y),
            "brier": brier(y, pred),
            "brier_gain": float(gain.mean()),
            "bss_gain": bss_gain(y, base, pred),
            "positive_gain_row_share": float((gain > 0).mean()),
            "negative_gain_row_share": float((gain < 0).mean()),
            "residual_correlation": float(np.corrcoef(y - base, y - pred)[0, 1]),
            "prediction_correlation": float(np.corrcoef(base, pred)[0, 1]),
            "mean_prediction_delta": float(np.mean(pred - base)),
        }
        for q in (0.05, 0.10, 0.20):
            mask = base_error >= np.quantile(base_error, 1.0 - q)
            row[f"top_{int(q*100)}pct_brier_gain"] = float(gain[mask].mean())
            row[f"top_{int(q*100)}pct_win_share"] = float((gain[mask] > 0).mean())
        rows.append(row)
    return rows, matrix, base_error, names


def overlap_metrics(season: int, y: np.ndarray, predictions: dict[str, np.ndarray]) -> list[dict]:
    base_error = (y - predictions["champion"]) ** 2
    names = [name for name in predictions if name != "champion"]
    gains = {name: base_error - (y - predictions[name]) ** 2 for name in names}
    rows = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            a, b = gains[left] > 0, gains[right] > 0
            union = a | b
            rows.append(
                {
                    "season": season,
                    "candidate_a": left,
                    "candidate_b": right,
                    "positive_row_jaccard": float((a & b).sum() / union.sum()) if union.any() else np.nan,
                    "positive_row_overlap_share": float((a & b).mean()),
                    "gain_correlation": float(np.corrcoef(gains[left], gains[right])[0, 1]),
                }
            )
    return rows


AXES = (
    "game_type",
    "count_family",
    "pitcher_history",
    "batter_history",
    "hand_combo_audit",
    "inning_phase",
    "base_out_state",
    "matchup_history",
)


def discover_route(
    labels: pd.Series, y: np.ndarray, predictions: dict[str, np.ndarray]
) -> dict[str, str]:
    route: dict[str, str] = {}
    for value in sorted(labels.astype(str).unique()):
        mask = labels.astype(str).eq(value).to_numpy()
        if mask.sum() < MIN_CELL:
            route[value] = "champion"
            continue
        route[value] = min(predictions, key=lambda name: brier(y[mask], predictions[name][mask]))
    return route


def apply_route(labels: pd.Series, predictions: dict[str, np.ndarray], route: dict[str, str]) -> np.ndarray:
    output = predictions["champion"].copy()
    string_labels = labels.astype(str)
    for value, candidate in route.items():
        mask = string_labels.eq(value).to_numpy()
        if candidate in predictions:
            output[mask] = predictions[candidate][mask]
    return output


def entropy_proxy(full: pd.DataFrame, validation: pd.DataFrame, season: int) -> np.ndarray:
    columns = ["game_type", "count_state", "pitcher_hand", "batter_hand", "base_state", "outs_before"]
    history = add_frozen_support(full[full.season < season])
    global_mean = float(history.control_success.mean())
    stats = history.groupby(columns, dropna=False).control_success.agg(["sum", "size"])
    keys = pd.MultiIndex.from_frame(validation[columns])
    success = stats["sum"].reindex(keys, fill_value=0).to_numpy(float)
    support = stats["size"].reindex(keys, fill_value=0).to_numpy(float)
    probability = (success + 100.0 * global_mean) / (support + 100.0)
    probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return -(probability * np.log2(probability) + (1.0 - probability) * np.log2(1.0 - probability))


def proxy_lb_transfer() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"submission": "19_full_state", "local_proxy_bss": 8.0, "local_definition": "coverage-proportional expectation", "actual_lb_delta": -4.0091059925, "reference": "18th"},
            {"submission": "20_stage_w0p500", "local_proxy_bss": 1.99, "local_definition": "prediction-difference 2SE proxy", "actual_lb_delta": -3.4923533629, "reference": "18th"},
            {"submission": "21_tree_reblend", "local_proxy_bss": 31.89, "local_definition": "2025-structure proxy", "actual_lb_delta": 6.5207072826, "reference": "18th"},
            {"submission": "22_two_strike", "local_proxy_bss": float(np.mean([5.75, 24.02, 4.34])), "local_definition": "mean rolling 2022/2023/2024 delta BSS", "actual_lb_delta": -1.810011281, "reference": "21st"},
        ]
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(TRAIN, encoding="utf-8-sig")
    frames: dict[int, pd.DataFrame] = {}
    targets: dict[int, np.ndarray] = {}
    predictions: dict[int, dict[str, np.ndarray]] = {}
    candidate_rows: list[dict] = []
    error_rows: list[dict] = []
    overlap_rows: list[dict] = []
    disagreement_rows: list[dict] = []

    for season in SEASONS:
        frame = add_matchup_support(full, add_frozen_support(full[full.season == season].copy()), season)
        frame.reset_index(drop=True, inplace=True)
        y = frame.control_success.to_numpy(float)
        pred = candidate_predictions(frame, season)
        frames[season], targets[season], predictions[season] = frame, y, pred
        error_rows.extend(error_summary(season, frame, y, pred["champion"]))
        rows, matrix, base_error, names = candidate_metrics(season, y, pred)
        candidate_rows.extend(rows)
        overlap_rows.extend(overlap_metrics(season, y, pred))
        entropy = entropy_proxy(full, frame, season)
        for scope, scoped_names in (
            ("all_candidates", names),
            ("quality_filtered", [name for name in QUALITY_CANDIDATES if name in pred]),
        ):
            scoped = np.column_stack([pred[name] for name in scoped_names])
            dispersion = scoped.std(axis=1)
            spread = scoped.max(axis=1) - scoped.min(axis=1)
            oracle_error = np.min((y[:, None] - scoped) ** 2, axis=1)
            bands = {
                "high_disagreement_top20": dispersion >= np.quantile(dispersion, 0.8),
                "low_disagreement_bottom20": dispersion <= np.quantile(dispersion, 0.2),
                "all": np.ones(len(y), dtype=bool),
            }
            for band, mask in bands.items():
                disagreement_rows.append(
                    {
                        "season": season,
                        "candidate_scope": scope,
                        "candidate_count": len(scoped_names),
                        "band": band,
                        "rows": int(mask.sum()),
                        "row_share": float(mask.mean()),
                        "champion_brier": float(base_error[mask].mean()),
                        "oracle_brier": float(oracle_error[mask].mean()),
                        "oracle_bss_gain": 100_000.0 * float((base_error[mask] - oracle_error[mask]).mean()) / (y.mean() * (1-y.mean())),
                        "prediction_std_mean": float(dispersion[mask].mean()),
                        "prediction_range_mean": float(spread[mask].mean()),
                        "state_entropy_bits_mean": float(entropy[mask].mean()),
                        "high_entropy_share": float((entropy[mask] >= 0.95).mean()),
                    }
                )

    error_df = pd.DataFrame(error_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    overlap_df = pd.DataFrame(overlap_rows)
    disagreement_df = pd.DataFrame(disagreement_rows)

    oracle_rows: list[dict] = []
    stable_rows: list[dict] = []
    stable_masks: dict[int, np.ndarray] = {season: np.zeros(len(targets[season]), dtype=bool) for season in PRIMARY}
    stable_axis_results = []

    for season in PRIMARY:
        y, pred = targets[season], predictions[season]
        matrix = np.column_stack(list(pred.values()))
        oracle = np.where(
            y[:, None] == 1,
            matrix.max(axis=1, keepdims=True),
            matrix.min(axis=1, keepdims=True),
        )[:, 0]
        oracle_rows.append(
            {"bound_type": "UNDEPLOYABLE_PER_ROW_ORACLE_ALL", "discovery_season": season, "evaluation_season": season, "axis": "row_target", "brier_gain": brier(y, pred["champion"]) - brier(y, oracle), "bss_gain": bss_gain(y, pred["champion"], oracle)}
        )
        quality_matrix = np.column_stack([pred[name] for name in QUALITY_CANDIDATES])
        quality_oracle = np.where(
            y[:, None] == 1,
            quality_matrix.max(axis=1, keepdims=True),
            quality_matrix.min(axis=1, keepdims=True),
        )[:, 0]
        oracle_rows.append(
            {"bound_type": "UNDEPLOYABLE_PER_ROW_ORACLE_QUALITY_FILTERED", "discovery_season": season, "evaluation_season": season, "axis": "row_target", "brier_gain": brier(y, pred["champion"]) - brier(y, quality_oracle), "bss_gain": bss_gain(y, pred["champion"], quality_oracle)}
        )

    for discovery, evaluation in ((2022, 2024), (2024, 2022)):
        yd, ye = targets[discovery], targets[evaluation]
        pd_, pe = predictions[discovery], predictions[evaluation]
        best_global = min(pd_, key=lambda name: brier(yd, pd_[name]))
        oracle_rows.append(
            {"bound_type": "CROSS_FITTED_GLOBAL_PROXY", "discovery_season": discovery, "evaluation_season": evaluation, "axis": best_global, "brier_gain": brier(ye, pe["champion"]) - brier(ye, pe[best_global]), "bss_gain": bss_gain(ye, pe["champion"], pe[best_global])}
        )
        for axis in AXES:
            route = discover_route(frames[discovery][axis], yd, pd_)
            routed = apply_route(frames[evaluation][axis], pe, route)
            oracle_rows.append(
                {"bound_type": "CROSS_FITTED_LEGAL_AXIS", "discovery_season": discovery, "evaluation_season": evaluation, "axis": axis, "brier_gain": brier(ye, pe["champion"]) - brier(ye, routed), "bss_gain": bss_gain(ye, pe["champion"], routed)}
            )

    for axis in AXES:
        routes = {season: discover_route(frames[season][axis], targets[season], predictions[season]) for season in PRIMARY}
        route_common: dict[str, str] = {}
        values = sorted(set(routes[2022]) | set(routes[2024]))
        for value in values:
            left, right = routes[2022].get(value, "champion"), routes[2024].get(value, "champion")
            if left == right and left != "champion":
                masks = {s: frames[s][axis].astype(str).eq(value).to_numpy() for s in PRIMARY}
                if all(mask.sum() >= MIN_CELL for mask in masks.values()):
                    gains = {
                        s: brier(targets[s][masks[s]], predictions[s]["champion"][masks[s]])
                        - brier(targets[s][masks[s]], predictions[s][left][masks[s]])
                        for s in PRIMARY
                    }
                    directions = {s: float(np.mean(predictions[s][left][masks[s]] - predictions[s]["champion"][masks[s]])) for s in PRIMARY}
                    if all(gain > 0 for gain in gains.values()) and np.sign(directions[2022]) == np.sign(directions[2024]):
                        route_common[value] = left
                        for s in PRIMARY:
                            stable_masks[s] |= masks[s]
                        stable_rows.append(
                            {"axis": axis, "value": value, "candidate": left, "rows_2022": int(masks[2022].sum()), "coverage_2022": float(masks[2022].mean()), "brier_gain_2022": gains[2022], "bss_gain_2022": 100_000*gains[2022]/(targets[2022].mean()*(1-targets[2022].mean())), "rows_2024": int(masks[2024].sum()), "coverage_2024": float(masks[2024].mean()), "brier_gain_2024": gains[2024], "bss_gain_2024": 100_000*gains[2024]/(targets[2024].mean()*(1-targets[2024].mean())), "prediction_delta_2022": directions[2022], "prediction_delta_2024": directions[2024]}
                        )
        routed_results = {}
        for season in PRIMARY:
            routed = apply_route(frames[season][axis], predictions[season], route_common)
            routed_results[season] = bss_gain(targets[season], predictions[season]["champion"], routed)
        stable_axis_results.append((axis, route_common, routed_results))
        oracle_rows.append(
            {"bound_type": "LEGAL_STABLE_BOTH_SEASONS_RETROSPECTIVE", "discovery_season": "2022+2024", "evaluation_season": 2024, "axis": axis, "brier_gain": routed_results[2024]*(targets[2024].mean()*(1-targets[2024].mean()))/100_000, "bss_gain": routed_results[2024]}
        )

    stable_df = pd.DataFrame(stable_rows)
    oracle_df = pd.DataFrame(oracle_rows)

    best_stable_axis, best_stable_route, best_stable_result = max(
        stable_axis_results, key=lambda item: item[2][2024]
    )
    best_stable_masks = {
        season: frames[season][best_stable_axis].astype(str).isin(best_stable_route).to_numpy()
        for season in PRIMARY
    }

    # Overlapping diagnostic decomposition. Shares must not be summed.
    decomposition = []
    for season in SEASONS:
        frame, y, pred = frames[season], targets[season], predictions[season]
        error = (y - pred["champion"]) ** 2
        if season in PRIMARY:
            other = 2024 if season == 2022 else 2022
            bias_now = frame.assign(residual=y-pred["champion"]).groupby(["game_type", "count_family"]).residual.mean()
            fo, yo, po = frames[other], targets[other], predictions[other]
            bias_other = fo.assign(residual=yo-po["champion"]).groupby(["game_type", "count_family"]).residual.mean()
            common = bias_now.to_frame("a").join(bias_other.to_frame("b"), how="inner")
            keys = pd.MultiIndex.from_frame(frame[["game_type", "count_family"]])
            stable_bias = common[(np.sign(common.a)==np.sign(common.b)) & (common.a.abs()>=.005) & (common.b.abs()>=.005)].index
            calibration = keys.isin(stable_bias)
        else:
            calibration = frame.game_type.eq("F").to_numpy()
        top20 = error >= np.quantile(error, .8)
        sparse = (frame.asof_pitcher_n.to_numpy()<100) | (frame.asof_batter_n.to_numpy()<100) | frame.matchup_history.eq("unseen").to_numpy()
        interaction = frame.matchup_history.eq("1-19").to_numpy()
        temporal = frame.game_type.eq("F").to_numpy() if season == 2023 else np.zeros(len(frame),bool)
        model_family = best_stable_masks[season] if season in PRIMARY else np.zeros(len(frame),bool)
        pred_matrix = np.column_stack([pred[name] for name in QUALITY_CANDIDATES])
        dispersion = pred_matrix.std(axis=1)
        entropy = entropy_proxy(full, frame, season)
        unexplained = (dispersion <= np.quantile(dispersion,.2)) & (entropy >= .95)
        masks = {
            "CALIBRATION_LIKE": calibration,
            "DISCRIMINATION_LIKE": top20 & ~calibration,
            "SPARSE_RELIABILITY": sparse,
            "INTERACTION_FAILURE": interaction,
            "TEMPORAL_PRIOR_FAILURE": temporal,
            "MODEL_FAMILY_RECOVERABLE": model_family,
            "UNEXPLAINED_IRREDUCIBLE_LOOKING": unexplained,
        }
        for name, mask in masks.items():
            decomposition.append({"season":season,"error_type":name,"rows":int(mask.sum()),"row_share":float(mask.mean()),"mean_brier":float(error[mask].mean()) if mask.any() else np.nan,"brier_contribution_per_all_rows":float((error*mask).mean()),"share_of_total_brier":float(error[mask].sum()/error.sum()) if mask.any() else 0.0,"overlapping_classes":True})

    decomposition_df = pd.DataFrame(decomposition)
    transfer_df = proxy_lb_transfer()
    target_rows = []
    for season in PRIMARY:
        denominator = targets[season].mean() * (1-targets[season].mean())
        for points in (10,20,40):
            target_rows.append({"season":season,"target_bss_gain":points,"required_overall_brier_gain":points*denominator/100_000})
    target_df = pd.DataFrame(target_rows)

    error_df.to_csv(OUT / "champion_error_summary.csv", index=False)
    candidate_df.to_csv(OUT / "candidate_recovery_matrix.csv", index=False)
    overlap_df.to_csv(OUT / "candidate_recovery_overlap.csv", index=False)
    disagreement_df.to_csv(OUT / "model_disagreement.csv", index=False)
    stable_df.to_csv(OUT / "stable_recoverable_slices.csv", index=False)
    oracle_df.to_csv(OUT / "oracle_bounds.csv", index=False)
    transfer_df.to_csv(OUT / "proxy_lb_transfer.csv", index=False)
    decomposition_df.to_csv(OUT / "headroom_decomposition.csv", index=False)
    target_df.to_csv(OUT / "bss_targets.csv", index=False)

    manifest = {
        "train": str(TRAIN),
        "train_sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
        "seasons": list(SEASONS),
        "candidate_names": list(predictions[2022]),
        "candidate_count": len(predictions[2022]),
        "no_training": True,
        "test_read": False,
        "minimum_legal_cell_rows": MIN_CELL,
        "legal_axes": list(AXES),
        "stable_axis_results": [{"axis":axis,"route":route,"bss":result} for axis,route,result in stable_axis_results],
        "best_stable_single_axis": {"axis":best_stable_axis,"route":best_stable_route,"bss":best_stable_result},
        "quality_candidates": list(QUALITY_CANDIDATES),
        "excluded_without_row_prediction": ["forward_residual_lgb", "trackman_profile", "hierarchical_eb", "stable_additive", "prototype_codebook"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

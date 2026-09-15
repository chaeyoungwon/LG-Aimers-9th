"""Re-evaluate frozen TrackMan LUPI OOF against the 21st champion.

This script never trains a model and never reads test data.  It aligns the
frozen legal-X-only student predictions by row_id with Cat30/team70 rolling
OOF, then reports paired Brier/BSS and fixed coarse blends.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "LG-Aimers-9th"
SEASONS = (2022, 2023, 2024)
WEIGHTS = (0.025, 0.05, 0.10, 0.15)


def metrics(y, champion, candidate):
    den = float(y.mean() * (1.0 - y.mean()))
    loss_delta = (champion - y) ** 2 - (candidate - y) ** 2
    return {
        "champion_brier": float(np.mean((champion - y) ** 2)),
        "candidate_brier": float(np.mean((candidate - y) ** 2)),
        "champion_bss": float(100000 * (1 - np.mean((champion - y) ** 2) / den)),
        "candidate_bss": float(100000 * (1 - np.mean((candidate - y) ** 2) / den)),
        "delta_bss": float(100000 * np.mean(loss_delta) / den),
        "two_sigma_bss": float(100000 * 2 * np.std(loss_delta) / np.sqrt(len(y)) / den),
        "prediction_correlation": float(np.corrcoef(champion, candidate)[0, 1]),
        "residual_correlation": float(np.corrcoef(champion - y, candidate - y)[0, 1]),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, default=ROOT.parent / "open/data/train.csv")
    p.add_argument("--champion-oof", type=Path, default=ROOT / "artifacts/backtest/champion_oof")
    p.add_argument("--upstream", type=Path, default=UPSTREAM)
    p.add_argument("--trackman", type=Path, default=ROOT.parent / "open/data/trackman_history.csv")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/backtest/lupi_champion21")
    args = p.parse_args(argv)
    train = pd.read_csv(args.train, encoding="utf-8-sig")
    args.output.mkdir(parents=True, exist_ok=True)
    kinds = {
        "auxiliary": ("lupi_aux_oof", "multi_seed42"),
        "distillation": ("lupi_distill_oof", "student_alpha_050"),
    }
    summary = {"test_read": False, "inference_trackman": False, "weights": WEIGHTS, "models": {}}
    blend_rows = []
    subset_rows = []
    anchors = pd.read_csv(
        args.upstream / "artifacts/lupi_linkage_audit/reciprocal_pitch_anchors.csv"
    )
    tm_ids = pd.read_csv(
        args.trackman, encoding="utf-8-sig",
        usecols=["trackman_id", "pitcher_trackman_id"],
    )
    anchors = anchors.merge(
        tm_ids, left_on="candidate_trackman_id", right_on="trackman_id",
        how="left", validate="one_to_one",
    )
    crosswalk = pd.read_csv(args.upstream / "artifacts/tm_crosswalk/mapping_candidates.csv")
    for label, (folder, key) in kinds.items():
        folds = []
        for season in SEASONS:
            z = np.load(args.upstream / f"artifacts/{folder}/cache/season_{season}.npz")
            rows = train[train.season.eq(season)]
            if not np.array_equal(z["row_id"], rows.row_id.to_numpy()):
                raise ValueError(f"row_id mismatch: {label}/{season}")
            y = z["target"].astype("float64")
            champion = (
                0.30 * np.load(args.champion_oof / f"{season}_cat.npy")
                + 0.70 * np.load(args.champion_oof / f"{season}_team.npy")
            )
            candidate = z[key].astype("float64")
            fold = {"season": season, "y": y, "champion": champion, "candidate": candidate}
            folds.append(fold)
            result = metrics(y, champion, candidate)
            result["season"] = season
            summary["models"].setdefault(label, {"folds": []})["folds"].append(result)
            for weight in WEIGHTS:
                blended = (1 - weight) * champion + weight * candidate
                row = metrics(y, champion, blended)
                blend_rows.append({"model": label, "season": season, "weight": weight, **row})

            # Coverage analysis uses only linkage/mapping rows at or before the
            # outer cutoff.  It never changes predictions or model inputs.
            cutoff = season - 1
            mapping = crosswalk[
                crosswalk.cutoff.eq(cutoff)
                & crosswalk.method.eq("hungarian")
                & crosswalk.accepted.astype(bool)
                & crosswalk.confidence.eq("HIGH")
            ][["main_pitcher_id", "tm_pitcher_id"]]
            safe = anchors[anchors.season.le(cutoff)].merge(
                mapping,
                left_on=["pitcher_id", "pitcher_trackman_id"],
                right_on=["main_pitcher_id", "tm_pitcher_id"],
                how="inner", validate="many_to_one",
            )
            counts = safe.groupby("pitcher_id").size()
            linked = rows.pitcher_id.map(counts).fillna(0).to_numpy()
            positive = counts[counts.gt(0)]
            threshold = float(positive.median()) if len(positive) else 0.0
            masks = {
                "linked_pitcher": linked > 0,
                "unlinked_pitcher": linked == 0,
                "high_history_pitcher": linked >= threshold,
                "low_history_pitcher": (linked > 0) & (linked < threshold),
                "cold_start": rows.asof_pitcher_n.fillna(0).le(0).to_numpy(),
                "non_cold_start": rows.asof_pitcher_n.fillna(0).gt(0).to_numpy(),
            }
            matchup = rows.pitcher_hand.astype(str) + "/" + rows.batter_hand.astype(str)
            count = rows.balls_before.astype(str) + "-" + rows.strikes_before.astype(str)
            for value in sorted(matchup.unique()):
                masks[f"matchup_{value}"] = matchup.eq(value).to_numpy()
            for value in sorted(count.unique()):
                masks[f"count_{value}"] = count.eq(value).to_numpy()
            blended = 0.95 * champion + 0.05 * candidate
            for subset, mask in masks.items():
                if mask.sum() == 0:
                    continue
                row = metrics(y[mask], champion[mask], blended[mask])
                subset_rows.append({
                    "model": label, "season": season, "subset": subset,
                    "n": int(mask.sum()), "history_median": threshold, **row,
                })

        y = np.concatenate([f["y"] for f in folds])
        champion = np.concatenate([f["champion"] for f in folds])
        candidate = np.concatenate([f["candidate"] for f in folds])
        pooled = metrics(y, champion, candidate)
        worst = np.argsort((champion - y) ** 2)[-len(y) // 10:]
        pooled["champion_error_top10_candidate_brier_gain"] = float(
            np.mean((champion[worst] - y[worst]) ** 2 - (candidate[worst] - y[worst]) ** 2)
        )
        summary["models"][label]["pooled"] = pooled
        for weight in WEIGHTS:
            blended = (1 - weight) * champion + weight * candidate
            row = metrics(y, champion, blended)
            blend_rows.append({"model": label, "season": "pooled", "weight": weight, **row})

    pd.DataFrame(blend_rows).to_csv(args.output / "blend_results.csv", index=False)
    pd.DataFrame(subset_rows).to_csv(args.output / "subset_results.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

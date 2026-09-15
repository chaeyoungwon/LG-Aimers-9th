"""Build resumable rolling OOF predictions for the actual five-member champion.

Each target season is predicted by models trained on strictly earlier seasons.
Results are saved after every member so a long run can resume without retraining
completed cells. This cache is the required baseline for future high-gain work;
the leaf63 proxy is not reliable enough for member/stage or F-segment decisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.champion import Champion


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = ROOT / "artifacts" / "submit_season_state.zip"
DEFAULT_TRAIN = ROOT.parent / "open" / "data" / "train.csv"
DEFAULT_OUT = ROOT / "artifacts" / "backtest" / "champion_oof"
TARGETS = (2021, 2022, 2023, 2024)
MEMBERS = ("hgb", "cat", "nn", "team", "team_nn")


def prediction_path(out_dir, target, member):
    return out_dir / f"{target}_{member}.npy"


def valid_cache(path, expected_rows):
    if not path.exists():
        return False
    try:
        values = np.load(path, mmap_mode="r")
        return values.shape == (expected_rows,) and np.isfinite(values).all()
    except Exception:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--targets", type=int, nargs="+", default=list(TARGETS))
    parser.add_argument("--members", nargs="+", choices=MEMBERS, default=list(MEMBERS))
    args = parser.parse_args(argv)

    df = pd.read_csv(args.train, encoding="utf-8-sig")
    seasons = sorted(int(s) for s in df.season.unique())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    champion = Champion(args.zip, workdir=str(args.out_dir / "_spec"))
    manifest = {
        "zip": args.zip.name,
        "train": str(args.train),
        "seasons": seasons,
        "targets": [int(t) for t in args.targets],
        "members": list(args.members),
        "row_independent": True,
        "cells": {},
    }

    for target in args.targets:
        if target not in seasons:
            raise ValueError(f"target season {target} not in {seasons}")
        train_seasons = [s for s in seasons if s < target]
        expected = int(df.season.eq(target).sum())
        for member in args.members:
            path = prediction_path(args.out_dir, target, member)
            key = f"{target}/{member}"
            if valid_cache(path, expected):
                print(f"skip {key}: valid cache {path.name}", flush=True)
            else:
                print(
                    f"fit {key}: train={train_seasons}, target_rows={expected:,}",
                    flush=True,
                )
                pred = champion.fit_predict(
                    df,
                    train_seasons=train_seasons,
                    target_season=target,
                    members=[member],
                    verbose=True,
                )[member]
                if pred.shape != (expected,) or not np.isfinite(pred).all():
                    raise ValueError(f"invalid predictions for {key}: {pred.shape}")
                np.save(path, pred.astype("float64"))
                print(f"saved {path}", flush=True)
            values = np.load(path, mmap_mode="r")
            manifest["cells"][key] = {
                "file": path.name,
                "rows": int(len(values)),
                "mean": float(np.mean(values)),
            }
            (args.out_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    print(f"complete: {args.out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()

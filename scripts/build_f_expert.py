"""Build the conservative recent-regime F specialist candidate.

The expert uses one constant derived from official train only: the latest train
season's game_type=F target mean. At inference it blends that constant into F
rows only. No evaluation-row aggregation or ordering is used.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts" / "submit_season_state.zip"
TRAIN = ROOT.parent / "open" / "data" / "train.csv"
DEFAULT_WEIGHT = 0.25


def build(base, train_path, out, weight):
    train = pd.read_csv(
        train_path,
        encoding="utf-8-sig",
        usecols=["season", "game_type", "control_success"],
    )
    latest = int(train.season.max())
    rows = train[train.season.eq(latest) & train.game_type.eq("F")]
    if rows.empty:
        raise ValueError(f"latest season {latest} has no F rows")
    prior = float(rows.control_success.mean())
    spec = {
        "segment": "F",
        "group_col": "game_type",
        "weight": float(weight),
        "prior": prior,
        "source": {
            "data": f"official train season={latest}, game_type=F",
            "rows": int(len(rows)),
            "row_independent": True,
            "validation": (
                "state proxy all-row delta at w=0.25: "
                "2022 +263.0 / 2023 -108.2 / 2024 +51.7; "
                "high-risk recent-regime candidate"
            ),
        },
    }

    with tempfile.TemporaryDirectory(prefix="f_expert_") as tmp:
        work = Path(tmp)
        with zipfile.ZipFile(base) as archive:
            archive.extractall(work)
        meta_path = work / "model" / "meta.json"
        meta = json.loads(meta_path.read_text("utf-8"))
        meta["f_expert"] = spec
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        shutil.copyfile(ROOT / "scripts" / "f_expert_runtime.py", work / "f_expert_runtime.py")

        script_path = work / "script.py"
        script = script_path.read_text("utf-8")
        import_anchor = "from coldstart_runtime import apply_coldstart_expert\n"
        if script.count(import_anchor) != 1:
            raise RuntimeError("unexpected f-expert import anchor")
        script = script.replace(
            import_anchor,
            import_anchor + "from f_expert_runtime import apply_f_expert\n",
            1,
        )
        correction = '\n        preds = apply_corrections(preds, test, "./model")'
        replacement = correction + '\n        preds = apply_f_expert(preds, test, meta.get("f_expert"))'
        if script.count(correction) != 1:
            raise RuntimeError("unexpected member correction anchor")
        script = script.replace(correction, replacement, 1)
        correction = '\n    preds = apply_corrections(preds, test, "./model")'
        replacement = correction + '\n    preds = apply_f_expert(preds, test, meta.get("f_expert"))'
        if script.count(correction) != 1:
            raise RuntimeError("unexpected plain correction anchor")
        script = script.replace(correction, replacement, 1)
        script_path.write_text(script, encoding="utf-8")

        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(work.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(work))
    return spec


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--weight", type=float, default=DEFAULT_WEIGHT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if not 0.0 <= args.weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    tag = f"{args.weight:.3f}".replace(".", "p")
    out = args.out or ROOT / "artifacts" / f"submit_f_expert_w{tag}.zip"
    spec = build(args.base, args.train, out, args.weight)
    print(
        f"built {out}; prior={spec['prior']:.12f}; "
        f"weight={spec['weight']:.3f}; rows={spec['source']['rows']:,}"
    )


if __name__ == "__main__":
    main()

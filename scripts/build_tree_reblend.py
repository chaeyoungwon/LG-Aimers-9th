"""Build the tree-only CatBoost/LightGBM reblend candidate.

The deployed models are unchanged.  Only ``model/meta.json`` is edited so the
first stage selects CatBoost, the second selects the LightGBM team, and the
final blend uses a fixed train-derived weight.  Inference therefore remains
row independent: no statistic is computed from evaluation rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts" / "submit_season_state.zip"
DEFAULT_CAT_WEIGHT = 0.30


def hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def build(base: Path, out: Path, cat_weight: float) -> None:
    if not 0.0 <= cat_weight <= 1.0:
        raise ValueError("cat_weight must be in [0, 1]")
    with tempfile.TemporaryDirectory(prefix="tree_reblend_") as tmp:
        work = Path(tmp)
        with zipfile.ZipFile(base) as archive:
            archive.extractall(work)
        before = hashes(work)

        meta_path = work / "model" / "meta.json"
        meta = json.loads(meta_path.read_text("utf-8"))
        ours, team = meta["blend"]["stages"]
        ours["weights"] = [0.0, 1.0, 0.0]  # CatBoost only
        ours["calibration"] = None
        team["weights"] = [1.0, 0.0]       # LightGBM team only
        meta["blend"]["stage_weights"] = [cat_weight, 1.0 - cat_weight]
        meta["blend"]["composition"] = {
            "variant": "tree_reblend_cat_team",
            "cat_weight": cat_weight,
            "team_weight": 1.0 - cat_weight,
            "selection": "rolling OOF 2021-2024 plus 2025-structure proxy",
            "row_independent": True,
        }
        meta["tree_reblend"] = {
            "base": base.name,
            "cat_weight": cat_weight,
            "team_weight": 1.0 - cat_weight,
            "only_meta_changed": True,
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

        changed = sorted(
            name for name in set(before) | set(hashes(work))
            if before.get(name) != hashes(work).get(name)
        )
        if changed != ["model/meta.json"]:
            raise RuntimeError(f"unexpected changed files: {changed}")

        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(work.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(work))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--cat-weight", type=float, default=DEFAULT_CAT_WEIGHT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    tag = f"{args.cat_weight:.3f}".replace(".", "p")
    out = args.out or ROOT / "artifacts" / f"submit_tree_reblend_cat_w{tag}.zip"
    build(args.base, out, args.cat_weight)
    with zipfile.ZipFile(out) as archive:
        bad = archive.testzip()
    if bad:
        raise RuntimeError(f"ZIP CRC failed: {bad}")
    print(f"built {out}")
    print(f"sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()

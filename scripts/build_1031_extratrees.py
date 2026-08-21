"""Build the 1031.899 champion plus the proven 2% ET25 complement."""
import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts" / "submitsubmit.zip"
TRAIN = ROOT.parent / "open" / "data" / "train.csv"
OUT = ROOT / "artifacts" / "submit_1031_et25_w020.zip"

import sys
sys.path.insert(0, str(ROOT))
from scripts.extratrees_runtime import (build_hgb52_features, fit_preprocessor,
                                        transform_features)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if not args.base.exists() or not args.train.exists():
        raise FileNotFoundError((args.base, args.train))

    with tempfile.TemporaryDirectory(prefix="champion_et25_") as name:
        work = Path(name)
        with zipfile.ZipFile(args.base) as archive:
            archive.extractall(work)
        champion_meta = json.loads(
            (work / "model" / "meta.json").read_text(encoding="utf-8"))
        hgb = next(m for m in champion_meta["members"] if m["name"] == "hgb")
        train = pd.read_csv(args.train, encoding="utf-8-sig")
        features = build_hgb52_features(train, champion_meta["cat_levels"])
        if list(features.columns) != hgb["feature_columns"]:
            raise SystemExit("ExtraTrees feature contract differs from champion HGB")
        preprocessor = fit_preprocessor(features)
        matrix = transform_features(features, preprocessor)
        model = ExtraTreesClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=8,
            max_features="sqrt", bootstrap=False, n_jobs=-1,
            random_state=42)
        model.fit(matrix, train.control_success.to_numpy())
        # Production model is the exact ordered prefix validated on wooh.
        model.estimators_ = model.estimators_[:25]
        model.n_estimators = 25
        joblib.dump(model, work / "model" / "extra_trees.joblib",
                    compress=("zlib", 3))
        extra_meta = {
            "feature_columns": list(features.columns),
            "cat_levels": champion_meta["cat_levels"],
            "preprocessor": preprocessor,
            "champion_weight": 0.98,
            "extra_weight": 0.02,
            "model": {"n_estimators": 25, "min_samples_leaf": 8,
                      "max_features": "sqrt", "bootstrap": False,
                      "random_state": 42},
            "source": {"base": args.base.name,
                       "validation": "ET25 w=.020 previously reached LB 1017.0233",
                       "train_only": True, "row_independent": True},
        }
        (work / "model" / "extra_trees_meta.json").write_text(
            json.dumps(extra_meta, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8")
        shutil.copyfile(ROOT / "scripts" / "extratrees_runtime.py",
                        work / "extratrees_runtime.py")

        script_path = work / "script.py"
        script = script_path.read_text(encoding="utf-8")
        anchor = "from corrections_runtime import apply_corrections\n"
        if script.count(anchor) != 1:
            raise SystemExit("corrections import anchor missing")
        script = script.replace(
            anchor, anchor +
            "from extratrees_runtime import apply_extratrees_blend\n", 1)
        call = 'apply_corrections(preds, test, "./model")'
        member = "\n        return " + call
        plain = "\n    return " + call
        if script.count(member) != 1 or script.count(plain) != 1:
            raise SystemExit("corrections return anchors missing")
        script = script.replace(
            member, "\n        preds = " + call +
            '\n        return apply_extratrees_blend(preds, test, "./model")', 1)
        script = script.replace(
            plain, "\n    preds = " + call +
            '\n    return apply_extratrees_blend(preds, test, "./model")', 1)
        script_path.write_text(script, encoding="utf-8")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(work.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(work))
    print(f"built {args.out}; rows={len(train):,}; features={matrix.shape[1]}")


if __name__ == "__main__":
    main()

"""Build a separate R10 candidate with a heterogeneous cold-start expert."""
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from catboost import CatBoostClassifier
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts" / "submit_r10_pitcher_w418194.zip"
TRAIN = ROOT.parent / "open" / "data" / "train.csv"
OUT = ROOT / "artifacts" / "submit_r10_cathetero.zip"
DROP = {"row_id", "control_success", "pitcher_id", "batter_id"}

import sys
sys.path.insert(0, str(ROOT))
from src.train_base import CAT_COLS, add_features


def main():
    raw = pd.read_csv(TRAIN, encoding="utf-8-sig")
    train = add_features(raw[raw.game_type.eq("R")].copy())
    features = [c for c in train.columns if c not in DROP]
    cats = [c for c in CAT_COLS if c in features]
    for col in cats:
        train[col] = (train[col].astype("object")
                      .where(train[col].notna(), "__NA__").astype(str))
    cat_indices = [features.index(c) for c in cats]
    model = CatBoostClassifier(
        loss_function="Logloss", iterations=500, learning_rate=0.05,
        depth=6, l2_leaf_reg=20.0, random_seed=42, verbose=100,
        allow_writing_files=False, thread_count=6)
    model.fit(train[features], train.control_success, cat_features=cat_indices)

    with tempfile.TemporaryDirectory(prefix="r10_cathetero_") as tmp_name:
        tmp = Path(tmp_name)
        with zipfile.ZipFile(BASE) as z:
            z.extractall(tmp)
        cat_file = "coldstart_catboost.cbm"
        model.save_model(tmp / "model" / cat_file)
        meta_path = tmp / "model" / "coldstart_meta.json"
        spec = json.loads(meta_path.read_text(encoding="utf-8"))
        if spec["feature_cols"] != features or spec["cat_cols"] != cats:
            raise SystemExit("CatBoost and LightGBM expert schemas differ")
        spec.update({"catboost_file": cat_file, "catboost_share": 0.5})
        spec["source"]["heterogeneous_expert"] = {
            "model": "CatBoost depth6 500 rounds, seed42",
            "share_inside_expert": 0.5,
            "validation": ("vs LightGBM expert at w50: pooled +2.112e-5, "
                           "z=1.88; 2023 +6.818e-5; 2024 +3.504e-5"),
            "row_independent": True,
        }
        meta_path.write_text(json.dumps(spec, ensure_ascii=False,
                                        separators=(",", ":")), encoding="utf-8")
        shutil.copyfile(ROOT / "scripts" / "coldstart_runtime.py",
                        tmp / "coldstart_runtime.py")
        req = tmp / "requirements.txt"
        text = req.read_text(encoding="utf-8")
        if "catboost" not in text.lower():
            text = text.rstrip() + "\ncatboost==1.2.8\n"
            req.write_text(text, encoding="utf-8")
        with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(tmp.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    z.write(path, path.relative_to(tmp))
    print(f"built {OUT}; train_R={len(train):,}; features={len(features)}")


if __name__ == "__main__":
    main()

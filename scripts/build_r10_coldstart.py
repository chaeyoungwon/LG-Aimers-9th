"""Train and package the forward-validated cold-start expert over the 1007 model."""
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train_base import (CAT_COLS, PARAMS, add_features,
                            apply_category_maps, build_category_maps)


BASE = ROOT / "artifacts" / "submit_r10_pitcher.zip"
OUT = ROOT / "artifacts" / "submit_r10_pitcher_w50.zip"
TRAIN = ROOT.parent / "open" / "data" / "train.csv"
WEIGHT = 0.50
DROP = {"row_id", "control_success", "pitcher_id", "batter_id"}


def train_expert(df, out_dir):
    r = add_features(df[df.game_type.eq("R")].copy())
    features = [c for c in r.columns if c not in DROP]
    cats = [c for c in CAT_COLS if c in features]
    maps = build_category_maps(r, cats)
    enc = apply_category_maps(r, cats, maps)
    params = dict(PARAMS, seed=42, num_leaves=31, min_data_in_leaf=800,
                  num_threads=6)
    ds = lgb.Dataset(enc[features], label=enc.control_success,
                     categorical_feature=cats, free_raw_data=False)
    model = lgb.train(params, ds, num_boost_round=236)
    filename = "coldstart_lgbm.txt"
    model.save_model(str(out_dir / filename))
    return {"model_file": filename, "weight": WEIGHT,
            "feature_cols": features, "cat_cols": cats, "category_maps": maps,
            "known_r_pitcher_ids": sorted(int(x) for x in r.pitcher_id.unique()),
            "source": {"train_rows": "train <= 2024, game_type=R only",
                       "ids": "pitcher_id and batter_id excluded from model",
                       "gate": "R pitcher absent from train-period R pitcher table",
                       "validation": "forward folds: 2/3 better; pooled gain 4.116e-4, z=12.51",
                       "row_independent": True}}


def main():
    if not BASE.exists():
        raise FileNotFoundError(BASE)
    df = pd.read_csv(TRAIN, encoding="utf-8-sig")
    with tempfile.TemporaryDirectory(prefix="r10_coldstart_") as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(BASE) as z:
            z.extractall(tmp)
        spec = train_expert(df, tmp / "model")
        (tmp / "model" / "coldstart_meta.json").write_text(
            json.dumps(spec, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        shutil.copyfile(ROOT / "scripts" / "coldstart_runtime.py",
                        tmp / "coldstart_runtime.py")
        script_path = tmp / "script.py"
        script = script_path.read_text(encoding="utf-8")
        import_anchor = "from batter_form_runtime import apply_batter_form\n"
        call = 'apply_batter_form(preds, test, "./model")'
        if script.count(call) != 2:
            raise SystemExit("unexpected batter-form anchors")
        script = script.replace(import_anchor, import_anchor +
            "from coldstart_runtime import apply_coldstart_expert\n", 1)
        script = script.replace("return " + call,
            "preds = " + call + "\n        return apply_coldstart_expert(preds, test, \"./model\")", 1)
        script = script.replace("return " + call,
            "preds = " + call + "\n    return apply_coldstart_expert(preds, test, \"./model\")", 1)
        script_path.write_text(script, encoding="utf-8")
        with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(tmp.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    z.write(path, path.relative_to(tmp))
    print(f"built {OUT}; weight={WEIGHT}; known_R_pitchers="
          f"{len(spec['known_r_pitcher_ids'])}")


if __name__ == "__main__":
    main()

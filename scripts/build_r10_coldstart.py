"""Train and package the forward-validated cold-start expert over the 1007 model."""
import argparse
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
TRAIN = ROOT.parent / "open" / "data" / "train.csv"
DEFAULT_WEIGHT = 0.50
REGISTERED = {
    0.50: "submit_r10_pitcher_w50.zip",
    # Rejected on leaderboard: 1010.8164029281 (-5.2857 vs weight=0.50).
    # Kept only to reproduce the failed experiment; do not submit again.
    0.75: "submit_r10_pitcher_w75.zip",
}
LB_RESULT = {0.50: 1016.102132613, 0.75: 1010.8164029281}
OOF_RESULT = {
    0.50: "forward folds: 2/3 better; pooled gain 4.116e-4, z=12.51",
    0.75: "forward folds: 2/3 better; pooled gain 5.219e-4, z=10.58",
}
DROP = {"row_id", "control_success", "pitcher_id", "batter_id"}


def train_expert(df, out_dir, weight, seeds):
    r = add_features(df[df.game_type.eq("R")].copy())
    features = [c for c in r.columns if c not in DROP]
    cats = [c for c in CAT_COLS if c in features]
    maps = build_category_maps(r, cats)
    enc = apply_category_maps(r, cats, maps)
    files = []
    for seed in seeds:
        params = dict(PARAMS, seed=seed, num_leaves=31, min_data_in_leaf=800,
                      num_threads=6)
        ds = lgb.Dataset(enc[features], label=enc.control_success,
                         categorical_feature=cats, free_raw_data=False)
        model = lgb.train(params, ds, num_boost_round=236)
        filename = ("coldstart_lgbm.txt" if len(seeds) == 1
                    else f"coldstart_lgbm_s{seed}.txt")
        model.save_model(str(out_dir / filename))
        files.append(filename)
    return {"model_file": files[0], "model_files": files, "weight": weight,
            "feature_cols": features, "cat_cols": cats, "category_maps": maps,
            "known_r_pitcher_ids": sorted(int(x) for x in r.pitcher_id.unique()),
            "source": {"train_rows": "train <= 2024, game_type=R only",
                       "ids": "pitcher_id and batter_id excluded from model",
                       "gate": "R pitcher absent from train-period R pitcher table",
                       "validation": OOF_RESULT[weight],
                       "leaderboard_bss": LB_RESULT[weight],
                       "decision": ("champion" if weight == DEFAULT_WEIGHT
                                    else "rejected_overmix"),
                       "row_independent": True}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=float, default=DEFAULT_WEIGHT,
                        choices=sorted(REGISTERED))
    parser.add_argument("--seed-ensemble", action="store_true",
                        help="average the preregistered seeds 42/43/44")
    parser.add_argument("--segment-boost", action="store_true",
                        help="use validated row-local runners2/hand11 w60 boost")
    args = parser.parse_args(argv)
    weight = args.weight
    seeds = (42, 43, 44) if args.seed_ensemble else (42,)
    out = (ROOT / "artifacts" / "submit_r10_coldw60.zip"
           if args.segment_boost else
           ROOT / "artifacts" / "submit_r10w50_s3.zip"
           if args.seed_ensemble else ROOT / "artifacts" / REGISTERED[weight])
    if args.seed_ensemble and weight != DEFAULT_WEIGHT:
        raise SystemExit("seed ensemble is registered only for weight=0.50")
    if args.segment_boost and (args.seed_ensemble or weight != DEFAULT_WEIGHT):
        raise SystemExit("segment boost requires the seed42 weight=0.50 champion")
    if not BASE.exists():
        raise FileNotFoundError(BASE)
    df = pd.read_csv(TRAIN, encoding="utf-8-sig")
    with tempfile.TemporaryDirectory(prefix="r10_coldstart_") as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(BASE) as z:
            z.extractall(tmp)
        spec = train_expert(df, tmp / "model", weight, seeds)
        if args.segment_boost:
            spec["segment_boost"] = {
                "weight": 0.60,
                "condition": "num_runners_on == 2 OR (pitcher_hand == 1 AND batter_hand == 1)",
            }
            spec["source"].update({
                "validation": ("3/3 forward folds positive; pooled gain vs w50 "
                               "1.953e-5, z=5.83"),
                "leaderboard_bss": None,
                "decision": "candidate_segment_boost",
                "segment_inputs": "current row official columns only",
            })
        if args.seed_ensemble:
            spec["source"].update({
                "validation": ("seed3 vs seed42 at w50: pooled gain 1.391e-5, "
                               "z=2.07"),
                "leaderboard_bss": 1015.4525572004,
                "decision": "rejected_seed_ensemble",
            })
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
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(tmp.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    z.write(path, path.relative_to(tmp))
    print(f"built {out}; weight={weight}; seeds={seeds}; known_R_pitchers="
          f"{len(spec['known_r_pitcher_ids'])}")


if __name__ == "__main__":
    main()

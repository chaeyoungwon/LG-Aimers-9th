"""Build NN10 + season-form + count/hand OOF-residual correction package."""
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OTHER = ROOT.parent / "LG-AIMERS_9TH"
DATA = ROOT.parent / "open" / "data" / "train.csv"
SRC = ROOT / "candidates" / "nn10_form_a28"
OUT = ROOT / "candidates" / "nn10_form_cells"
ARTIFACT = ROOT / "artifacts" / "submit_nn10_a28_cells_candidate.zip"
TEAM_CACHE = ROOT / "artifacts" / "validation_hetero" / "exp23_cache" / "fold_2023.npz"
NN_CACHE = ROOT / "artifacts" / "validation_nn" / "nn_2023_2024_seed42.npy"
SHRINK_K = 2000.0


def build_table():
    sys.path.insert(0, str(OTHER))
    from src import season_form as sf, team_lgbm as tl

    cols = ["season", "game_type", "pitcher_id", "asof_pitcher_n",
            "asof_pitcher_success_rate", "balls_before", "strikes_before",
            "pitcher_hand", "batter_hand", "control_success"]
    df = pd.read_csv(DATA, encoding="utf-8-sig", usecols=cols)
    inner = df[df.season == 2023].reset_index(drop=True)
    val = df[df.season == 2024].reset_index(drop=True)
    z = np.load(TEAM_CACHE)
    p_inner = {n: z[f"p_inner_{n}"].astype("float64") for n in tl.MODEL_ORDER}
    p_val = {n: z[f"p_val_{n}"].astype("float64") for n in tl.MODEL_ORDER}
    y_inner = inner.control_success.to_numpy(dtype="float64")
    calib = [tl.fit_platt(tl.group_raw(p_inner, g), y_inner)
             for g in tl.ENSEMBLE_GROUPS]
    team = tl.blend(p_val, calib=calib)
    nn = np.load(NN_CACHE).astype("float64")
    if len(nn) != len(val):
        raise SystemExit(f"NN OOF length mismatch: {len(nn)} != {len(val)}")
    form = sf.compute_form_adj(
        val, sf.build_table(df[df.season <= 2023]), m=75, mu=sf.MU_R11C)
    pred = np.clip(0.9 * team + 0.1 * nn + 0.28 * form, 0.0, 1.0)
    residual = val.control_success.to_numpy(dtype="float64") - pred
    is_r = val.game_type.eq("R").to_numpy()
    cell = (val.balls_before.astype(str) + "-" + val.strikes_before.astype(str)
            + "|" + val.pitcher_hand.astype(str) + "-"
            + val.batter_hand.astype(str))
    work = pd.DataFrame({"cell": cell[is_r], "residual": residual[is_r]})
    agg = work.groupby("cell").residual.agg(["sum", "count"])
    global_residual = float(residual[is_r].mean())
    adj = ((agg["sum"] - agg["count"] * global_residual)
           / (agg["count"] + SHRINK_K))
    return {
        "segment": "R", "shrink_k": SHRINK_K,
        "adjustments": {str(k): float(v) for k, v in adj.items()},
        "source": {
            "fit_rows": "2024 R rows only",
            "predictions": "OOF: train<=2023 -> 2024, NN10 + season-form",
            "centering": "subtract global 2024 R residual; interaction only",
            "features": ["balls_before", "strikes_before", "pitcher_hand",
                         "batter_hand"],
            "row_independent": True,
        },
    }


def main():
    spec = build_table()
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(SRC, OUT, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(ROOT / "scripts" / "residual_cell_runtime.py",
                    OUT / "residual_cell_runtime.py")
    script_path = OUT / "script.py"
    script = script_path.read_text(encoding="utf-8")
    anchor = "from season_form_runtime import apply_season_form\n"
    script = script.replace(anchor, anchor +
                            "from residual_cell_runtime import apply_residual_cells\n", 1)
    call = "        preds = apply_season_form(preds, test, MODEL_DIR)\n"
    script = script.replace(call, call +
                            "        preds = apply_residual_cells(preds, test, MODEL_DIR)\n", 1)
    script_path.write_text(script, encoding="utf-8")
    (OUT / "model" / "residual_cells.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(ARTIFACT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                z.write(path, path.relative_to(OUT))
    print(f"built {ARTIFACT}; cells={len(spec['adjustments'])}; "
          f"range=[{min(spec['adjustments'].values()):.6f}, "
          f"{max(spec['adjustments'].values()):.6f}]")


if __name__ == "__main__":
    main()

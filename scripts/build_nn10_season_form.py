"""Build the proven NN10 package with the train-only season-form correction."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "candidates" / "nn10"
OUT = ROOT / "candidates" / "nn10_form_a28"
TRAIN = ROOT.parent / "open" / "data" / "train.csv"

COLS = ["season", "game_type", "pitcher_id", "asof_pitcher_n",
        "asof_pitcher_success_rate", "control_success"]
MU = -0.022624493630570446


def cumulative_success(n, rate):
    n = np.asarray(n, dtype="float64")
    rate = pd.Series(rate).to_numpy(dtype="float64")
    return np.rint(n * np.where(np.isfinite(rate), rate, 0.0))


def build_table(df):
    n = df["asof_pitcher_n"].to_numpy(dtype="float64")
    pid = df["pitcher_id"].to_numpy(dtype="int64")
    s = cumulative_success(n, df["asof_pitcher_success_rate"])
    y = df["control_success"].to_numpy(dtype="float64")
    order = np.lexsort((n, pid))
    pid_sorted = pid[order]
    last = np.r_[pid_sorted[1:] != pid_sorted[:-1], True]
    idx = order[last]
    return {"pitcher_ids": pid_sorted[last].astype(int).tolist(),
            "n0": (n[idx] + 1.0).tolist(),
            "s0": (s[idx] + y[idx]).tolist()}


def form_values(df, table, mu=0.0):
    ids = np.asarray(table["pitcher_ids"], dtype="int64")
    n0t = np.asarray(table["n0"], dtype="float64")
    s0t = np.asarray(table["s0"], dtype="float64")
    pid = df["pitcher_id"].to_numpy(dtype="int64")
    pos = np.searchsorted(ids, pid)
    pc = np.clip(pos, 0, len(ids) - 1)
    seen = (pos < len(ids)) & (ids[pc] == pid)
    n = df["asof_pitcher_n"].to_numpy(dtype="float64")
    s = cumulative_success(n, df["asof_pitcher_success_rate"])
    n0, s0 = n0t[pc], s0t[pc]
    p0 = s0 / n0
    ns, ss = n - n0, s - s0
    active = seen & (ns > 0) & df["game_type"].eq("R").to_numpy()
    out = np.zeros(len(df), dtype="float64")
    out[active] = ((ss[active] + 75.0 * p0[active])
                   / (ns[active] + 75.0) - p0[active] - mu)
    return out


def main():
    raw = pd.read_csv(TRAIN, encoding="utf-8-sig", usecols=COLS)
    cut = build_table(raw[raw["season"] <= 2023])
    v24 = raw[raw["season"] == 2024]
    # The deployed mean includes seen R rows with Nseason=0 (their form is zero).
    uncentered = form_values(v24, cut, mu=0.0)
    ids = np.asarray(cut["pitcher_ids"], dtype="int64")
    pid = v24["pitcher_id"].to_numpy(dtype="int64")
    pos = np.searchsorted(ids, pid)
    pc = np.clip(pos, 0, len(ids) - 1)
    seen_r = ((pos < len(ids)) & (ids[pc] == pid)
              & v24["game_type"].eq("R").to_numpy())
    got_mu = float(uncentered[seen_r].mean())
    if abs(got_mu - MU) > 1e-9:
        raise SystemExit(f"mu mismatch: {got_mu!r} != {MU!r}")

    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(SRC, OUT)
    shutil.copyfile(ROOT / "scripts" / "season_form_runtime.py",
                    OUT / "season_form_runtime.py")
    script_path = OUT / "script.py"
    script = script_path.read_text(encoding="utf-8")
    import_anchor = "import pandas as pd\n"
    script = script.replace(import_anchor,
                            import_anchor + "from season_form_runtime import apply_season_form\n",
                            1)
    blend_anchor = "        preds = (1.0 - nn_weight) * preds + nn_weight * nn_preds\n"
    if blend_anchor not in script:
        raise SystemExit("script blend anchor missing")
    script = script.replace(blend_anchor,
                            blend_anchor + "        preds = apply_season_form(preds, test, MODEL_DIR)\n",
                            1)
    script_path.write_text(script, encoding="utf-8")
    table = build_table(raw)
    spec = {"alpha": 0.28, "m": 75.0, "mu": MU, "segment": "R",
            **table,
            "source": {"base": "NN10, leaderboard 957.6295435994",
                       "table": "train <= 2024 only",
                       "mu": "train cutoff <=2023 evaluated on 2024 R seen rows",
                       "row_independent": True}}
    path = OUT / "model" / "season_form.json"
    path.write_text(json.dumps(spec, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8")
    print(f"built {OUT}; pitchers={len(table['pitcher_ids'])}; mu={got_mu!r}")


if __name__ == "__main__":
    main()

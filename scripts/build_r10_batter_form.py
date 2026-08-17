"""Put the validated batter-form layer on the rebuilt R10 + pitcher-form ZIP."""
import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts" / "submit_r10_form_a28_rebuild.zip"
TRAIN = ROOT.parent / "open" / "data" / "train.csv"
M = 50.0
DEFAULT_ALPHA = 0.11250648296393913
REGISTERED = {
    DEFAULT_ALPHA: "submit_r10_pitcher.zip",
    # Rejected on leaderboard: 994.9143132327 (-12.5901 vs DEFAULT_ALPHA).
    # Kept only to reproduce the failed experiment; do not submit again.
    0.20: "submit_r10_pitcher_a20.zip",
}
OOF_GAIN = {DEFAULT_ALPHA: 25.67, 0.20: 32.98}
LB_RESULT = {DEFAULT_ALPHA: 1007.50439, 0.20: 994.9143132327}


def build_table(df):
    n = df["asof_batter_n"].to_numpy(dtype="float64")
    rate = df["asof_batter_success_rate"].to_numpy(dtype="float64")
    s = np.rint(n * np.where(np.isfinite(rate), rate, 0.0))
    bid = df["batter_id"].to_numpy(dtype="int64")
    y = df["control_success"].to_numpy(dtype="float64")
    order = np.lexsort((n, bid))
    ordered_ids = bid[order]
    last = np.r_[ordered_ids[1:] != ordered_ids[:-1], True]
    idx = order[last]
    return {"batter_ids": ordered_ids[last].astype(int).tolist(),
            "n0": (n[idx] + 1.0).tolist(),
            "s0": (s[idx] + y[idx]).tolist()}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        choices=sorted(REGISTERED))
    args = parser.parse_args(argv)
    alpha = args.alpha
    out = ROOT / "artifacts" / REGISTERED[alpha]
    if not BASE.exists():
        raise FileNotFoundError(BASE)
    cols = ["batter_id", "asof_batter_n", "asof_batter_success_rate",
            "control_success"]
    train = pd.read_csv(TRAIN, encoding="utf-8-sig", usecols=cols)
    spec = {"alpha": alpha, "m": M, "segment": "R", **build_table(train),
            "source": {"base": BASE.name,
                       "coefficient_fit": "2022 and 2023 forward OOF only",
                       "evaluation": (f"2024 team-proxy forward OOF, "
                                      f"+{OOF_GAIN[alpha]:.2f} BSS-equivalent"),
                       "leaderboard_bss": LB_RESULT[alpha],
                       "decision": ("champion" if alpha == DEFAULT_ALPHA
                                    else "rejected_overcorrection"),
                       "row_independent": True}}

    with tempfile.TemporaryDirectory(prefix="r10_batter_form_") as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(BASE) as z:
            z.extractall(tmp)
        (tmp / "model" / "batter_form.json").write_text(
            json.dumps(spec, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8")
        (tmp / "batter_form_runtime.py").write_text(
            (ROOT / "scripts" / "batter_form_runtime.py").read_text(encoding="utf-8"),
            encoding="utf-8")
        script_path = tmp / "script.py"
        script = script_path.read_text(encoding="utf-8")
        import_anchor = "import pandas as pd\n"
        call = ("apply_season_form(preds, test, "
                "meta.get(\"season_form\"), verbose)")
        member_anchor = "\n        return " + call
        plain_anchor = "\n    return " + call
        if script.count(member_anchor) != 1 or script.count(plain_anchor) != 1:
            raise SystemExit("unexpected R10 season-form return anchors")
        script = script.replace(
            import_anchor,
            import_anchor + "from batter_form_runtime import apply_batter_form\n", 1)
        script = script.replace(
            member_anchor,
            "\n        preds = " + call + "\n"
            "        return apply_batter_form(preds, test, \"./model\")", 1)
        script = script.replace(
            plain_anchor,
            "\n    preds = " + call + "\n"
            "    return apply_batter_form(preds, test, \"./model\")", 1)
        script_path.write_text(script, encoding="utf-8")

        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(tmp.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    z.write(path, path.relative_to(tmp))
    print(f"built {out}; batters={len(spec['batter_ids'])}; "
          f"M={M}; alpha={alpha}")


if __name__ == "__main__":
    main()

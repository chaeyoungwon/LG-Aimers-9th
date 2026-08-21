"""Add a stable train-only pitcher x base-pressure residual correction."""
import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts" / "submit_1031_et25.zip"
TRAIN = ROOT.parent / "open" / "data" / "train.csv"
OUT = ROOT / "artifacts" / "submit_1032_basepressure.zip"
K = 5000.0


def base_pressure(rows):
    n = rows.num_runners_on.to_numpy(dtype="int64")
    return np.where(n == 0, 0, np.where(n == 1, 1, 2))


def build_table(train):
    frames = []
    regular = train[train.game_type.eq("R")].copy()
    for _, rows in regular.groupby("season", sort=True):
        y = rows.control_success.to_numpy(dtype="float64")
        frames.append(pd.DataFrame({
            "pid": rows.pitcher_id.to_numpy(dtype="int64"),
            "g": base_pressure(rows),
            "r": y - y.mean(),
        }))
    data = pd.concat(frames, ignore_index=True)
    cell = data.groupby(["pid", "g"]).r.agg(["sum", "size"])
    main = data.groupby("pid").r.agg(["sum", "size"]).rename(
        columns={"sum": "ps", "size": "pn"})
    joined = cell.join(main, on="pid")
    expected = joined.ps * joined["size"] / joined.pn
    value = (joined["sum"] - expected) / (joined["size"] + K)
    return {f"{pid}|{group}": float(x)
            for (pid, group), x in value.items()}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    train = pd.read_csv(args.train, encoding="utf-8-sig")
    table = build_table(train)
    with tempfile.TemporaryDirectory(prefix="basepressure_") as name:
        work = Path(name)
        with zipfile.ZipFile(args.base) as archive:
            archive.extractall(work)
        spec_path = work / "model" / "corrections.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if any(item["name"] == "epsilon_base_pressure" for item in spec["items"]):
            raise SystemExit("base-pressure correction already exists")
        spec["items"].append({
            "name": "epsilon_base_pressure", "kind": "interaction",
            "key": "base_pressure", "k": K, "table": table,
            "source": ("train R rows only; target centered within season; "
                       "pitcher main effect removed"),
            "validation": ("2022/2023/2024 +5.306e-6/+1.668e-6/+6.539e-6; "
                           "pooled +4.513e-6, z=2.39"),
        })
        spec_path.write_text(json.dumps(spec, ensure_ascii=False,
                                        separators=(",", ":")), encoding="utf-8")
        runtime = work / "corrections_runtime.py"
        code = runtime.read_text(encoding="utf-8")
        anchor = '    if name == "leverage":\n'
        addition = ('    if name == "base_pressure":\n'
                    '        n = df["num_runners_on"].to_numpy(dtype="int64")\n'
                    '        return np.where(n == 0, 0, np.where(n == 1, 1, 2))\n')
        if code.count(anchor) != 1:
            raise SystemExit("correction runtime key anchor missing")
        runtime.write_text(code.replace(anchor, addition + anchor, 1),
                           encoding="utf-8")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(work.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(work))
    print(f"built {args.out}; table={len(table):,}; k={K:g}")


if __name__ == "__main__":
    main()

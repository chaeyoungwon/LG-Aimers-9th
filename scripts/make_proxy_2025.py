"""진짜 2025처럼 생긴 평가 프록시를 만들어 제출 ZIP의 예측 수준을 비교한다.

2024 행의 ``season``만 2025로 바꾸면 ``asof_n < n0(2024말)``이 되어 시즌 상태
피처가 전부 결측이다. 대신 2024 행의 시즌 내 누적분을 2024말 경계 위로 옮긴다.

    ns       = asof_n - n0(2023말)
    asof_n'  = n0(2024말) + ns
    count'   = c0(2024말) + (count - c0(2023말))
    rate'    = count' / asof_n'

경계표는 공식 학습 데이터로만 만들며 각 프록시 행도 자기 as-of 값만 사용한다.

예시:
  DYLD_LIBRARY_PATH=/path/to/.venv/lib OMP_NUM_THREADS=1 \
    .venv/bin/python scripts/make_proxy_2025.py \
    --train /path/to/train.csv artifacts/submit_season_state.zip
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import season_state as ss  # noqa: E402

DEFAULT_TRAIN = ROOT / "data" / "train.csv"
DEFAULT_ZIPS = (
    ROOT / "artifacts" / "submit_corrections.zip",
    ROOT / "artifacts" / "submit_season_state.zip",
)


def lookup(tab, ids_query):
    order = np.argsort(tab["ids"])
    ids = tab["ids"][order]
    key = np.where(np.isfinite(ids_query), ids_query, -1).astype("int64")
    pos = np.searchsorted(ids, key)
    clip = np.clip(pos, 0, len(ids) - 1)
    seen = (pos < len(ids)) & (ids[clip] == key)
    return seen, order, clip


def build_proxy(df):
    """2024 행의 시즌 내 상태를 2025 누적값 위로 이식한다."""
    seasons = sorted(int(s) for s in df.season.unique())
    if seasons[-1] != 2024:
        raise ValueError(f"마지막 학습 시즌이 2024가 아니다: {seasons}")
    prior = [s for s in seasons if s < 2024]
    boundary_2023 = ss.build_boundary(df, prior)
    boundary_2024 = ss.build_boundary(df, seasons)

    out = df[df.season == 2024].drop(columns=["control_success"]).reset_index(drop=True)
    out["season"] = 2025
    for name, idcol, ncol, rates in ss.GROUPS:
        old, new = boundary_2023[name], boundary_2024[name]
        query = out[idcol].to_numpy(dtype="float64")
        seen_old, order_old, clip_old = lookup(old, query)
        seen_new, order_new, clip_new = lookup(new, query)
        both = seen_old & seen_new

        n = out[ncol].to_numpy(dtype="float64")
        n0_old = np.where(seen_old, old["n0"][order_old][clip_old], np.nan)
        n0_new = np.where(seen_new, new["n0"][order_new][clip_new], np.nan)
        n_season = n - n0_old
        valid = both & np.isfinite(n_season) & (n_season > 0)
        shifted_n = np.where(valid, n0_new + n_season, n)

        for short, col in rates.items():
            rate = out[col].to_numpy(dtype="float64")
            count = np.rint(n * np.where(np.isfinite(rate), rate, 0.0))
            c0_old = np.where(
                seen_old, old["c0"][short][order_old][clip_old], np.nan
            )
            c0_new = np.where(
                seen_new, new["c0"][short][order_new][clip_new], np.nan
            )
            shifted_count = c0_new + (count - c0_old)
            with np.errstate(invalid="ignore", divide="ignore"):
                shifted_rate = shifted_count / shifted_n
            out[col] = np.where(valid & np.isfinite(shifted_rate), shifted_rate, rate)
        out[ncol] = shifted_n

    out["row_id"] = [f"TEST_{i:06d}" for i in range(len(out))]
    return out, boundary_2024


def predict_zip(zip_path, proxy):
    """배포 ZIP을 그대로 풀어 전체 ``predict`` 경로를 실행한다."""
    zip_path = Path(zip_path)
    with tempfile.TemporaryDirectory(prefix="proxy2025_") as tmp:
        work = Path(tmp)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(work)
        cwd = os.getcwd()
        os.chdir(work)
        sys.path.insert(0, str(work))
        try:
            module_name = f"proxy_submit_{abs(hash(zip_path.resolve()))}"
            spec = importlib.util.spec_from_file_location(module_name, work / "script.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            meta = json.loads((work / "model" / "meta.json").read_text("utf-8"))
            models = module.load_models("./model", meta)
            pred = np.asarray(
                module.predict(proxy, models, meta, verbose=False), dtype="float64"
            )
        finally:
            sys.path.pop(0)
            os.chdir(cwd)
    return pred, meta


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("zips", type=Path, nargs="*")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.train, encoding="utf-8-sig")
    proxy, boundary_2024 = build_proxy(df)
    print(f"proxy rows {len(proxy):,}", flush=True)

    state = ss.add_features(proxy, boundary_2024)
    print(
        f"  상태피처 결측 아닌 비율 {state.cur_p_succ.notna().mean():.3f} "
        f"(실제 2024에서는 0.801)"
    )
    print(f"  cur_p_n 중앙값 {state.cur_p_n.median():.0f}", flush=True)

    for zip_path in args.zips or DEFAULT_ZIPS:
        pred, meta = predict_zip(zip_path, proxy)
        shift = float(meta.get("final_logit_shift", 0.0))
        logit = np.log(np.clip(pred, 1e-6, 1 - 1e-6) / np.clip(1 - pred, 1e-6, 1))
        unshifted = 1 / (1 + np.exp(-(logit - shift)))
        print(
            f"{zip_path.name:>28}: 프록시2025 평균 {pred.mean():.6f} "
            f"(shift 제거 시 {unshifted.mean():.6f})",
            flush=True,
        )


if __name__ == "__main__":
    main()

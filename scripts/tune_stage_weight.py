"""18차 ZIP의 ours/team 스테이지 비중을 프록시 2025에서 사전 스크리닝한다.

멤버 예측은 한 번만 계산하고, 후처리까지 포함한 최종 예측을 스테이지 비중별로
재조합한다. 예측 차이 기반 2σ와 평균 수준 이동을 보고 제출할 가치가 있는 간격인지
판정한다. ``--build``를 주면 메타데이터만 바꾼 후보 ZIP도 만든다.

예시:
  DYLD_LIBRARY_PATH=/path/to/.venv/lib OMP_NUM_THREADS=1 \
    .venv/bin/python scripts/tune_stage_weight.py \
    --train /path/to/train.csv --weights 0.5 0.6 0.7
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_proxy_2025 import build_proxy  # noqa: E402

DEFAULT_BASE = ROOT / "artifacts" / "submit_season_state.zip"
DEFAULT_TRAIN = ROOT / "data" / "train.csv"
DEFAULT_REPORT = ROOT / "artifacts" / "backtest" / "stage_weight_screen.json"
BRIER_BASELINE = 0.2498


def validate_weight(weight):
    weight = float(weight)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError(f"ours 비중은 [0,1]이어야 한다: {weight!r}")
    return weight


def load_submission(work):
    spec = importlib.util.spec_from_file_location("stage_weight_submit", work / "script.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meta = json.loads((work / "model" / "meta.json").read_text("utf-8"))
    models = module.load_models("./model", meta)
    return module, meta, models


def member_predictions(module, meta, models, proxy):
    """배포본 ``predict_blend``의 멤버 예측 부분을 그대로 재현한다."""
    frame = module.add_state(proxy.copy(), "./model")
    for col, levels in meta["cat_levels"].items():
        frame[col] = pd.Categorical(frame[col], categories=levels)

    lookups = meta.get("lookups")
    out = {}
    for spec, member_models in zip(meta["members"], models):
        X = module.build_submit_features(frame, spec, lookups)
        pred = module._mean_proba(list(member_models), X)
        out[spec["name"]] = module.apply_calibration(
            pred, spec.get("calibration"), frame
        )
    return frame, out


def stage_predictions(module, meta, frame, member_pred):
    stages = []
    for stage in meta["blend"]["stages"]:
        weights = module.normalize_weights(stage.get("weights"), len(stage["members"]))
        pred = np.average(
            [member_pred[name] for name in stage["members"]],
            axis=0,
            weights=weights,
        )
        stages.append(module.apply_calibration(pred, stage.get("calibration"), frame))
    return stages


def final_prediction(module, meta, frame, stages, ours_weight):
    """스테이지 혼합 뒤의 배포 후처리까지 전부 적용한다."""
    pred = ours_weight * stages[0] + (1.0 - ours_weight) * stages[1]
    pred = module.apply_calibration(pred, meta["blend"].get("calibration"), frame)
    pred = module.apply_season_form(pred, frame, meta.get("season_form"), False)
    pred = module.apply_batter_form(pred, frame, "./model")
    pred = module.apply_coldstart_expert(pred, frame, "./model")
    pred = module.apply_corrections(pred, frame, "./model")
    shift = float(meta.get("final_logit_shift", 0.0))
    return module.apply_logit_shift(pred, shift) if shift else pred


def two_sigma(base, candidate):
    delta = np.asarray(candidate) - np.asarray(base)
    return float(
        2 * 100000 * np.sqrt(np.sum(delta**2))
        / (len(delta) * BRIER_BASELINE)
    )


def content_hashes(work):
    return {
        str(path.relative_to(work)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in work.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def build_candidate(base, out, weight):
    """원본 ZIP에서 meta의 스테이지 비중만 바꾼 후보를 만든다."""
    with tempfile.TemporaryDirectory(prefix="stage_candidate_") as tmp:
        work = Path(tmp)
        with zipfile.ZipFile(base) as archive:
            archive.extractall(work)
        before = content_hashes(work)
        meta_path = work / "model" / "meta.json"
        meta = json.loads(meta_path.read_text("utf-8"))
        old = list(meta["blend"]["stage_weights"])
        meta["blend"]["stage_weights"] = [weight, 1.0 - weight]
        meta["blend"]["composition"]["w_ours"] = weight
        meta["blend"]["composition"]["w_team"] = 1.0 - weight
        meta["lb_tuning"] = {
            "handle": "stage_weight_ours",
            "base": base.name,
            "old_stage_weights": old,
            "new_stage_weights": [weight, 1.0 - weight],
            "only_meta_changed": True,
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )
        after = content_hashes(work)
        changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
        if changed != ["model/meta.json"]:
            raise RuntimeError(f"meta 외 파일도 바뀌었다: {changed}")

        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(work.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(work))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.5, 0.6, 0.7])
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args(argv)
    weights = [validate_weight(w) for w in args.weights]

    df = pd.read_csv(args.train, encoding="utf-8-sig")
    proxy, _ = build_proxy(df)
    print(f"proxy rows {len(proxy):,}", flush=True)

    with tempfile.TemporaryDirectory(prefix="stage_screen_") as tmp:
        work = Path(tmp)
        with zipfile.ZipFile(args.base) as archive:
            archive.extractall(work)
        cwd = os.getcwd()
        os.chdir(work)
        sys.path.insert(0, str(work))
        try:
            module, meta, models = load_submission(work)
            current = float(meta["blend"]["stage_weights"][0])
            frame, member_pred = member_predictions(module, meta, models, proxy)
            stages = stage_predictions(module, meta, frame, member_pred)
            base_pred = final_prediction(module, meta, frame, stages, current)
            rows = []
            print(
                f"baseline ours={current:.3f}: mean={base_pred.mean():.6f}", flush=True
            )
            for weight in weights:
                pred = final_prediction(module, meta, frame, stages, weight)
                sigma = two_sigma(base_pred, pred)
                row = {
                    "ours_weight": weight,
                    "team_weight": 1.0 - weight,
                    "mean": float(pred.mean()),
                    "mean_shift": float(pred.mean() - base_pred.mean()),
                    "max_abs_prediction_delta": float(np.max(np.abs(pred - base_pred))),
                    "two_sigma_bss": sigma,
                }
                rows.append(row)
                print(
                    f"ours={weight:.3f}: mean={row['mean']:.6f} "
                    f"shift={row['mean_shift']:+.6f} "
                    f"max|dp|={row['max_abs_prediction_delta']:.6f} "
                    f"~2sigma={sigma:.2f}",
                    flush=True,
                )
        finally:
            sys.path.pop(0)
            os.chdir(cwd)

    report = {
        "base": args.base.name,
        "baseline_ours_weight": current,
        "baseline_mean": float(base_pred.mean()),
        "proxy_rows": len(proxy),
        "candidates": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")

    if args.build:
        for row in rows:
            weight = row["ours_weight"]
            tag = f"{weight:.3f}".replace(".", "p")
            out = ROOT / "artifacts" / f"submit_stage_ours_w{tag}.zip"
            build_candidate(args.base, out, weight)
            print(f"built {out}", flush=True)


if __name__ == "__main__":
    main()

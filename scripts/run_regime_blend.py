"""Run nested temporal diagnostics and shrunk regime-blend experiments.

The script consumes precomputed temporal OOF predictions.  It never reads
test.csv and never fits any value from validation labels onto the same fold's
validation predictions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.regime_blend import (  # noqa: E402
    SUPPORTED_REGIMES,
    brier_score,
    bss_delta_from_brier_gain,
    constrained_brier_weights,
    fit_global_blend,
    fit_regime_blend,
    member_diagnostics,
)


DEFAULT_RIDGES = (0.0, 1e-6, 1e-5)
DEFAULT_TAUS = (2000.0, 10000.0, 50000.0)


def _float_grid(text: str) -> tuple:
    try:
        values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values or any(not np.isfinite(v) or v < 0.0 for v in values):
        raise argparse.ArgumentTypeError("grid values must be finite and non-negative")
    return values


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _required_array(artifact: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in artifact:
        raise ValueError(f"OOF artifact is missing required array {key!r}")
    return np.asarray(artifact[key])


def _prediction_matrix(
    artifact: Mapping[str, np.ndarray], split: str, names: Sequence[str]
) -> np.ndarray:
    return np.column_stack(
        [_required_array(artifact, f"p_{split}_{name}").astype("float64") for name in names]
    )


def _features(
    artifact: Mapping[str, np.ndarray], split: str, names: Sequence[str]
) -> Dict[str, np.ndarray]:
    return {name: _required_array(artifact, f"x_{split}_{name}") for name in names}


def _optional_prediction(
    artifact: Mapping[str, np.ndarray], key: str, fallback: np.ndarray
) -> np.ndarray:
    return np.asarray(artifact[key], dtype="float64") if key in artifact else fallback


def _fixed_blend(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.zeros(matrix.shape[0], dtype="float64")
    for column, weight in enumerate(weights):
        out += matrix[:, column] * float(weight)
    return np.clip(out, 0.0, 1.0)


def _fold_data(manifest_dir: Path, fold_spec: Mapping[str, object], manifest: dict) -> dict:
    path = manifest_dir / str(fold_spec["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as loaded:
        artifact = {key: loaded[key] for key in loaded.files}

    members = tuple(manifest["members"])
    blend_members = tuple(manifest["blend_members"])
    feature_columns = tuple(manifest["feature_columns"])
    anchor = np.asarray(manifest["anchor_weights"], dtype="float64")
    anchor /= anchor.sum()
    inner_blend = _prediction_matrix(artifact, "inner", blend_members)
    val_blend = _prediction_matrix(artifact, "val", blend_members)
    champion_inner_fallback = _fixed_blend(inner_blend, anchor)
    champion_val_fallback = _fixed_blend(val_blend, anchor)

    data = {
        "path": str(path),
        "inner_season": int(fold_spec["inner_season"]),
        "validation_season": int(fold_spec["validation_season"]),
        "y_inner": _required_array(artifact, "y_inner").astype("float64"),
        "y_val": _required_array(artifact, "y_val").astype("float64"),
        "inner_members": _prediction_matrix(artifact, "inner", members),
        "val_members": _prediction_matrix(artifact, "val", members),
        "inner_blend": inner_blend,
        "val_blend": val_blend,
        "inner_features": _features(artifact, "inner", feature_columns),
        "val_features": _features(artifact, "val", feature_columns),
        "champion_inner": _optional_prediction(
            artifact, "p_inner_champion_base", champion_inner_fallback
        ),
        "champion_val": _optional_prediction(
            artifact, "p_val_champion_base", champion_val_fallback
        ),
        "form_offset_inner": None,
        "form_offset_val": None,
        "champion_form_val": None,
    }
    if "form_offset_inner" in artifact and "form_offset_val" in artifact:
        data["form_offset_inner"] = np.asarray(artifact["form_offset_inner"], dtype="float64")
        data["form_offset_val"] = np.asarray(artifact["form_offset_val"], dtype="float64")
        fallback_form = np.clip(
            data["champion_val"] + data["form_offset_val"], 0.0, 1.0
        )
        data["champion_form_val"] = _optional_prediction(
            artifact, "p_val_champion_form", fallback_form
        )

    for split in ("inner", "val"):
        n_rows = len(data[f"y_{split}"])
        for key in (f"{split}_members", f"{split}_blend"):
            if len(data[key]) != n_rows:
                raise ValueError(f"{path}: {key} length mismatch")
        for name, values in data[f"{split}_features"].items():
            if len(values) != n_rows:
                raise ValueError(f"{path}: feature {split}/{name} length mismatch")
    anchor_parity = float(
        np.max(np.abs(champion_val_fallback - data["champion_val"]))
    )
    data["anchor_parity_max_abs"] = anchor_parity
    return data


def _score_candidate(
    name: str,
    fold: dict,
    prediction: np.ndarray,
    fitted: dict,
) -> dict:
    y = fold["y_val"]
    champion = fold["champion_val"]
    champion_brier = brier_score(y, champion)
    candidate_brier = brier_score(y, prediction)
    gain = champion_brier - candidate_brier
    result = {
        "method": name,
        "validation_season": fold["validation_season"],
        "n": len(y),
        "champion_brier": champion_brier,
        "candidate_brier": candidate_brier,
        "brier_gain": gain,
        "bss_delta": bss_delta_from_brier_gain(y, gain),
        "weights": fitted,
        "form": None,
    }
    if fold["form_offset_val"] is not None:
        new_form = np.clip(prediction + fold["form_offset_val"], 0.0, 1.0)
        champion_form = fold["champion_form_val"]
        champion_form_brier = brier_score(y, champion_form)
        candidate_form_brier = brier_score(y, new_form)
        form_gain = champion_form_brier - candidate_form_brier
        result["form"] = {
            "champion_brier": champion_form_brier,
            "candidate_brier": candidate_form_brier,
            "brier_gain": form_gain,
            "bss_delta": bss_delta_from_brier_gain(y, form_gain),
        }
    return result


def _aggregate(
    fold_results: Sequence[dict], max_worst_loss: float
) -> list:
    by_method: Dict[str, list] = {}
    for row in fold_results:
        by_method.setdefault(row["method"], []).append(row)
    summaries = []
    for method, rows in sorted(by_method.items()):
        rows = sorted(rows, key=lambda row: row["validation_season"])
        gains = np.asarray([row["brier_gain"] for row in rows], dtype="float64")
        weights = np.asarray([row["n"] for row in rows], dtype="float64")
        weight_arrays = np.asarray(
            [row["weights"]["global_weights"] for row in rows], dtype="float64"
        )
        ranges = np.ptp(weight_arrays, axis=0)
        form_rows = [row["form"] for row in rows]
        has_form = all(row is not None for row in form_rows)
        form_gains = (
            np.asarray([row["brier_gain"] for row in form_rows], dtype="float64")
            if has_form
            else None
        )
        base_pass = (
            gains[-1] > 0.0
            and int((gains > 0.0).sum()) >= 2
            and float(np.average(gains, weights=weights)) > 0.0
            and float(gains.min()) >= -max_worst_loss
        )
        form_pass = True if form_gains is None else (
            form_gains[-1] > 0.0
            and int((form_gains > 0.0).sum()) >= 2
            and float(np.average(form_gains, weights=weights)) > 0.0
            and float(form_gains.min()) >= -max_worst_loss
        )
        summaries.append(
            {
                "method": method,
                "folds": rows,
                "mean_brier_gain": float(gains.mean()),
                "pooled_brier_gain": float(np.average(gains, weights=weights)),
                "worst_brier_gain": float(gains.min()),
                "recent_brier_gain": float(gains[-1]),
                "improved_folds": int((gains > 0.0).sum()),
                "form_mean_brier_gain": None if form_gains is None else float(form_gains.mean()),
                "form_worst_brier_gain": None if form_gains is None else float(form_gains.min()),
                "form_recent_brier_gain": None if form_gains is None else float(form_gains[-1]),
                "weight_range": ranges.tolist(),
                "weight_instability_warning": bool(np.any(ranges > 0.20)),
                "verdict": "pass" if base_pass and form_pass else "reject",
            }
        )
    return summaries


def _diagnostics_markdown(diagnostics: Sequence[dict]) -> str:
    lines = [
        "# Member prediction diagnostics",
        "",
        "All values are from outer validation predictions. Full segment and pairwise tables are in `member_diagnostics.json`.",
        "",
    ]
    for fold in diagnostics:
        body = fold["diagnostics"]
        lines.extend(
            [
                f"## Validation {fold['validation_season']}",
                "",
                f"Rows: {body['n']:,}",
                "",
                "| member | overall Brier |",
                "| --- | ---: |",
            ]
        )
        for name in body["members"]:
            lines.append(f"| {name} | {body['overall_brier'][name]:.10f} |")
        lines.extend(["", "Prediction correlation:", "", "| member | " + " | ".join(body["members"]) + " |"])
        lines.append("| --- | " + " | ".join(["---:"] * len(body["members"])) + " |")
        for name, row in zip(body["members"], body["prediction_correlation"]):
            cells = ["NA" if value is None else f"{value:.6f}" for value in row]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def _results_markdown(summaries: Sequence[dict]) -> str:
    if not summaries:
        return "# Regime blend results\n\nNo candidates were evaluated.\n"
    seasons = [row["validation_season"] for row in summaries[0]["folds"]]
    header = "| method | " + " | ".join(str(year) for year in seasons)
    header += " | mean | worst | recent | stable weights | verdict |"
    align = "| --- | " + " | ".join(["---:"] * (len(seasons) + 3))
    align += " | :---: | :---: |"
    lines = [
        "# Regime-aware blend temporal stability",
        "",
        "A positive number is improvement over the existing champion. Each fold cell is `Brier gain / BSS delta`.",
        "",
        header,
        align,
    ]
    for summary in summaries:
        cells = [
            f"{row['brier_gain']:+.3e} / {row['bss_delta']:+.3f}"
            for row in summary["folds"]
        ]
        stable = "warn" if summary["weight_instability_warning"] else "yes"
        lines.append(
            "| " + summary["method"] + " | " + " | ".join(cells)
            + f" | {summary['mean_brier_gain']:+.3e}"
            + f" | {summary['worst_brier_gain']:+.3e}"
            + f" | {summary['recent_brier_gain']:+.3e}"
            + f" | {stable} | {summary['verdict']} |"
        )
    lines.extend(
        [
            "",
            "Methods marked `pass` are research candidates only. Deployment additionally requires form/coldstart composition parity, row-independence, and package smoke tests.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    manifest_path: Path,
    output_dir: Path,
    *,
    ridges: Iterable[float] = DEFAULT_RIDGES,
    taus: Iterable[float] = DEFAULT_TAUS,
    min_samples: int = 500,
    max_worst_loss: float = 1e-5,
) -> dict:
    ridges = tuple(ridges)
    taus = tuple(taus)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"members", "blend_members", "anchor_weights", "feature_columns", "folds"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"manifest is missing keys: {missing}")
    if "regimes" in manifest and tuple(manifest["regimes"]) != SUPPORTED_REGIMES:
        raise ValueError(f"manifest regimes must be {SUPPORTED_REGIMES}")

    folds = [
        _fold_data(manifest_path.parent, fold_spec, manifest)
        for fold_spec in manifest["folds"]
    ]
    validation_seasons = [fold["validation_season"] for fold in folds]
    if validation_seasons != [2022, 2023, 2024]:
        raise ValueError(
            f"expected validation seasons [2022, 2023, 2024], got {validation_seasons}"
        )

    member_names = tuple(manifest["members"])
    blend_names = tuple(manifest["blend_members"])
    anchor = np.asarray(manifest["anchor_weights"], dtype="float64")
    diagnostics = []
    fold_results = []
    fitted_artifacts = {}
    for fold in folds:
        diagnostic_predictions = {
            name: fold["val_members"][:, i] for i, name in enumerate(member_names)
        }
        for i, name in enumerate(blend_names):
            diagnostic_predictions[name] = fold["val_blend"][:, i]
        body = member_diagnostics(
            diagnostic_predictions, fold["y_val"], fold["val_features"]
        )
        diagnostics.append(
            {
                "validation_season": fold["validation_season"],
                "anchor_parity_max_abs": fold["anchor_parity_max_abs"],
                "diagnostics": body,
            }
        )

        for ridge in ridges:
            global_model = fit_global_blend(
                fold["inner_blend"],
                fold["y_inner"],
                blend_names,
                anchor=anchor,
                ridge=ridge,
            )
            global_name = f"global_rho={ridge:g}"
            fitted_artifacts.setdefault(global_name, {})[str(fold["validation_season"])] = global_model.to_dict()
            fold_results.append(
                _score_candidate(
                    global_name,
                    fold,
                    global_model.predict(fold["val_blend"]),
                    global_model.to_dict(),
                )
            )
            for regime in SUPPORTED_REGIMES:
                for tau in taus:
                    model = fit_regime_blend(
                        fold["inner_blend"],
                        fold["y_inner"],
                        fold["inner_features"],
                        blend_names,
                        regime,
                        anchor=anchor,
                        ridge=ridge,
                        tau=tau,
                        min_samples=min_samples,
                    )
                    name = f"regime={regime},rho={ridge:g},tau={tau:g}"
                    fitted = model.to_dict()
                    fitted_artifacts.setdefault(name, {})[
                        str(fold["validation_season"])
                    ] = fitted
                    prediction = model.predict(
                        fold["val_blend"], fold["val_features"]
                    )
                    fold_results.append(_score_candidate(name, fold, prediction, fitted))

    summaries = _aggregate(fold_results, max_worst_loss=max_worst_loss)
    pooled_predictions = np.concatenate([fold["val_blend"] for fold in folds], axis=0)
    pooled_target = np.concatenate([fold["y_val"] for fold in folds], axis=0)
    descriptive_weights = constrained_brier_weights(
        pooled_predictions, pooled_target, anchor=anchor, ridge=0.0
    )
    descriptive_prediction = _fixed_blend(pooled_predictions, descriptive_weights)
    descriptive = {
        "warning": "descriptive in-sample OOF optimum; never use its score as a deployment estimate",
        "weights": descriptive_weights.tolist(),
        "brier": brier_score(pooled_target, descriptive_prediction),
    }
    result = {
        "manifest": str(manifest_path),
        "rule_compliance": {
            "fit_source": "inner temporal OOF only per outer fold",
            "test_rows_used_for_fit": False,
            "validation_labels_used_for_fit": False,
            "row_independent_runtime": True,
        },
        "configuration": {
            "ridges": list(ridges),
            "taus": list(taus),
            "min_samples": min_samples,
            "max_worst_loss": max_worst_loss,
            "regimes": list(SUPPORTED_REGIMES),
        },
        "descriptive_global_optimum": descriptive,
        "summaries": summaries,
        "fitted_by_fold": fitted_artifacts,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_diagnostics = _json_safe(diagnostics)
    safe_result = _json_safe(result)
    (output_dir / "member_diagnostics.json").write_text(
        json.dumps(safe_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "member_diagnostics.md").write_text(
        _diagnostics_markdown(safe_diagnostics), encoding="utf-8"
    )
    (output_dir / "regime_blend_results.json").write_text(
        json.dumps(safe_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "regime_blend_results.md").write_text(
        _results_markdown(safe_result["summaries"]), encoding="utf-8"
    )
    return safe_result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts" / "regime_blend_oof" / "manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "regime_blend_results",
    )
    parser.add_argument(
        "--ridges", type=_float_grid, default=DEFAULT_RIDGES,
        help="comma-separated pre-registered anchor penalties",
    )
    parser.add_argument(
        "--taus", type=_float_grid, default=DEFAULT_TAUS,
        help="comma-separated pre-registered regime shrinkage strengths",
    )
    parser.add_argument("--min-samples", type=int, default=500)
    parser.add_argument("--max-worst-loss", type=float, default=1e-5)
    args = parser.parse_args(argv)
    if not args.manifest.is_file():
        parser.error(
            f"OOF manifest not found: {args.manifest}. "
            "Restore the temporal champion OOF artifacts before running the experiment."
        )
    result = run(
        args.manifest,
        args.output_dir,
        ridges=args.ridges,
        taus=args.taus,
        min_samples=args.min_samples,
        max_worst_loss=args.max_worst_loss,
    )
    passed = [row["method"] for row in result["summaries"] if row["verdict"] == "pass"]
    print(f"saved regime-blend artifacts to {args.output_dir}")
    print(f"passing research candidates: {len(passed)}")
    for name in passed:
        print(f"  {name}")


if __name__ == "__main__":
    main()

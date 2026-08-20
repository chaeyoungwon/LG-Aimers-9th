"""Build cell-level weight and endpoint diagnostics for regime blends.

This is a reporting-only companion to ``run_regime_blend.py``.  It refits no
outer-fold value: local optima are reconstructed from each fold's inner OOF,
and Brier gains are evaluated on the untouched outer validation rows.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.regime_blend import (  # noqa: E402
    RegimeBlend,
    brier_score,
    constrained_brier_weights,
    regime_labels,
)


COLDSTART_WEIGHT = 0.4181944103545871


def _prediction_matrix(artifact: Mapping[str, np.ndarray], split: str, names: list[str]) -> np.ndarray:
    return np.column_stack(
        [np.asarray(artifact[f"p_{split}_{name}"], dtype="float64") for name in names]
    )


def _features(
    artifact: Mapping[str, np.ndarray], split: str, names: list[str]
) -> dict[str, np.ndarray]:
    return {name: np.asarray(artifact[f"x_{split}_{name}"]) for name in names}


def _fixed_blend(predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    output = np.zeros(len(predictions), dtype="float64")
    for column, weight in enumerate(weights):
        output += predictions[:, column] * float(weight)
    return np.clip(output, 0.0, 1.0)


def _best_method_per_family(summaries: list[dict]) -> dict[str, dict]:
    selected: dict[str, dict] = {}
    for summary in summaries:
        method = str(summary["method"])
        if not method.startswith("regime="):
            continue
        family = method.split(",", 1)[0].split("=", 1)[1]
        previous = selected.get(family)
        if previous is None or float(summary["pooled_brier_gain"]) > float(
            previous["pooled_brier_gain"]
        ):
            selected[family] = summary
    return selected


def _ridge_from_method(method: str) -> float:
    match = re.search(r"(?:^|,)rho=([^,]+)", method)
    if match is None:
        raise ValueError(f"method does not contain rho: {method}")
    return float(match.group(1))


def _endpoint_metrics(
    artifact: Mapping[str, np.ndarray], candidate_base: np.ndarray
) -> dict[str, dict[str, float]]:
    y = np.asarray(artifact["y_val"], dtype="float64")
    champion_base = np.asarray(artifact["p_val_champion_base"], dtype="float64")
    champion_form = np.asarray(artifact["p_val_champion_form"], dtype="float64")
    champion_final = np.asarray(artifact["p_val_champion_final"], dtype="float64")
    form_offset = np.asarray(artifact["form_offset_val"], dtype="float64")
    cold_expert = np.asarray(artifact["p_val_coldstart_expert"], dtype="float64")
    cold_mask = np.asarray(artifact["coldstart_mask_val"], dtype=bool)

    candidate_form = np.clip(candidate_base + form_offset, 0.0, 1.0)
    candidate_final = candidate_form.copy()
    candidate_final[cold_mask] = (
        (1.0 - COLDSTART_WEIGHT) * candidate_final[cold_mask]
        + COLDSTART_WEIGHT * cold_expert[cold_mask]
    )
    stages = {
        "base": (champion_base, candidate_base),
        "form": (champion_form, candidate_form),
        "final": (champion_final, candidate_final),
    }
    return {
        stage: {
            "reference_brier": brier_score(y, reference),
            "candidate_brier": brier_score(y, candidate),
            "brier_gain": brier_score(y, reference) - brier_score(y, candidate),
        }
        for stage, (reference, candidate) in stages.items()
    }


def build_diagnostics(manifest_path: Path, results_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    selected = _best_method_per_family(results["summaries"])
    blend_members = list(manifest["blend_members"])
    feature_columns = list(manifest["feature_columns"])
    folds_by_season = {
        int(fold["validation_season"]): fold for fold in manifest["folds"]
    }

    families: dict[str, dict] = {}
    for family, summary in selected.items():
        method = str(summary["method"])
        ridge = _ridge_from_method(method)
        fold_rows = []
        endpoint_rows = []
        for validation_season in sorted(folds_by_season):
            fold_spec = folds_by_season[validation_season]
            path = manifest_path.parent / str(fold_spec["path"])
            with np.load(path, allow_pickle=False) as loaded:
                artifact = {key: loaded[key] for key in loaded.files}

            inner_predictions = _prediction_matrix(artifact, "inner", blend_members)
            validation_predictions = _prediction_matrix(artifact, "val", blend_members)
            y_inner = np.asarray(artifact["y_inner"], dtype="float64")
            y_validation = np.asarray(artifact["y_val"], dtype="float64")
            inner_features = _features(artifact, "inner", feature_columns)
            validation_features = _features(artifact, "val", feature_columns)
            fitted_payload = results["fitted_by_fold"][method][str(validation_season)]
            fitted = RegimeBlend.from_dict(fitted_payload)
            inner_labels = regime_labels(inner_features, family, len(y_inner))
            validation_labels = regime_labels(
                validation_features, family, len(y_validation)
            )

            global_prediction = _fixed_blend(
                validation_predictions, fitted.global_weights
            )
            regime_prediction = fitted.predict(
                validation_predictions, validation_features
            )
            endpoint_rows.append(
                {
                    "validation_season": validation_season,
                    **_endpoint_metrics(artifact, regime_prediction),
                }
            )

            cells = []
            labels = sorted(set(map(str, np.unique(inner_labels))))
            for label in labels:
                inner_mask = inner_labels.astype("U") == label
                validation_mask = validation_labels.astype("U") == label
                local_optimum = constrained_brier_weights(
                    inner_predictions[inner_mask],
                    y_inner[inner_mask],
                    anchor=fitted.global_weights,
                    ridge=ridge,
                )
                shrunk = fitted.local_weights.get(label, fitted.global_weights)
                validation_n = int(validation_mask.sum())
                gain = None
                if validation_n:
                    gain = brier_score(
                        y_validation[validation_mask],
                        global_prediction[validation_mask],
                    ) - brier_score(
                        y_validation[validation_mask],
                        regime_prediction[validation_mask],
                    )
                cells.append(
                    {
                        "label": label,
                        "inner_n": int(inner_mask.sum()),
                        "validation_n": validation_n,
                        "global_weights": fitted.global_weights.tolist(),
                        "local_optimum_weights": local_optimum.tolist(),
                        "shrunk_weights": shrunk.tolist(),
                        "validation_brier_gain_vs_global": gain,
                    }
                )
            fold_rows.append(
                {
                    "validation_season": validation_season,
                    "overall_validation_brier_gain_vs_global": brier_score(
                        y_validation, global_prediction
                    )
                    - brier_score(y_validation, regime_prediction),
                    "cells": cells,
                }
            )
        families[family] = {
            "selected_method": method,
            "selection_basis": "largest pooled outer-fold gain within family; descriptive only",
            "folds": fold_rows,
            "endpoint_composition": endpoint_rows,
        }
    return {
        "manifest": str(manifest_path),
        "results": str(results_path),
        "warning": (
            "Family hyperparameters are selected descriptively across outer folds. "
            "Do not treat this report as an additional deployment estimate."
        ),
        "coldstart_weight": COLDSTART_WEIGHT,
        "families": families,
    }


def _weights(values: list[float]) -> str:
    return "/".join(f"{value:.4f}" for value in values)


def markdown(report: dict) -> str:
    lines = [
        "# Regime weight diagnostics",
        "",
        report["warning"],
        "",
    ]
    for family, body in report["families"].items():
        lines.extend(
            [
                f"## {family}",
                "",
                f"Selected diagnostic method: `{body['selected_method']}`",
                "",
                "Weights are `ours_stage/team_stage`. Positive gain beats the fold's global blend.",
                "",
                "| validation | cell | inner n | validation n | global | local optimum | shrunk | validation Brier gain vs global |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for fold in body["folds"]:
            for cell in fold["cells"]:
                gain = cell["validation_brier_gain_vs_global"]
                gain_text = "NA" if gain is None else f"{gain:+.3e}"
                lines.append(
                    f"| {fold['validation_season']} | {cell['label']} "
                    f"| {cell['inner_n']:,} | {cell['validation_n']:,} "
                    f"| {_weights(cell['global_weights'])} "
                    f"| {_weights(cell['local_optimum_weights'])} "
                    f"| {_weights(cell['shrunk_weights'])} | {gain_text} |"
                )
        lines.extend(
            [
                "",
                "Endpoint composition against the matching champion stage:",
                "",
                "| validation | base gain | + form gain | + form/coldstart gain |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for row in body["endpoint_composition"]:
            lines.append(
                f"| {row['validation_season']} "
                f"| {row['base']['brier_gain']:+.3e} "
                f"| {row['form']['brier_gain']:+.3e} "
                f"| {row['final']['brier_gain']:+.3e} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts" / "regime_blend_oof" / "manifest.json",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "artifacts" / "regime_blend_results" / "regime_blend_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "regime_blend_results",
    )
    args = parser.parse_args(argv)
    report = build_diagnostics(args.manifest, args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "regime_weight_diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "regime_weight_diagnostics.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(f"saved regime weight diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()

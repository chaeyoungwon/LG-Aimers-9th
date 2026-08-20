"""Promote the accepted w=.0175 endpoint and finalize the frozen upper grid.

This follow-up never evaluates a new weight.  It reads the prior upper-weight
OOF artifacts, compares only .0200/.0225/.0250 with .0175, and sends the
smallest clearly stable candidate through one production audit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_extratrees_candidate import (  # noqa: E402
    CHAMPION as ORIGINAL_CHAMPION,
    EXPECTED_CHAMPION_SHA256 as ORIGINAL_CHAMPION_SHA256,
    sha256_path,
    write_json,
)
from scripts.validate_et25_upper_weight import (  # noqa: E402
    ET25_MODEL,
    MAX_SUBMISSION_FILENAME_LENGTH,
    OOF_SOURCE_DIR,
    OUTPUT_DIR,
    SUBSET_DELTA_FLOOR_VS_REFERENCE,
    SUBSET_LOSS_FLOOR,
    VALIDATION_SEASONS,
    build_weight_candidate,
    candidate_filename,
    production_audit,
)


CURRENT_CHAMPION = ROOT / "artifacts" / "sub_et25_w0175.zip"
CURRENT_CHAMPION_SHA256 = (
    "7c884834651d38fa4e734e3a28846367a861fe406f05054398adba4ddbd825da"
)
CURRENT_CHAMPION_LB_BSS = 1017.0016684619
REFERENCE_WEIGHT = 0.0175
CANDIDATE_WEIGHTS = (0.0200, 0.0225, 0.0250)
MIN_CLEAR_POOLED_DELTA = 5e-7
DEFAULT_DATA_DIR = Path("/Users/wooh/Documents/dev/open/data")

REFERENCE_ZIP_SIZE_BYTES = 121_125_294
REFERENCE_RUNTIME_SECONDS = 9.825881500029936
REFERENCE_PEAK_RSS_BYTES = 2_410_430_464

COMPARISON_COLUMNS = (
    "weight",
    "gain_2022",
    "gain_2023",
    "gain_2024",
    "gain_pooled",
    "incremental_gain_2022_vs_w0175",
    "incremental_gain_2023_vs_w0175",
    "incremental_gain_2024_vs_w0175",
    "incremental_gain_pooled_vs_w0175",
    "worst_major_subset_gain",
    "worst_major_subset_delta_vs_w0175",
    "gain_2024_F",
    "delta_2024_F_vs_w0175",
    "gain_2024_full_count",
    "delta_2024_full_count_vs_w0175",
    "condition_2023_improves",
    "condition_2024_improves",
    "condition_pooled_improves",
    "condition_pooled_clearly_improves",
    "condition_major_subset_floor",
    "condition_subset_delta_floor",
    "passed",
)


def _one_gain(results: pd.DataFrame, weight: float, season: str) -> float:
    rows = results[
        np.isclose(results["weight"], weight)
        & results["validation_season"].astype(str).eq(season)
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one prior result for weight={weight}, season={season}")
    return float(rows.iloc[0]["brier_gain"])


def _one_subset_gain(
    subsets: pd.DataFrame,
    weight: float,
    season: str,
    axis: str,
    subset: str,
) -> float:
    rows = subsets[
        np.isclose(subsets["weight"], weight)
        & subsets["validation_season"].astype(str).eq(season)
        & subsets["axis"].eq(axis)
        & subsets["subset"].eq(subset)
    ]
    if len(rows) != 1:
        raise ValueError(
            "expected one prior subset result for "
            f"weight={weight}, season={season}, {axis}={subset}"
        )
    return float(rows.iloc[0]["gain_vs_original_champion"])


def build_comparison(
    results: pd.DataFrame,
    subsets: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize only the three frozen candidates relative to w=.0175."""
    required_weights = {REFERENCE_WEIGHT, *CANDIDATE_WEIGHTS}
    missing_results = required_weights - set(results["weight"].astype(float))
    missing_subsets = required_weights - set(subsets["weight"].astype(float))
    if missing_results or missing_subsets:
        raise ValueError(
            f"prior upper-weight artifacts incomplete: {missing_results=}, {missing_subsets=}"
        )

    reference_gains = {
        season: _one_gain(results, REFERENCE_WEIGHT, season)
        for season in ("2022", "2023", "2024", "pooled")
    }
    rows = []
    for weight in CANDIDATE_WEIGHTS:
        gains = {
            season: _one_gain(results, weight, season)
            for season in ("2022", "2023", "2024", "pooled")
        }
        increments = {
            season: gains[season] - reference_gains[season]
            for season in gains
        }
        eligible = subsets[
            np.isclose(subsets["weight"], weight) & subsets["eligible_for_gate"]
        ].copy()
        reference_subset = subsets[
            np.isclose(subsets["weight"], REFERENCE_WEIGHT)
            & subsets["eligible_for_gate"]
        ][
            ["validation_season", "axis", "subset", "gain_vs_original_champion"]
        ].rename(columns={"gain_vs_original_champion": "reference_gain"})
        eligible = eligible.merge(
            reference_subset,
            on=["validation_season", "axis", "subset"],
            how="left",
            validate="one_to_one",
        )
        eligible["delta_vs_w0175"] = (
            eligible["gain_vs_original_champion"] - eligible["reference_gain"]
        )
        gain_2024_f = _one_subset_gain(
            subsets, weight, "2024", "game_type", "F"
        )
        reference_2024_f = _one_subset_gain(
            subsets, REFERENCE_WEIGHT, "2024", "game_type", "F"
        )
        gain_2024_full = _one_subset_gain(
            subsets, weight, "2024", "full_count", "full"
        )
        reference_2024_full = _one_subset_gain(
            subsets, REFERENCE_WEIGHT, "2024", "full_count", "full"
        )
        conditions = {
            "condition_2023_improves": increments["2023"] > 0.0,
            "condition_2024_improves": increments["2024"] > 0.0,
            "condition_pooled_improves": increments["pooled"] > 0.0,
            "condition_pooled_clearly_improves": (
                increments["pooled"] >= MIN_CLEAR_POOLED_DELTA
            ),
            "condition_major_subset_floor": bool(
                (eligible["gain_vs_original_champion"] >= SUBSET_LOSS_FLOOR).all()
            ),
            "condition_subset_delta_floor": bool(
                (eligible["delta_vs_w0175"] >= SUBSET_DELTA_FLOOR_VS_REFERENCE).all()
            ),
        }
        rows.append(
            {
                "weight": weight,
                "gain_2022": gains["2022"],
                "gain_2023": gains["2023"],
                "gain_2024": gains["2024"],
                "gain_pooled": gains["pooled"],
                "incremental_gain_2022_vs_w0175": increments["2022"],
                "incremental_gain_2023_vs_w0175": increments["2023"],
                "incremental_gain_2024_vs_w0175": increments["2024"],
                "incremental_gain_pooled_vs_w0175": increments["pooled"],
                "worst_major_subset_gain": float(
                    eligible["gain_vs_original_champion"].min()
                ),
                "worst_major_subset_delta_vs_w0175": float(
                    eligible["delta_vs_w0175"].min()
                ),
                "gain_2024_F": gain_2024_f,
                "delta_2024_F_vs_w0175": gain_2024_f - reference_2024_f,
                "gain_2024_full_count": gain_2024_full,
                "delta_2024_full_count_vs_w0175": (
                    gain_2024_full - reference_2024_full
                ),
                **conditions,
                "passed": all(conditions.values()),
            }
        )
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def select_smallest_stable(comparison: pd.DataFrame) -> float | None:
    expected = list(CANDIDATE_WEIGHTS)
    actual = comparison["weight"].astype(float).tolist()
    if actual != expected:
        raise ValueError(f"comparison must preserve the frozen candidate grid: {actual}")
    passing = comparison.loc[comparison["passed"], "weight"].astype(float).tolist()
    return min(passing) if passing else None


def comparison_records(comparison: pd.DataFrame) -> list[dict]:
    records = []
    for row in comparison.to_dict(orient="records"):
        records.append(
            {
                key: bool(value) if isinstance(value, (bool, np.bool_)) else value
                for key, value in row.items()
            }
        )
    return records


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    weight_path = args.output / "weight_results.csv"
    subset_path = args.output / "subset_results.csv"
    required = [
        weight_path,
        subset_path,
        CURRENT_CHAMPION,
        ORIGINAL_CHAMPION,
        ET25_MODEL,
        args.data_dir / "test.csv",
        args.data_dir / "sample_submission.csv",
    ]
    required.extend(OOF_SOURCE_DIR / f"season_{s}.npz" for s in VALIDATION_SEASONS)
    for path in required:
        if not path.is_file():
            parser.error(f"required file not found: {path}")

    current_sha_before = sha256_path(CURRENT_CHAMPION)
    original_sha_before = sha256_path(ORIGINAL_CHAMPION)
    if current_sha_before != CURRENT_CHAMPION_SHA256:
        raise SystemExit("immutable w=.0175 champion checksum mismatch")
    if original_sha_before != ORIGINAL_CHAMPION_SHA256:
        raise SystemExit("original champion checksum mismatch")

    results = pd.read_csv(weight_path)
    subsets = pd.read_csv(subset_path)
    comparison = build_comparison(results, subsets)
    selected_weight = select_smallest_stable(comparison)
    comparison.to_csv(args.output / "next_weight_comparison.csv", index=False)

    if selected_weight is None:
        verdict = "B. w=.0175 REMAINS CHAMPION — STOP WEIGHT AXIS"
        audit: Mapping = {
            "performed": False,
            "reason": "no frozen upper candidate passed the temporal OOF gate",
        }
    else:
        verdict = "A. ONE NEXT LB CANDIDATE READY"
        name = candidate_filename(selected_weight)
        if len(name) > MAX_SUBMISSION_FILENAME_LENGTH:
            raise RuntimeError("candidate filename exceeds 30 characters")
        candidate = ROOT / "artifacts" / name
        allowed = {
            "sub_et25_w015.zip",
            CURRENT_CHAMPION.name,
            candidate.name,
        }
        unexpected = [
            path.name
            for path in sorted((ROOT / "artifacts").glob("sub_et25_w*.zip"))
            if path.name not in allowed
        ]
        if unexpected:
            raise RuntimeError(
                "more than one next-weight production candidate would exist: "
                f"{unexpected}"
            )
        build_weight_candidate(
            CURRENT_CHAMPION,
            candidate,
            selected_weight,
            source_weight=REFERENCE_WEIGHT,
            source_champion_sha256=CURRENT_CHAMPION_SHA256,
        )
        test = pd.read_csv(args.data_dir / "test.csv", encoding="utf-8-sig")
        sample = pd.read_csv(
            args.data_dir / "sample_submission.csv", encoding="utf-8-sig"
        )
        if test["row_id"].tolist() != sample["row_id"].tolist():
            raise ValueError("test/sample row order mismatch")
        audit = production_audit(
            candidate,
            selected_weight,
            test,
            results,
            current_sha_before,
            original_sha_before,
            current_champion=CURRENT_CHAMPION,
            expected_current_sha256=CURRENT_CHAMPION_SHA256,
            reference_zip_size_bytes=REFERENCE_ZIP_SIZE_BYTES,
            reference_runtime_seconds=REFERENCE_RUNTIME_SECONDS,
            reference_peak_rss_bytes=REFERENCE_PEAK_RSS_BYTES,
        )
        if not audit["passed"]:
            raise RuntimeError("selected w=.0200 package failed production audit")

    summary = {
        "verdict": verdict,
        "current_lb_champion": {
            "lb_bss": CURRENT_CHAMPION_LB_BSS,
            "artifact": str(CURRENT_CHAMPION),
            "sha256": CURRENT_CHAMPION_SHA256,
            "weight": REFERENCE_WEIGHT,
            "immutable": sha256_path(CURRENT_CHAMPION) == CURRENT_CHAMPION_SHA256,
        },
        "selection_source": "existing temporal OOF artifacts only; LB record not used",
        "new_weights_evaluated": [],
        "candidate_weights_reorganized": list(CANDIDATE_WEIGHTS),
        "selection_policy": {
            "choose_smallest_passing_candidate": True,
            "require_2023_improvement_vs_w0175": True,
            "require_2024_improvement_vs_w0175": True,
            "require_pooled_improvement_vs_w0175": True,
            "minimum_clear_pooled_delta": MIN_CLEAR_POOLED_DELTA,
            "major_subset_gain_floor": SUBSET_LOSS_FLOOR,
            "major_subset_delta_floor_vs_w0175": SUBSET_DELTA_FLOOR_VS_REFERENCE,
        },
        "comparisons": comparison_records(comparison),
        "selected_weight": selected_weight,
        "production_audit": audit,
        "candidate": audit.get("candidate"),
    }
    write_json(args.output / "next_candidate.json", summary)
    print(json.dumps({
        "verdict": verdict,
        "selected_weight": selected_weight,
        "candidate": audit.get("candidate"),
        "production_audit_passed": audit.get("passed", False),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

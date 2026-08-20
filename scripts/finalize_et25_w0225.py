"""Build and audit the single frozen ET25 w=.0225 LB candidate.

No OOF prediction or new weight is evaluated here.  The script reads the
existing upper-weight artifacts, gates .0225 directly against the immutable
.020 champion, and only then creates one production-equivalent package.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

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
    production_audit,
)


CURRENT_CHAMPION = ROOT / "artifacts" / "sub_et25_w020.zip"
CURRENT_CHAMPION_SHA256 = (
    "316bf6dab11fb78f06cc74089ea38bdb3b957355f4510db5fcc8a485dc26cecf"
)
CURRENT_CHAMPION_LB_BSS = 1017.0233029621
CURRENT_WEIGHT = 0.0200
CANDIDATE_WEIGHT = 0.0225
CANDIDATE_NAME = "sub_et25_w0225.zip"
TREE_COUNT = 25
DEFAULT_DATA_DIR = Path("/Users/wooh/Documents/dev/open/data")

CURRENT_ZIP_SIZE_BYTES = 121_125_327
CURRENT_RUNTIME_SECONDS = 9.570166416931897
CURRENT_PEAK_RSS_BYTES = 2_459_631_616
MAX_RUNTIME_RATIO_VS_CURRENT = 1.25
MAX_RSS_RATIO_VS_CURRENT = 1.10
MAX_ZIP_SIZE_RATIO_VS_CURRENT = 1.02


def _gain(results: pd.DataFrame, weight: float, season: str) -> float:
    rows = results[
        np.isclose(results["weight"], weight)
        & results["validation_season"].astype(str).eq(season)
    ]
    if len(rows) != 1:
        raise ValueError(f"missing frozen result for weight={weight}, season={season}")
    return float(rows.iloc[0]["brier_gain"])


def build_direct_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Compare the two existing endpoints without creating new predictions."""
    rows = []
    for season in ("2022", "2023", "2024", "pooled"):
        current = _gain(results, CURRENT_WEIGHT, season)
        candidate = _gain(results, CANDIDATE_WEIGHT, season)
        rows.append(
            {
                "validation_season": season,
                "current_w020_gain_vs_original": current,
                "candidate_w0225_gain_vs_original": candidate,
                "incremental_gain_vs_w020": candidate - current,
            }
        )
    return pd.DataFrame(rows)


def build_subset_comparison(subsets: pd.DataFrame) -> pd.DataFrame:
    keys = ["validation_season", "axis", "subset"]
    current = subsets[np.isclose(subsets["weight"], CURRENT_WEIGHT)][
        [*keys, "n", "gain_vs_original_champion", "eligible_for_gate"]
    ].rename(
        columns={
            "n": "current_n",
            "gain_vs_original_champion": "current_w020_gain_vs_original",
            "eligible_for_gate": "current_eligible_for_gate",
        }
    )
    candidate = subsets[np.isclose(subsets["weight"], CANDIDATE_WEIGHT)][
        [*keys, "n", "gain_vs_original_champion", "eligible_for_gate"]
    ].rename(
        columns={
            "n": "candidate_n",
            "gain_vs_original_champion": "candidate_w0225_gain_vs_original",
            "eligible_for_gate": "candidate_eligible_for_gate",
        }
    )
    comparison = candidate.merge(current, on=keys, how="inner", validate="one_to_one")
    if len(comparison) != len(candidate) or len(comparison) != len(current):
        raise ValueError("w=.020/.0225 subset keys do not match")
    if not np.array_equal(comparison["candidate_n"], comparison["current_n"]):
        raise ValueError("w=.020/.0225 subset row counts differ")
    comparison["incremental_gain_vs_w020"] = (
        comparison["candidate_w0225_gain_vs_original"]
        - comparison["current_w020_gain_vs_original"]
    )
    return comparison[
        [
            *keys,
            "candidate_n",
            "candidate_w0225_gain_vs_original",
            "current_w020_gain_vs_original",
            "incremental_gain_vs_w020",
            "candidate_eligible_for_gate",
        ]
    ].rename(columns={"candidate_n": "n", "candidate_eligible_for_gate": "eligible_for_gate"})


def safety_gate(
    direct: pd.DataFrame,
    subsets: pd.DataFrame,
) -> tuple[bool, dict]:
    gains = {
        str(row.validation_season): float(row.candidate_w0225_gain_vs_original)
        for row in direct.itertuples()
    }
    deltas = {
        str(row.validation_season): float(row.incremental_gain_vs_w020)
        for row in direct.itertuples()
    }
    eligible = subsets[subsets["eligible_for_gate"]]
    conditions = {
        "2023_delta_vs_w020_positive": deltas["2023"] > 0.0,
        "2024_delta_vs_w020_positive": deltas["2024"] > 0.0,
        "pooled_delta_vs_w020_positive": deltas["pooled"] > 0.0,
        "2022_gain_vs_original_positive": gains["2022"] > 0.0,
        "2023_gain_vs_original_positive": gains["2023"] > 0.0,
        "2024_gain_vs_original_positive": gains["2024"] > 0.0,
        "major_subset_gain_floor": bool(
            (eligible["candidate_w0225_gain_vs_original"] >= SUBSET_LOSS_FLOOR).all()
        ),
        "no_large_subset_degradation_vs_w020": bool(
            (eligible["incremental_gain_vs_w020"] >= SUBSET_DELTA_FLOOR_VS_REFERENCE).all()
        ),
    }
    diagnostics = {
        "conditions": conditions,
        "worst_major_subset_gain_vs_original": float(
            eligible["candidate_w0225_gain_vs_original"].min()
        ),
        "worst_major_subset_delta_vs_w020": float(
            eligible["incremental_gain_vs_w020"].min()
        ),
    }
    return all(conditions.values()), diagnostics


def add_resource_parity(audit: dict) -> None:
    runtime = audit["runtime"]
    candidate = audit["candidate"]
    ratios = {
        "zip_size_ratio_vs_w020": candidate["size_bytes"] / CURRENT_ZIP_SIZE_BYTES,
        "runtime_ratio_vs_w020": runtime["wall_seconds"] / CURRENT_RUNTIME_SECONDS,
        "peak_rss_ratio_vs_w020": runtime["peak_rss_bytes"] / CURRENT_PEAK_RSS_BYTES,
    }
    conditions = {
        "zip_size_near_w020": ratios["zip_size_ratio_vs_w020"]
        <= MAX_ZIP_SIZE_RATIO_VS_CURRENT,
        "runtime_near_w020": ratios["runtime_ratio_vs_w020"]
        <= MAX_RUNTIME_RATIO_VS_CURRENT,
        "peak_rss_near_w020": ratios["peak_rss_ratio_vs_w020"]
        <= MAX_RSS_RATIO_VS_CURRENT,
    }
    audit["resource_parity_vs_w020"] = {
        **ratios,
        "conditions": conditions,
        "passed": all(conditions.values()),
    }
    audit["passed"] = bool(audit["passed"] and all(conditions.values()))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    if CANDIDATE_NAME != "sub_et25_w0225.zip":
        raise RuntimeError("candidate filename is not the exact frozen name")
    if len(CANDIDATE_NAME) > MAX_SUBMISSION_FILENAME_LENGTH:
        raise RuntimeError("candidate filename exceeds the 30-character hard gate")

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
    required.extend(OOF_SOURCE_DIR / f"season_{season}.npz" for season in VALIDATION_SEASONS)
    for path in required:
        if not path.is_file():
            parser.error(f"required file not found: {path}")

    current_sha_before = sha256_path(CURRENT_CHAMPION)
    original_sha_before = sha256_path(ORIGINAL_CHAMPION)
    if current_sha_before != CURRENT_CHAMPION_SHA256:
        raise SystemExit("immutable w=.020 champion checksum mismatch")
    if original_sha_before != ORIGINAL_CHAMPION_SHA256:
        raise SystemExit("original champion checksum mismatch")

    existing = sorted((ROOT / "artifacts").glob("sub_et25_w*.zip"))
    allowed_before = {
        "sub_et25_w015.zip",
        "sub_et25_w0175.zip",
        CURRENT_CHAMPION.name,
        CANDIDATE_NAME,
    }
    unexpected = [path.name for path in existing if path.name not in allowed_before]
    if unexpected:
        raise RuntimeError(f"forbidden extra weight package exists: {unexpected}")

    results = pd.read_csv(weight_path)
    subset_source = pd.read_csv(subset_path)
    direct = build_direct_comparison(results)
    subsets = build_subset_comparison(subset_source)
    passed, gate = safety_gate(direct, subsets)
    direct.to_csv(args.output / "w0225_vs_w020.csv", index=False)
    subsets.to_csv(args.output / "w0225_subset_vs_w020.csv", index=False)

    if not passed:
        summary = {
            "verdict": "B. w=.020 REMAINS SAFER — DO NOT SUBMIT",
            "safety_gate": gate,
            "production_audit": {"performed": False},
        }
        write_json(args.output / "w0225_candidate.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    candidate = ROOT / "artifacts" / CANDIDATE_NAME
    build_weight_candidate(
        CURRENT_CHAMPION,
        candidate,
        CANDIDATE_WEIGHT,
        source_weight=CURRENT_WEIGHT,
        source_champion_sha256=CURRENT_CHAMPION_SHA256,
    )
    test = pd.read_csv(args.data_dir / "test.csv", encoding="utf-8-sig")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv", encoding="utf-8-sig")
    if test["row_id"].tolist() != sample["row_id"].tolist():
        raise ValueError("test/sample row order mismatch")
    audit = production_audit(
        candidate,
        CANDIDATE_WEIGHT,
        test,
        results,
        current_sha_before,
        original_sha_before,
        current_champion=CURRENT_CHAMPION,
        expected_current_sha256=CURRENT_CHAMPION_SHA256,
        reference_zip_size_bytes=CURRENT_ZIP_SIZE_BYTES,
        reference_runtime_seconds=CURRENT_RUNTIME_SECONDS,
        reference_peak_rss_bytes=CURRENT_PEAK_RSS_BYTES,
    )
    add_resource_parity(audit)
    if not audit["passed"]:
        candidate.unlink(missing_ok=True)
        raise RuntimeError("w=.0225 failed production audit; candidate ZIP removed")

    summary = {
        "verdict": "A. w=.0225 READY FOR ONE LB SUBMISSION",
        "current_immutable_lb_champion": {
            "lb_bss": CURRENT_CHAMPION_LB_BSS,
            "artifact": str(CURRENT_CHAMPION),
            "sha256": CURRENT_CHAMPION_SHA256,
            "weight": CURRENT_WEIGHT,
            "tree_count": TREE_COUNT,
        },
        "selection_source": "existing upper-weight OOF artifacts only",
        "new_weights_evaluated": [],
        "candidate_weight": CANDIDATE_WEIGHT,
        "prediction_formula": "0.9775 * original_champion + 0.0225 * frozen_et25",
        "direct_comparison": direct.to_dict(orient="records"),
        "safety_gate": gate,
        "production_audit": audit,
        "candidate": audit["candidate"],
    }
    write_json(args.output / "w0225_candidate.json", summary)
    print(
        json.dumps(
            {
                "verdict": summary["verdict"],
                "candidate": audit["candidate"],
                "delta_vs_w020": {
                    str(row.validation_season): row.incremental_gain_vs_w020
                    for row in direct.itertuples()
                },
                "worst_subset_delta": gate["worst_major_subset_delta_vs_w020"],
                "runtime": audit["runtime"],
                "production_audit_passed": audit["passed"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

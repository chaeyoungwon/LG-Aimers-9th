"""Brier-optimal, row-independent ensemble blending utilities.

This module deliberately has no dependency on the submission runtime or test
data.  Fitting consumes temporal OOF predictions and labels only.  Inference
uses a row's member predictions plus a regime label derived from that same row.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


Array = np.ndarray
SUPPORTED_REGIMES = (
    "game_type",
    "hand_combo",
    "count_state",
    "game_type_count_bucket",
)


def _as_prediction_matrix(predictions: Array) -> Array:
    values = np.asarray(predictions, dtype="float64")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("predictions must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("predictions contain NaN or infinity")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("predictions must be probabilities in [0, 1]")
    return values


def _as_binary_target(target: Array, n_rows: int) -> Array:
    values = np.asarray(target, dtype="float64")
    if values.shape != (n_rows,):
        raise ValueError(f"target shape must be ({n_rows},), got {values.shape}")
    if not np.all(np.isfinite(values)) or np.any((values != 0.0) & (values != 1.0)):
        raise ValueError("target must contain finite binary labels only")
    return values


def _normalize_anchor(anchor: Optional[Array], n_members: int) -> Array:
    if anchor is None:
        return np.full(n_members, 1.0 / n_members, dtype="float64")
    values = np.asarray(anchor, dtype="float64")
    if values.shape != (n_members,):
        raise ValueError(f"anchor shape must be ({n_members},), got {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("anchor weights must be finite and non-negative")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("anchor weights must have a positive sum")
    return values / total


def _row_weighted_sum(predictions: Array, weights: Array) -> Array:
    """Use a fixed column accumulation order for batch/order invariance."""
    out = np.zeros(predictions.shape[0], dtype="float64")
    for column, weight in enumerate(weights):
        out += predictions[:, column] * float(weight)
    return out


def brier_score(target: Array, prediction: Array) -> float:
    y = np.asarray(target, dtype="float64")
    p = np.asarray(prediction, dtype="float64")
    if y.shape != p.shape:
        raise ValueError(f"target/prediction shape mismatch: {y.shape} != {p.shape}")
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("target and prediction must be non-empty vectors")
    return float(np.mean((p - y) ** 2))


def bss_delta_from_brier_gain(target: Array, brier_gain: float) -> float:
    """Convert positive Brier improvement into the competition BSS delta."""
    y = np.asarray(target, dtype="float64")
    prevalence = float(y.mean())
    baseline = prevalence * (1.0 - prevalence)
    return 0.0 if baseline <= 0.0 else 100000.0 * float(brier_gain) / baseline


def constrained_brier_weights(
    predictions: Array,
    target: Array,
    *,
    anchor: Optional[Array] = None,
    ridge: float = 0.0,
    tolerance: float = 1e-10,
) -> Array:
    """Solve a non-negative, sum-to-one Brier blend exactly by active sets.

    The optimized objective is::

        mean((P @ w - y) ** 2) + ridge * sum((w - anchor) ** 2)

    Enumerating non-empty active sets is deterministic and exact for the small
    ensembles used here.  It also avoids adding SciPy to the submission.
    """
    matrix = _as_prediction_matrix(predictions)
    y = _as_binary_target(target, matrix.shape[0])
    n_members = matrix.shape[1]
    if n_members > 15:
        raise ValueError("active-set solver supports at most 15 members")
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    anchor_values = _normalize_anchor(anchor, n_members)

    gram = matrix.T @ matrix / matrix.shape[0]
    linear = matrix.T @ y / matrix.shape[0]
    if ridge:
        gram = gram + ridge * np.eye(n_members, dtype="float64")
        linear = linear + ridge * anchor_values

    best_weights = None
    best_objective = np.inf
    member_indices = range(n_members)
    for active_size in range(1, n_members + 1):
        for active_tuple in combinations(member_indices, active_size):
            active = np.asarray(active_tuple, dtype="int64")
            local_gram = gram[np.ix_(active, active)]
            kkt = np.empty((active_size + 1, active_size + 1), dtype="float64")
            kkt[:active_size, :active_size] = local_gram
            kkt[:active_size, active_size] = 1.0
            kkt[active_size, :active_size] = 1.0
            kkt[active_size, active_size] = 0.0
            rhs = np.r_[linear[active], 1.0]
            solution, _, _, _ = np.linalg.lstsq(kkt, rhs, rcond=None)
            local_weights = solution[:active_size]
            if np.any(local_weights < -tolerance):
                continue
            local_weights = np.maximum(local_weights, 0.0)
            local_total = float(local_weights.sum())
            if local_total <= 0.0:
                continue
            local_weights /= local_total
            weights = np.zeros(n_members, dtype="float64")
            weights[active] = local_weights
            blended = _row_weighted_sum(matrix, weights)
            objective = float(np.mean((blended - y) ** 2))
            if ridge:
                objective += float(ridge * np.sum((weights - anchor_values) ** 2))
            if objective < best_objective - tolerance:
                best_objective = objective
                best_weights = weights

    if best_weights is None:
        raise RuntimeError("no feasible simplex solution found")
    best_weights[np.abs(best_weights) < tolerance] = 0.0
    return best_weights / best_weights.sum()


def _require_feature(features: Mapping[str, Array], name: str, n_rows: int) -> Array:
    if name not in features:
        raise ValueError(f"missing regime feature: {name}")
    values = np.asarray(features[name])
    if values.shape != (n_rows,):
        raise ValueError(f"feature {name!r} shape must be ({n_rows},), got {values.shape}")
    return values


def regime_labels(features: Mapping[str, Array], regime: str, n_rows: int) -> Array:
    """Build a regime label from current-row values only."""
    if regime not in SUPPORTED_REGIMES:
        raise ValueError(f"unsupported regime {regime!r}; expected {SUPPORTED_REGIMES}")
    if regime == "game_type":
        return _require_feature(features, "game_type", n_rows).astype("U32")

    if regime == "hand_combo":
        pitcher = _require_feature(features, "pitcher_hand", n_rows).astype("U16")
        batter = _require_feature(features, "batter_hand", n_rows).astype("U16")
        return np.char.add(np.char.add(pitcher, "-"), batter)

    balls = _require_feature(features, "balls_before", n_rows).astype("int16")
    strikes = _require_feature(features, "strikes_before", n_rows).astype("int16")
    if np.any((balls < 0) | (balls > 3) | (strikes < 0) | (strikes > 2)):
        raise ValueError("balls_before/strikes_before contain an invalid count")
    if regime == "count_state":
        return (balls * 3 + strikes).astype("U8")

    game_type = _require_feature(features, "game_type", n_rows).astype("U32")
    bucket = np.full(n_rows, "even", dtype="U16")
    bucket[strikes > balls] = "ahead"
    bucket[strikes < balls] = "behind"
    return np.char.add(np.char.add(game_type, "-"), bucket)


@dataclass(frozen=True)
class RegimeBlend:
    """A fitted train-only lookup of global and shrunk local weights."""

    member_names: Tuple[str, ...]
    regime: Optional[str]
    global_weights: Array
    local_weights: Dict[str, Array]
    local_sample_sizes: Dict[str, int]
    tau: float
    min_samples: int
    ridge: float

    def __post_init__(self) -> None:
        names = tuple(self.member_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("member_names must be non-empty and unique")
        global_weights = _normalize_anchor(self.global_weights, len(names))
        object.__setattr__(self, "member_names", names)
        object.__setattr__(self, "global_weights", global_weights)
        if self.regime is not None and self.regime not in SUPPORTED_REGIMES:
            raise ValueError(f"unsupported regime: {self.regime}")
        normalized_local = {}
        for label, weights in self.local_weights.items():
            normalized_local[str(label)] = _normalize_anchor(weights, len(names))
        object.__setattr__(self, "local_weights", normalized_local)

    def predict(self, predictions: Array, features: Optional[Mapping[str, Array]] = None) -> Array:
        matrix = _as_prediction_matrix(predictions)
        if matrix.shape[1] != len(self.member_names):
            raise ValueError(
                f"prediction members {matrix.shape[1]} != fitted members {len(self.member_names)}"
            )
        if self.regime is None:
            return np.clip(_row_weighted_sum(matrix, self.global_weights), 0.0, 1.0)
        if features is None:
            raise ValueError("features are required for a regime-aware blend")
        labels = regime_labels(features, self.regime, matrix.shape[0])
        out = np.empty(matrix.shape[0], dtype="float64")
        for label in np.unique(labels):
            mask = labels == label
            weights = self.local_weights.get(str(label), self.global_weights)
            out[mask] = _row_weighted_sum(matrix[mask], weights)
        return np.clip(out, 0.0, 1.0)

    def to_dict(self) -> dict:
        return {
            "member_names": list(self.member_names),
            "regime": self.regime,
            "global_weights": self.global_weights.tolist(),
            "local_weights": {k: v.tolist() for k, v in sorted(self.local_weights.items())},
            "local_sample_sizes": dict(sorted(self.local_sample_sizes.items())),
            "tau": float(self.tau),
            "min_samples": int(self.min_samples),
            "ridge": float(self.ridge),
            "source": "temporal inner OOF only; no test distribution statistics",
            "row_independent": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RegimeBlend":
        return cls(
            member_names=tuple(str(x) for x in payload["member_names"]),
            regime=None if payload.get("regime") is None else str(payload["regime"]),
            global_weights=np.asarray(payload["global_weights"], dtype="float64"),
            local_weights={
                str(k): np.asarray(v, dtype="float64")
                for k, v in dict(payload.get("local_weights", {})).items()
            },
            local_sample_sizes={
                str(k): int(v) for k, v in dict(payload.get("local_sample_sizes", {})).items()
            },
            tau=float(payload.get("tau", 0.0)),
            min_samples=int(payload.get("min_samples", 0)),
            ridge=float(payload.get("ridge", 0.0)),
        )


def fit_global_blend(
    predictions: Array,
    target: Array,
    member_names: Sequence[str],
    *,
    anchor: Optional[Array] = None,
    ridge: float = 0.0,
) -> RegimeBlend:
    matrix = _as_prediction_matrix(predictions)
    if len(member_names) != matrix.shape[1]:
        raise ValueError("member_names length does not match prediction columns")
    weights = constrained_brier_weights(matrix, target, anchor=anchor, ridge=ridge)
    return RegimeBlend(
        member_names=tuple(member_names),
        regime=None,
        global_weights=weights,
        local_weights={},
        local_sample_sizes={},
        tau=0.0,
        min_samples=0,
        ridge=ridge,
    )


def fit_regime_blend(
    predictions: Array,
    target: Array,
    features: Mapping[str, Array],
    member_names: Sequence[str],
    regime: str,
    *,
    anchor: Optional[Array] = None,
    ridge: float = 0.0,
    tau: float = 10000.0,
    min_samples: int = 500,
) -> RegimeBlend:
    """Fit local OOF optima and shrink them toward the global OOF optimum."""
    matrix = _as_prediction_matrix(predictions)
    y = _as_binary_target(target, matrix.shape[0])
    if len(member_names) != matrix.shape[1]:
        raise ValueError("member_names length does not match prediction columns")
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and non-negative")
    if min_samples < 1:
        raise ValueError("min_samples must be positive")
    labels = regime_labels(features, regime, matrix.shape[0])
    global_weights = constrained_brier_weights(matrix, y, anchor=anchor, ridge=ridge)
    local_weights: Dict[str, Array] = {}
    local_sizes: Dict[str, int] = {}
    for raw_label in np.unique(labels):
        label = str(raw_label)
        mask = labels == raw_label
        sample_size = int(mask.sum())
        local_sizes[label] = sample_size
        if sample_size < min_samples:
            continue
        local_optimum = constrained_brier_weights(
            matrix[mask], y[mask], anchor=global_weights, ridge=ridge
        )
        reliability = sample_size / (sample_size + tau) if tau else 1.0
        local_weights[label] = (
            reliability * local_optimum + (1.0 - reliability) * global_weights
        )
    return RegimeBlend(
        member_names=tuple(member_names),
        regime=regime,
        global_weights=global_weights,
        local_weights=local_weights,
        local_sample_sizes=local_sizes,
        tau=tau,
        min_samples=min_samples,
        ridge=ridge,
    )


def member_diagnostics(
    member_predictions: Mapping[str, Array],
    target: Array,
    features: Mapping[str, Array],
) -> dict:
    """Return member Brier, segment Brier, and prediction/residual diversity."""
    if not member_predictions:
        raise ValueError("member_predictions is empty")
    names = tuple(member_predictions)
    columns = [np.asarray(member_predictions[name], dtype="float64") for name in names]
    lengths = {len(column) for column in columns}
    if len(lengths) != 1:
        raise ValueError("member prediction lengths differ")
    n_rows = lengths.pop()
    matrix = _as_prediction_matrix(np.column_stack(columns))
    y = _as_binary_target(target, n_rows)
    with np.errstate(invalid="ignore", divide="ignore"):
        prediction_correlation = np.corrcoef(matrix, rowvar=False)
        residual_correlation = np.corrcoef(matrix - y[:, None], rowvar=False)
    diagnostics = {
        "n": n_rows,
        "members": list(names),
        "overall_brier": {
            name: brier_score(y, matrix[:, i]) for i, name in enumerate(names)
        },
        "prediction_correlation": prediction_correlation.tolist(),
        "residual_correlation": residual_correlation.tolist(),
        "segments": {},
        "pairwise_error_advantage": {},
    }

    segment_labels = {
        "game_type": regime_labels(features, "game_type", n_rows),
        "count_state": regime_labels(features, "count_state", n_rows),
        "pitcher_hand": _require_feature(features, "pitcher_hand", n_rows).astype("U16"),
        "batter_hand": _require_feature(features, "batter_hand", n_rows).astype("U16"),
        "hand_combo": regime_labels(features, "hand_combo", n_rows),
    }
    if "pitcher_seen" in features:
        seen = _require_feature(features, "pitcher_seen", n_rows).astype(bool)
        segment_labels["pitcher_seen"] = np.where(seen, "seen", "unseen")

    for segment_name, labels in segment_labels.items():
        cells = {}
        for raw_label in np.unique(labels):
            label = str(raw_label)
            mask = labels == raw_label
            cells[label] = {
                "n": int(mask.sum()),
                "brier": {
                    name: brier_score(y[mask], matrix[mask, i])
                    for i, name in enumerate(names)
                },
            }
        diagnostics["segments"][segment_name] = cells

    squared_errors = (matrix - y[:, None]) ** 2
    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if i == j:
                continue
            advantage = squared_errors[:, j] - squared_errors[:, i]
            wins = advantage > 0.0
            key = f"{name_a}_over_{name_b}"
            if wins.any():
                values = advantage[wins]
                stderr = 0.0 if len(values) == 1 else float(values.std(ddof=1) / np.sqrt(len(values)))
                diagnostics["pairwise_error_advantage"][key] = {
                    "win_rows": int(wins.sum()),
                    "win_rate": float(wins.mean()),
                    "conditional_mean_brier_gain": float(values.mean()),
                    "conditional_stderr": stderr,
                    "unconditional_mean_brier_gain": float(advantage.mean()),
                }
            else:
                diagnostics["pairwise_error_advantage"][key] = {
                    "win_rows": 0,
                    "win_rate": 0.0,
                    "conditional_mean_brier_gain": 0.0,
                    "conditional_stderr": 0.0,
                    "unconditional_mean_brier_gain": float(advantage.mean()),
                }
    return diagnostics

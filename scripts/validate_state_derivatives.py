"""시즌 상태 기반 파생 후보를 하나씩 다중 시드로 판정한다.

현행 ``prev_vs_career`` 는 ``prev5 - career`` 이다. 시즌 상태 피처 위의 싼 파생을
후보 하나씩 검사한다. ``recent``는 ``prev5 - cur_p_succ``, ``context``는 상태와
카운트/좌우의 명시적 상호작용, ``delta``는 당해 시즌과 커리어 비율의 차이,
``exact``는 이전 시즌 마지막 성공 라벨까지 완결한 정확 경계다. 한 번의 실행에서는
후보 하나만 추가하며 다른 설정은 바꾸지 않는다.

판정 규칙:
  1. 기준선과 후보를 모두 seed 42/43/44로 재학습한다.
  2. 롤링 3폴드(2022/2023/2024)의 R 구간을 비교한다.
  3. 세 폴드 모두 개선하고, 각 개선폭이 예측 차이로 계산한 2σ를 넘어야 채택한다.

예시:
  DYLD_LIBRARY_PATH=/path/to/.venv/lib OMP_NUM_THREADS=1 \
    .venv/bin/python scripts/validate_state_derivatives.py \
    --train /path/to/data/train.csv
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import backtest as bt  # noqa: E402
from src import season_state as ss  # noqa: E402
from src.train_base import (  # noqa: E402
    CAT_COLS,
    MODEL_CONFIGS,
    TARGET_COL,
    add_features,
    apply_category_maps,
    build_category_maps,
)

DEFAULT_TRAIN = ROOT / "data" / "train.csv"
DEFAULT_SEEDS = (42, 43, 44)
BASELINE_BRIER = 0.2498
CANDIDATES = {
    "recent": {
        "feature": "prev5_vs_cur_p_succ",
        "formula": "asof_pitcher_prev5_game_success_rate - cur_p_succ",
    },
    "context": {
        "feature": "pitcher_state_x_count_hand",
        "formula": "5 pitcher state rates gated into 12 count + 4 hand cells",
    },
    "delta": {
        "feature": "season_state_minus_career",
        "formula": "10 current-season rates minus their career asof rates",
    },
    "exact": {
        "feature": "exact_success_season_boundary",
        "formula": "pitcher/batter success state with final prior-season label included",
    },
}


def make_configs(seeds):
    """leaf63 프록시의 구조는 고정하고 랜덤 시드만 바꾼다."""
    base = next(c for c in MODEL_CONFIGS if c["name"] == "leaf63")
    configs = []
    for seed in seeds:
        cfg = copy.deepcopy(base)
        cfg.update(name=f"leaf63_s{seed}", seed=int(seed), weight=1.0)
        configs.append(cfg)
    return configs


class StateFeatureBuilders:
    """비싼 시즌 상태 복원은 한 번만 하고 두 실험군이 공유한다."""

    def __init__(self, df):
        self.base = add_features(df)
        self.state = ss.add_features_by_season(df)
        self.candidates = {
            "recent": ss.recent_form_feature(df, self.state),
            "context": ss.context_interaction_features(df, self.state),
            "delta": ss.state_vs_career_features(df, self.state),
            "exact": ss.exact_success_features_by_season(df),
        }

    def builder(self, candidate=None):
        def build(df, train_seasons):
            if len(df) != len(self.base) or not df.index.equals(self.base.index):
                raise ValueError("초기화 때 사용한 동일 DataFrame만 빌더에 전달해야 한다")

            feature_cols = [
                c for c in self.base.columns if c not in ("row_id", TARGET_COL)
            ] + list(self.state.columns)
            extras = [self.state]
            if candidate is not None:
                feature_cols += list(self.candidates[candidate].columns)
                extras.append(self.candidates[candidate])

            cat_cols = [c for c in CAT_COLS if c in feature_cols]
            maps = build_category_maps(
                self.base[self.base.season.isin(train_seasons)], cat_cols
            )
            mapped = apply_category_maps(self.base, cat_cols, maps)
            X = pd.concat([mapped, *extras], axis=1)[feature_cols]
            return X, feature_cols, cat_cols

        return build


def calibrated_r_predictions(raw, df):
    """``evaluate(by_game_type=True)``와 같은 R 보정을 거친 폴드별 예측."""
    season = df.season.astype(int).to_numpy()
    game_type = df.game_type.to_numpy()
    y = df[TARGET_COL].to_numpy(dtype="float64")
    out = {}
    for target in bt.EVAL_SEASONS:
        prev = target - 1
        mp_all = season == prev
        mt_all = season == target
        mp_r = mp_all & (game_type == "R")
        mt_r = mt_all & (game_type == "R")
        a, b = bt._fit_calibration(raw[prev][game_type[mp_all] == "R"], y[mp_r])
        out[str(target)] = bt._apply_calibration(
            raw[target][game_type[mt_all] == "R"], a, b
        )
    return out


def r_results(evaluated):
    return evaluated[["season", "bss_R"]].rename(columns={"bss_R": "bss"})


def two_sigma(pred_a, pred_b):
    rows = {}
    for season in bt.EVAL_SEASONS:
        key = str(season)
        delta = np.asarray(pred_b[key]) - np.asarray(pred_a[key])
        rows[season] = float(
            2 * 100000 * np.sqrt(np.sum(delta**2))
            / (len(delta) * BASELINE_BRIER)
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), default="recent")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args(argv)

    df = pd.read_csv(args.train, encoding="utf-8-sig")
    print(f"train {df.shape}; seeds={args.seeds}", flush=True)
    frames = StateFeatureBuilders(df)
    candidate = frames.candidates[args.candidate]
    coverage = candidate.notna().any(axis=1).mean()
    print(
        f"candidate={args.candidate}; n_features={candidate.shape[1]}; "
        f"row_coverage={coverage:.3f}",
        flush=True,
    )

    configs = make_configs(args.seeds)
    default_dir = {
        "recent": "recent_vs_state",
        "context": "state_context",
        "delta": "state_vs_career",
        "exact": "exact_success_state",
    }[args.candidate]
    out_dir = args.out_dir or ROOT / "artifacts" / "backtest" / default_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = {}
    results = {}
    candidate_name = f"state_{args.candidate}"
    for name, selected in (("state", None), (candidate_name, args.candidate)):
        print(f"\n[{name}] {len(configs)}시드 평균 학습", flush=True)
        raw[name] = bt.fit_raw(
            frames.builder(selected),
            name,
            configs=configs,
            df=df,
            out_dir=out_dir,
        )
        results[name] = bt.evaluate(
            raw[name], df, by_game_type=True, name=name, verbose=True
        )
        results[name].to_csv(out_dir / f"{name}_folds.csv", index=False)

    pred_base = calibrated_r_predictions(raw["state"], df)
    pred_candidate = calibrated_r_predictions(raw[candidate_name], df)
    base_r = r_results(results["state"])
    candidate_r = r_results(results[candidate_name])
    comparison = bt.compare(
        base_r,
        candidate_r,
        name_a="state",
        name_b=f"state+{args.candidate}",
        preds_a=pred_base,
        preds_b=pred_candidate,
    )
    sigma = two_sigma(pred_base, pred_candidate)
    comparison["two_sigma"] = comparison.season.map(sigma)
    comparison["clears_two_sigma"] = comparison.delta > comparison.two_sigma
    accepted = bool(
        (comparison.delta > 0).all() and comparison.clears_two_sigma.all()
    )

    print("\n[R 구간 최종 판정]")
    print(
        comparison[["season", "bss_a", "bss_b", "delta", "two_sigma",
                    "clears_two_sigma"]].to_string(index=False)
    )
    print(f"\n최종: {'ACCEPT' if accepted else 'REJECT'}")

    report = {
        **CANDIDATES[args.candidate],
        "seeds": [int(s) for s in args.seeds],
        "coverage": float(coverage),
        "criterion": "R three folds positive and each delta > 2sigma",
        "accepted": accepted,
        "folds": comparison[
            ["season", "bss_a", "bss_b", "delta", "two_sigma",
             "clears_two_sigma"]
        ].to_dict(orient="records"),
    }
    (out_dir / "verdict.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())

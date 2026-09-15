"""hgb 멤버를 시즌 상태 피처로 재학습한다 — **반드시 .venv312 에서 실행**.

    .venv312/bin/python scripts/train_hgb_state.py <base.zip> <out.pkl>

**왜 별도 환경인가.** hgb 는 joblib/numpy **pickle** 로 저장된다. 로컬 numpy 2.x 로
pickle 하면 두 가지가 평가 서버(numpy 1.26.4)에서 깨진다:

  1. fitted 모델이 품는 `_feature_subsample_rng` (numpy Generator) —
     2.x 는 BitGenerator 를 클래스로 직렬화하는데 1.26 은 문자열을 기대한다.
     실제 제출 실패로 확인했다.
  2. **더 결정적:** numpy 2.x 는 `numpy._core.multiarray.scalar` 를 참조하는데
     1.26.4 에는 `numpy._core` 자체가 없다. RNG 를 제거해도 이건 남는다.

그래서 Python 3.12 + numpy 1.26.4 환경에서 만든다. 이 스크립트는 저장 후 pickle 이
실제로 요구하는 심볼을 전수 열거해 위험 참조가 없음을 확인한다.
"""
import json
import sys
import zipfile
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import season_state as ss  # noqa: E402
from src.champion import Champion  # noqa: E402


def verify_pickle(path):
    """pickle 이 요구하는 심볼을 전수 열거하고 위험 참조를 막는다."""
    from joblib.numpy_pickle import NumpyUnpickler

    class Spy(NumpyUnpickler):
        seen = set()

        def find_class(self, mod, name):
            Spy.seen.add(f"{mod}.{name}")
            return super().find_class(mod, name)

    Spy.seen = set()
    with open(path, "rb") as f:
        Spy(str(path), f, ensure_native_byte_order=True).load()
    syms = sorted(Spy.seen)
    bad = [s for s in syms if "_core" in s or "random" in s or "Generator" in s]
    print(f"  pickle 심볼 {len(syms)}개, numpy: "
          f"{[s for s in syms if s.startswith('numpy')]}")
    if bad:
        raise SystemExit(f"평가 서버에서 못 읽을 참조: {bad}")
    print("  위험 참조 없음 ✓")


def main():
    base, out = Path(sys.argv[1]), Path(sys.argv[2])
    df = pd.read_csv(ROOT / "data" / "train.csv", encoding="utf-8-sig")
    y = df["control_success"].to_numpy(dtype="float64")
    print(f"train {df.shape}", flush=True)

    STATE = ss.add_features_by_season(df)
    ch = Champion(base)
    X = pd.concat([ch.frame(df, "hgb"), STATE], axis=1)
    print(f"features {X.shape[1]} (기존 {X.shape[1]-STATE.shape[1]} + 상태 "
          f"{STATE.shape[1]})", flush=True)

    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(
        categorical_features="from_dtype", random_state=42,
        **ch.members["hgb"]["model_params"])
    m.fit(X, y)
    probe = float(m.predict_proba(X.head(3000))[:, 1].mean())

    # 학습에만 쓰이는 RNG 를 떼어낸다 — 예측 경로는 건드리지 않는다(검증됨).
    m._feature_subsample_rng = None
    joblib.dump(m, out)
    verify_pickle(out)

    (out.with_suffix(".cols.json")).write_text(
        json.dumps(list(X.columns), ensure_ascii=False), encoding="utf-8")
    print(f"saved {out}  (probe mean {probe:.4f})")


if __name__ == "__main__":
    main()

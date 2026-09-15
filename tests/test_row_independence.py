"""대회 규칙 3절을 기계적으로 강제하는 테스트.

> "test.csv에 해당 행 1개만 있는 경우"와 "전체 평가 데이터가 함께 있는 경우"의
> 예측값이 동일해야 한다. 달라진다면 다른 행이 추론에 영향을 준 것이다.

이 테스트는 **실제 배포 ZIP**을 풀어 그 안의 `script.predict` 를 그대로 호출한다.
소스 트리가 아니라 제출물 자체를 검사하는 것이 요점이다.

  OMP_NUM_THREADS=1 python -m pytest tests/test_row_independence.py -q
  OMP_NUM_THREADS=1 python tests/test_row_independence.py artifacts/submit_corrections.zip

**`OMP_NUM_THREADS=1` 이 필요하다.** macOS 로컬에서 torch·lightgbm·catboost 가
각자의 OpenMP 런타임을 들고 오면 첫 예측에서 교착한다(CPU 0%로 멈춤). 평가
서버 문제가 아니라 로컬 환경 문제이며, 스레드를 1로 묶으면 사라진다.
`ROWINDEP_N` 으로 표본 크기, `ROWINDEP_VERBOSE=1` 로 진행 로그를 조절한다.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = Path(os.environ.get(
    "ROWINDEP_ZIP", ROOT / "artifacts" / "submit_corrections.zip"
))
TRAIN = Path(os.environ.get("ROWINDEP_TRAIN", ROOT / "data" / "train.csv"))
N_ROWS = int(os.environ.get("ROWINDEP_N", "3000"))
SEED = 0
VERBOSE = bool(os.environ.get("ROWINDEP_VERBOSE"))


def _log(msg):
    if VERBOSE:
        print(msg, flush=True)


def _assert_bit_equal(actual, expected, message):
    """비트 불일치 시 건수와 크기를 남겨 구조적 의존과 부동소수 잡음을 구분한다."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if np.array_equal(actual, expected):
        return
    delta = np.abs(actual.astype("float64") - expected.astype("float64"))
    raise AssertionError(
        f"{message}: 불일치 {np.count_nonzero(delta):,}/{delta.size:,}, "
        f"max|Δ|={delta.max():.3e}, mean|Δ|={delta.mean():.3e}"
    )


@contextlib.contextmanager
def _chdir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _load_script(work):
    spec = importlib.util.spec_from_file_location("submit_script",
                                                  work / "script.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(work))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def _sample_rows():
    """평가 데이터와 같은 컬럼 구조(타깃 없음)의 표본을 train 에서 만든다.

    배포된 test.csv 는 5행뿐이라 카운트·좌우·투수를 충분히 덮지 못한다.
    """
    df = pd.read_csv(TRAIN, encoding="utf-8-sig")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=min(N_ROWS, len(df)), replace=False)
    sub = df.iloc[np.sort(idx)].drop(columns=["control_success"]).copy()
    sub["row_id"] = [f"TEST_{i:06d}" for i in range(len(sub))]
    return sub.reset_index(drop=True)


def run_checks(zip_path=DEFAULT_ZIP):
    zip_path = Path(zip_path)
    _log("sampling rows ...")
    rows = _sample_rows()
    _log(f"  {len(rows)} rows")
    with tempfile.TemporaryDirectory(prefix="rowindep_") as tmp:
        work = Path(tmp)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(work)
        with _chdir(work):
            _log("loading script ...")
            script = _load_script(work)
            import json
            with open("./model/meta.json", encoding="utf-8") as f:
                meta = json.load(f)
            _log("loading models ...")
            models = script.load_models("./model", meta)
            _log("  models loaded")

            n_calls = [0]

            def predict(frame):
                n_calls[0] += 1
                _log(f"  predict #{n_calls[0]} (n={len(frame)})")
                return np.asarray(script.predict(frame.reset_index(drop=True),
                                                 models, meta, verbose=False),
                                  dtype="float64")

            full = predict(rows)

            # (1) 순서를 섞어도 각 행의 값은 그대로여야 한다
            perm = np.random.default_rng(1).permutation(len(rows))
            shuffled = predict(rows.iloc[perm])
            _assert_bit_equal(shuffled, full[perm], "순서에 의존한다")

            # (2) 부분집합만 넣어도 그 행들의 값은 그대로여야 한다
            for frac in (0.5, 0.1):
                take = np.sort(np.random.default_rng(2).choice(
                    len(rows), size=int(len(rows) * frac), replace=False))
                _assert_bit_equal(
                    predict(rows.iloc[take]), full[take],
                    f"부분집합({frac})에서 값이 달라진다",
                )

            # (3) 1행만 넣어도 같아야 한다 — 규칙이 명시한 판정 기준
            for i in np.random.default_rng(3).choice(len(rows), 12, replace=False):
                one = predict(rows.iloc[[i]])
                _assert_bit_equal(
                    one, full[[i]], f"단일 행 {i}: {one[0]!r} != {full[i]!r}"
                )

            # (4) 다른 행을 바꿔치기해도 대상 행은 영향받지 않아야 한다
            mutated = rows.copy()
            keep = 7
            other = mutated.index != keep
            mutated.loc[other, "pitcher_id"] = 999999
            mutated.loc[other, "balls_before"] = 3
            assert predict(mutated)[keep] == full[keep], \
                "다른 행의 값이 대상 행의 예측을 바꾼다"
    return len(rows)


def test_row_independence():
    if not DEFAULT_ZIP.exists():
        import pytest
        pytest.skip(f"{DEFAULT_ZIP} 없음 — scripts/build_corrections.py 먼저 실행")
    run_checks()


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ZIP
    n = run_checks(target)
    print(f"OK — {target.name}: {n}행에 대해 순서/부분집합/단일행/변조 4종 통과")

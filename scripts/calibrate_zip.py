"""제출 ZIP 의 최종 수준 보정 상수를 다시 맞춘다.

**왜 필요한가.** 챔피언의 스테이지 상수(`ours` 의 `b=-0.0509`)는 **그때 모델들의
출력 분포**에 맞춰 적합된 값이다. 멤버를 재학습하면 분포가 달라져 그 상수가 낡고,
2025 예측 평균이 목표 기저율(`calib_details.ours.r_hat`, 현재 0.4747)에서 벗어난다.
수준 오차는 Brier 에 제곱으로 들어와 웬만한 이득을 통째로 지운다.

**절차 (챔피언과 동일).** 2024 행에서 파이프라인의 블렌드 평균을 구하고, 그것을
`r_hat` 으로 보내는 로짓 상수 하나를 찾아 `meta["final_logit_shift"]` 에 넣는다.
2024 행의 시즌 상태 피처는 경계표 <=2023 으로 계산되므로, 실제 2025 행이 <=2024
경계표로 갖게 될 구조와 같다.

**규칙 준수.** 상수는 train 데이터에서만 나오고 모든 행에 동일하게 적용된다.
평가 데이터의 분포는 쓰지 않는다.

**왜 빌드와 분리했나.** 이 단계는 NN 예측이 필요해 torch 를 부른다. macOS 에서
`OMP_NUM_THREADS>1` 로 torch + lightgbm + catboost 가 각자의 OpenMP 런타임을
들고 오면 교착한다. 학습은 다중 스레드가 필요하므로 한 프로세스에 둘 수 없다.

    OMP_NUM_THREADS=1 python scripts/calibrate_zip.py artifacts/submit_season_state.zip
"""
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
TRAIN = ROOT / "data" / "train.csv"
REF_SEASON = 2024


def load_script(work):
    spec = importlib.util.spec_from_file_location("zscript", work / "script.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(work))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def main(argv=None):
    zip_path = Path(argv[0] if argv else sys.argv[1])
    df = pd.read_csv(TRAIN, encoding="utf-8-sig")
    ref = df[df.season == REF_SEASON].drop(columns=["control_success"]).reset_index(
        drop=True)
    print(f"reference: season {REF_SEASON}, {len(ref):,} rows", flush=True)

    with tempfile.TemporaryDirectory(prefix="calzip_") as tmp:
        work = Path(tmp)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(work)
        cwd = os.getcwd()
        os.chdir(work)
        try:
            script = load_script(work)
            meta = json.loads(Path("./model/meta.json").read_text("utf-8"))
            r_hat = float(meta["calib_details"]["ours"]["r_hat"])
            meta["final_logit_shift"] = 0.0          # 우선 0 으로 두고 측정
            Path("./model/meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8")
            models = script.load_models("./model", meta)
            p = np.asarray(script.predict(ref, models, meta, verbose=False),
                           dtype="float64")
        finally:
            os.chdir(cwd)

    print(f"  현재 {REF_SEASON} 예측 평균 {p.mean():.6f}   목표 r_hat {r_hat:.6f}")
    lo, hi = -3.0, 3.0
    z_ = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if (1 / (1 + np.exp(-(z_ + mid)))).mean() < r_hat:
            lo = mid
        else:
            hi = mid
    shift = 0.5 * (lo + hi)
    after = (1 / (1 + np.exp(-(z_ + shift)))).mean()
    print(f"  shift {shift:+.6f}  ->  보정 후 평균 {after:.6f}")

    # ZIP 안의 meta.json 만 교체한다
    with tempfile.TemporaryDirectory(prefix="calzip2_") as tmp:
        work = Path(tmp)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(work)
        meta = json.loads((work / "model" / "meta.json").read_text("utf-8"))
        meta["final_logit_shift"] = shift
        meta.setdefault("season_state", {})["recalibrated"] = (
            f"{REF_SEASON} blend mean {p.mean():.6f} -> r_hat {r_hat:.6f} "
            f"via logit shift {shift:+.6f}")
        (work / "model" / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8")
        tmp_zip = zip_path.with_suffix(".tmp.zip")
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(work.rglob("*")):
                if f.is_file() and "__pycache__" not in f.parts:
                    z.write(f, f.relative_to(work))
        shutil.move(tmp_zip, zip_path)
    print(f"updated {zip_path}")


if __name__ == "__main__":
    main()

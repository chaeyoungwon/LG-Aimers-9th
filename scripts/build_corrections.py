"""Put the platoon / count / leverage corrections on the champion ZIP.

**검증 경로.** 기존 보정층(season_form, batter_form)과 같은 team-proxy forward
OOF 방식이지만 폴드를 1개가 아니라 3개 쓴다. 전체 행 기준 델타는
2022 +28.8 / 2023 +10.3 / 2024 +34.7 (평균 +24.6, 결합 z=6.59) 이고 세 폴드 모두
양수다. 다만 프록시는 챔피언의 team 스테이지(블렌드의 60%)만 근사하므로 실제
LB 이득은 이보다 작다 — 타자 form의 전례가 프록시 +25.67 → LB +12.2 로 약 절반
이었다. 그 배율이면 +15~20 이 예상 범위다.

**규정 준수.** 세 표 모두 train(<=2024) 잔차에서만 만들어지고, 추론 시 각 행은
자기 행의 값으로 조회만 한다. 평가 데이터의 다른 행·누적·분포는 쓰지 않는다.
"""
import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import backtest as bt
from src import corrections as co
from src import season_form as sf
from src.train_base import MODEL_CONFIGS

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts" / "submit_r10_pitcher_w418194.zip"
OUT = ROOT / "artifacts" / "submit_corrections.zip"
TRAIN = ROOT / "data" / "train.csv"
CACHE = ROOT / "artifacts" / "backtest" / "full_raw.npz"

# 챔피언과 동일한 season-form 상수 (model/meta.json 의 season_form 블록)
PITCHER_FORM = dict(m=75, alpha=0.28)
BATTER_FORM = dict(m=50, alpha=0.11250648296393913)


def team_proxy_refs(df):
    """시즌 s 를 시즌 <s 로 학습한 모델로 예측한 forward-OOF 참조값.

    챔피언의 team 스테이지(LGBM 앙상블 + season-form)를 근사한다. 이 예측이
    out-of-sample 이어야 잔차에 선수별 구조가 남는다 — 그 시즌을 학습한 모델의
    잔차는 이미 0에 가깝다.
    """
    if CACHE.exists():
        z = np.load(CACHE)
        raw = {int(k): z[k] for k in z.files}
        print(f"loaded cached raw preds: seasons={sorted(raw)}")
    else:
        print("no cache; training fold models (~15min) ...")
        raw = bt.fit_raw(bt.baseline_builder, "full", configs=MODEL_CONFIGS,
                         df=df, out_dir=str(CACHE.parent))
    se = df.season.astype(int).to_numpy()
    gt = df.game_type.to_numpy()
    y = df.control_success.to_numpy(dtype="float64")

    ref = {}
    for t in sorted(raw):
        if t - 1 not in raw:
            continue
        mp = se == t - 1
        a, b = bt._fit_calibration(raw[t - 1][gt[mp] == "R"], y[mp][gt[mp] == "R"])
        p = bt._sigmoid(a * bt._logit(raw[t]) + b)
        ts = [s for s in bt.SEASONS if s < t]
        rows = df[df.season == t]
        for who, cfg in (("pitcher", PITCHER_FORM), ("batter", BATTER_FORM)):
            p = sf.apply_form(p, rows, sf.build_table(df, ts, who), cfg["m"],
                              cfg["alpha"], sf.fit_mu(df, ts, cfg["m"], who), who)
        ref[t] = p
    return ref


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not BASE.exists():
        raise FileNotFoundError(BASE)

    df = pd.read_csv(TRAIN, encoding="utf-8-sig")
    ref = team_proxy_refs(df)
    train_seasons = sorted(df.season.astype(int).unique())
    tables = co.build_all(df, ref, train_seasons)

    spec = {
        "items": [{"name": n, **tables[n]} for n, _, _, _ in co.SPEC],
        "source": {
            "base": BASE.name,
            "residuals": f"team-proxy forward OOF, seasons {sorted(ref)}",
            "validation": ("rolling 3-fold, all-rows delta "
                           "2022 +28.8 / 2023 +10.3 / 2024 +34.7 "
                           "(mean +24.6, combined z=6.59)"),
            "expected_lb": ("proxy overstates: batter-form precedent was "
                            "proxy +25.67 -> LB +12.2 (~0.5x); expect +15~20"),
            "row_independent": True,
        },
    }
    for item in spec["items"]:
        print(f"  {item['name']:>16}: {len(item['table']):>5} entries  "
              f"k={item['k']:.0f}  key={item['key']}")

    call = 'apply_coldstart_expert(preds, test, "./model")'
    with tempfile.TemporaryDirectory(prefix="corrections_") as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(BASE) as z:
            z.extractall(tmp)
        (tmp / "model" / "corrections.json").write_text(
            json.dumps(spec, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8")
        (tmp / "corrections_runtime.py").write_text(
            (ROOT / "scripts" / "corrections_runtime.py").read_text(encoding="utf-8"),
            encoding="utf-8")

        path = tmp / "script.py"
        script = path.read_text(encoding="utf-8")
        member_anchor = "\n        return " + call
        plain_anchor = "\n    return " + call
        if script.count(member_anchor) != 1 or script.count(plain_anchor) != 1:
            raise SystemExit("unexpected coldstart return anchors")
        script = script.replace(
            "import pandas as pd\n",
            "import pandas as pd\n"
            "from corrections_runtime import apply_corrections\n", 1)
        script = script.replace(
            member_anchor,
            "\n        preds = " + call
            + '\n        return apply_corrections(preds, test, "./model")', 1)
        script = script.replace(
            plain_anchor,
            "\n    preds = " + call
            + '\n    return apply_corrections(preds, test, "./model")', 1)
        path.write_text(script, encoding="utf-8")

        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(tmp.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    z.write(p, p.relative_to(tmp))
    print(f"built {out}")


if __name__ == "__main__":
    main()

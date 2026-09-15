"""시즌 상태 피처를 5멤버 전부에 적용한 제출물을 조립한다.

베이스는 `submit_season_state.zip` (cat·team 은 이미 상태 피처 적용). 여기에
  - hgb : .venv312 에서 만든 pkl (numpy 1.26.4 경로, RNG 제거 검증됨)
  - nn / team_nn : 재학습한 npz + 재생성한 prep/vocab/arch
를 얹는다.

수준 보정은 **넣지 않는다.** 2024 기준 행으로 수준을 맞추면 이중 보정이 된다는 것을
17차 실패로 배웠다 — 상태 피처가 있으면 모델이 드리프트를 스스로 흡수한다.
수준은 `scripts/make_proxy_2025.py` 로 사후 확인한다.
"""
import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts" / "submit_season_state.zip"
OUT = ROOT / "artifacts" / "submit_full_state.zip"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hgb", default="/tmp/hgb_state.pkl")
    ap.add_argument("--nn-dir", default="/tmp/nn_state")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    out, nn_dir = Path(a.out), Path(a.nn_dir)

    with tempfile.TemporaryDirectory(prefix="fullstate_") as tmp:
        w = Path(tmp)
        with zipfile.ZipFile(BASE) as z:
            z.extractall(w)
        meta = json.loads((w / "model" / "meta.json").read_text("utf-8"))
        members = {m["name"]: m for m in meta["members"]}

        # ---- hgb -----------------------------------------------------------
        shutil.copy(a.hgb, w / "model" / "hgb_model.pkl")
        cols = json.loads(Path(a.hgb).with_suffix(".cols.json").read_text("utf-8"))
        members["hgb"]["feature_columns"] = cols
        print(f"  hgb  : {len(cols)} features")

        # ---- nn / team_nn ---------------------------------------------------
        specs = json.loads((nn_dir / "nn_specs.json").read_text("utf-8"))
        for name, s in specs.items():
            shutil.copy(nn_dir / f"{name}_model.npz",
                        w / "model" / members[name]["model_files"][0])
            members[name]["feature_columns"] = s["feature_columns"]
            members[name]["nn"] = s["nn"]
            print(f"  {name:<7}: {len(s['feature_columns'])} features, "
                  f"{s['nn']['arch']['n_features']} prepped, "
                  f"batch={s['nn']['runtime']['batch_size']}")

        meta["final_logit_shift"] = 0.0
        meta.setdefault("season_state", {}).update({
            "applied_to": ["hgb", "cat", "nn", "team", "team_nn"],
            "hgb_build": ("Python 3.12 + numpy 1.26.4 (.venv312); "
                          "_feature_subsample_rng 제거 후 pickle 심볼 전수 검증"),
            "nn_build": "prep/vocab/arch 재생성, runtime.batch_size 는 원본 유지",
            "level": ("보정 없음 — 2024 기준으로 맞추면 이중 보정이 된다 "
                      "(17차 실패). make_proxy_2025.py 로 사후 확인"),
        })
        (w / "model" / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8")

        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(w.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    z.write(p, p.relative_to(w))
    print(f"built {out}")


if __name__ == "__main__":
    main()

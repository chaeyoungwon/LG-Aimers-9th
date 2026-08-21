# NOTE: 이 도구를 진작 만들었어야 했다. "2024 행의 season 만 2025 로 바꾼" 합성
# 리허설은 시즌 상태 피처를 전부 결측으로 만들어 수준 검증이 불가능했고, 그 결과
# 이중 보정을 못 잡아 제출 하나를 날렸다 (899.15).
"""진짜 2025 처럼 생긴 평가 프록시를 만들어 파이프라인 수준을 비교한다.

지금까지 리허설은 2024 행의 `season` 만 2025 로 바꾼 합성이었다. 그러면
`asof_n < n0(2024말)` 이라 시즌 상태 피처가 전부 결측이 되어, 실제 2025 와
전혀 다른 입력이 된다. 수준 검증이 불가능했던 이유다.

제대로 만들려면 asof 값 자체를 옮겨야 한다. 2024 행의 "2024 시즌 상태"를
그대로 "2025 시즌 상태"로 이식한다:

    ns = asof_n - n0(2023말)          # 원래 행의 2024 시즌 투구수
    asof_n'    = n0(2024말) + ns      # 2025 행이라면 가졌을 값
    count'     = c0(2024말) + (count - c0(2023말))
    rate'      = count' / asof_n'

이러면 커리어 누적도 시즌 상태도 실제 2025 행과 같은 구조가 된다.
"""
import numpy as np
import pandas as pd

from src import season_state as ss

S = "/private/tmp/claude-501/-Users-yeonho-LG-Aimers-9th/9b152936-a8da-41b4-9201-46bb85cd2077/scratchpad"
df = pd.read_parquet(f"{S}/train.parquet")

B23 = ss.build_boundary(df, [2019, 2020, 2021, 2022, 2023])
B24 = ss.build_boundary(df, [2019, 2020, 2021, 2022, 2023, 2024])


def lookup(tab, ids_query):
    o = np.argsort(tab["ids"])
    ids = tab["ids"][o]
    k = np.where(np.isfinite(ids_query), ids_query, -1).astype("int64")
    pos = np.searchsorted(ids, k)
    clip = np.clip(pos, 0, len(ids) - 1)
    seen = (pos < len(ids)) & (ids[clip] == k)
    return seen, o, clip


def build_proxy():
    out = df[df.season == 2024].drop(columns=["control_success"]).reset_index(drop=True)
    out["season"] = 2025
    for name, idcol, ncol, rates in ss.GROUPS:
        t23, t24 = B23[name], B24[name]
        q = out[idcol].to_numpy(dtype="float64")
        s23, o23, c23 = lookup(t23, q)
        s24, o24, c24 = lookup(t24, q)
        both = s23 & s24
        n = out[ncol].to_numpy(dtype="float64")
        n0a = np.where(s23, t23["n0"][o23][c23], np.nan)
        n0b = np.where(s24, t24["n0"][o24][c24], np.nan)
        ns = n - n0a
        newn = np.where(both & (ns > 0), n0b + ns, n)   # 매핑 불가하면 원값 유지
        for k, col in rates.items():
            r = out[col].to_numpy(dtype="float64")
            c = np.rint(n * np.where(np.isfinite(r), r, 0.0))
            c0a = np.where(s23, t23["c0"][k][o23][c23], np.nan)
            c0b = np.where(s24, t24["c0"][k][o24][c24], np.nan)
            newc = c0b + (c - c0a)
            with np.errstate(invalid="ignore", divide="ignore"):
                nr = newc / newn
            out[col] = np.where(both & (ns > 0) & np.isfinite(nr), nr, r)
        out[ncol] = newn
    return out


if __name__ == "__main__":
    import importlib.util, json, os, sys, tempfile, zipfile
    from pathlib import Path

    ROOT = Path("/Users/yeonho/LG-Aimers-9th")
    proxy = build_proxy()
    proxy["row_id"] = [f"TEST_{i:06d}" for i in range(len(proxy))]
    print(f"proxy rows {len(proxy):,}", flush=True)

    # 시즌 상태 피처가 실제로 채워지는지 확인
    spec_tab = B24
    F = ss.add_features(proxy, spec_tab)
    print(f"  상태피처 결측 아닌 비율 {F.cur_p_succ.notna().mean():.3f} "
          f"(실제 2024 에서는 0.801)")
    print(f"  cur_p_n 중앙값 {F.cur_p_n.median():.0f}", flush=True)

    for name in ("submit_corrections.zip", "submit_season_state.zip"):
        with tempfile.TemporaryDirectory() as tmp:
            w = Path(tmp)
            with zipfile.ZipFile(ROOT / "artifacts" / name) as z:
                z.extractall(w)
            cwd = os.getcwd()
            os.chdir(w)
            sys.path.insert(0, str(w))
            try:
                sp = importlib.util.spec_from_file_location("s", w / "script.py")
                m = importlib.util.module_from_spec(sp)
                sp.loader.exec_module(m)
                meta = json.load(open("./model/meta.json", encoding="utf-8"))
                p = np.asarray(m.predict(proxy, m.load_models("./model", meta),
                                         meta, verbose=False), dtype=float)
                sh = meta.get("final_logit_shift", 0.0)
                z_ = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
                p0 = 1 / (1 + np.exp(-(z_ - sh)))
            finally:
                os.chdir(cwd)
                sys.path.pop(0)
        print(f"{name:>28}: 프록시2025 평균 {p.mean():.6f}"
              f"   (shift 제거 시 {p0.mean():.6f})", flush=True)

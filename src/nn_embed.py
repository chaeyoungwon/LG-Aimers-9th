"""선수 임베딩 신경망 — Round 1에서 버린 id 정보를 다시 여는 프로토타입.

**왜 이걸 만드는가.** Round 1 실측에서 `pitcher_id`/`batter_id`를 raw 숫자로
트리에 넣으면 성능이 떨어졌다(579 -> 502). 하지만 그건 "선수 정체성에 정보가
없다"는 뜻이 아니라 **트리가 쓸 수 있는 형태가 아니었다**는 뜻이다. 축 정렬 분기는
id 축을 구간으로 쪼개는데, id 번호는 순서에 아무 의미가 없어서 모든 분기가
무작위 그룹핑이 된다. 792명을 의미 있게 가르려면 분기가 수백 개 필요하고,
그 깊이는 곧 과적합이다.

임베딩은 정확히 그 자리를 메운다. 각 선수에게 학습되는 8~16차원 벡터를 주면
"이 투수는 어떤 투수인가"가 **연속 공간의 좌표**가 되고, 비슷한 선수는 가까운
좌표로 수렴한다. 792 x 16 = 12,672개 파라미터로 792개의 분기 대신 하나의
연속 축을 얻는 셈이다. `asof_*` 집계는 선수의 **평균**만 담지만 임베딩은
평균으로 요약되지 않는 잔차(구종 조합, 특정 카운트의 습관 등)를 담을 수 있다.

**2025 신인 문제와 UNK 학습.** 평가 데이터(2025)에는 train에 없는 선수가 섞여
있다(배포 샘플 5행에 이미 24713/24745가 등장 — train 최대 24633). 임베딩 테이블에
없는 id는 UNK 슬롯으로 떨어지는데, 학습 중에 UNK가 한 번도 안 나오면 그 슬롯은
초기값 그대로 남아 신인 행의 예측이 통째로 망가진다. 그래서 학습 중 각 배치에서
확률 `id_dropout`으로 id를 UNK로 치환한다 — word dropout과 같은 장치다. UNK
임베딩은 "선수를 모를 때의 평균적인 선수"로 학습되고, 동시에 모델이 임베딩에만
의존하지 못하게 하는 정규화로도 작동한다.

**행 단위 독립성 (규칙 준수).** `TabularPrep`/`IdVocab`의 통계(중앙값, 평균,
표준편차, 레벨 목록, id 사전)는 전부 **fit 시점의 학습 데이터**에서만 나온다.
transform은 그 상수를 각 행에 적용할 뿐이라 같은 배치에 무엇이 들어왔는지에
의존하지 않는다. `tests/test_nn_embed.py`의 순열/부분집합 테스트가 이를 고정한다.

torch는 **lazy import**한다. 전처리(IdVocab/TabularPrep)는 numpy/pandas만으로
동작해야 torch 없는 환경에서도 단위 테스트가 돈다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# UNK 슬롯은 항상 0번이다. 0으로 고정하면 "id를 모른다"를 표현할 때 dtype/장치와
# 무관하게 `torch.zeros_like(idx)`로 쓸 수 있고, 실수로 실제 선수를 가리킬 일이 없다.
UNK_INDEX = 0


@dataclass
class IdVocab:
    """선수 id -> 임베딩 행 번호. 0번은 UNK 고정, 실제 선수는 1번부터.

    **학습 데이터에서만** fit한다. 검증/평가에서 처음 보는 id는 조용히 UNK로
    떨어진다 — 예외를 내지 않는 것이 의도다. 2025에 신인이 섞이는 것은 정상
    상황이고, 그 행도 수치 피처만으로 예측이 나와야 한다.
    """

    ids_: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="int64"))

    @classmethod
    def fit(cls, values) -> "IdVocab":
        arr = pd.Series(values).dropna().to_numpy(dtype="int64")
        return cls(ids_=np.unique(arr))

    @property
    def size(self) -> int:
        """임베딩 테이블의 행 수 = 실제 선수 수 + UNK 1개."""
        return len(self.ids_) + 1

    def transform(self, values) -> np.ndarray:
        """미등장 id와 결측은 UNK_INDEX(0). 값 하나만 보고 정해지는 행 단위 변환이다."""
        arr = pd.Series(values).to_numpy(dtype="float64")
        if len(self.ids_) == 0:
            return np.full(len(arr), UNK_INDEX, dtype="int64")
        # searchsorted는 정렬된 ids_에 대한 이진탐색이라 150만 행에서도 즉시 끝난다.
        # 결측은 먼저 -1 같은 사전에 없는 값으로 떨어뜨려 동일 경로로 처리한다.
        na = ~np.isfinite(arr)
        key = np.where(na, -1, arr).astype("int64")
        pos = np.searchsorted(self.ids_, key)
        pos_clipped = np.clip(pos, 0, len(self.ids_) - 1)
        hit = (pos < len(self.ids_)) & (self.ids_[pos_clipped] == key) & ~na
        return np.where(hit, pos + 1, UNK_INDEX).astype("int64")


class TabularPrep:
    """수치/범주 피처 -> 신경망이 먹을 수 있는 float32 밀집 행렬.

    트리와 달리 신경망은 (a) 결측을 스스로 처리하지 못하고 (b) 스케일이 제각각이면
    학습이 망가진다. 그래서 두 가지를 fit 시점 상수로 굳혀 둔다.

    수치 컬럼
      중앙값으로 대치하고 (x - mean) / std 로 표준화한다. 평균/표준편차는
      **대치 이후** 값에서 잰다(대치값이 곧 0 근처로 가야 결측 행이 극단에 놓이지
      않는다). 학습 데이터에서 결측이 있었던 컬럼은 결측 플래그를 따로 붙인다 —
      "값을 모른다"는 것 자체가 신호이기 때문이다(cold-start 선수).

    범주 컬럼
      **고정 레벨** one-hot. 레벨 목록은 fit 시점의 dtype categories(또는 정렬된
      고유값)에서 온다. 미지 값/결측은 전부 0 벡터가 된다 — 트리 경로에서 미지
      레벨을 NaN으로 떨어뜨리는 것과 같은 처리다.

    카디널리티가 큰 id 컬럼은 여기 들어오지 않는다(그건 임베딩 담당이다).
    안전을 위해 `max_levels`를 넘는 컬럼이 있으면 예외를 낸다 — one-hot이 조용히
    수백 차원으로 부푸는 사고를 막는다.
    """

    def __init__(self, max_levels: int = 64):
        self.max_levels = max_levels
        self.num_cols_: list[str] = []
        self.cat_cols_: list[str] = []
        self.levels_: dict[str, list] = {}
        self.median_: dict[str, float] = {}
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}
        self.na_cols_: list[str] = []
        self.feature_names_: list[str] = []

    @staticmethod
    def _is_cat(s: pd.Series) -> bool:
        return isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object

    def fit(self, X: pd.DataFrame) -> "TabularPrep":
        self.num_cols_ = []
        self.cat_cols_ = []
        for c in X.columns:
            (self.cat_cols_ if self._is_cat(X[c]) else self.num_cols_).append(c)

        for c in self.cat_cols_:
            s = X[c]
            if isinstance(s.dtype, pd.CategoricalDtype):
                levels = list(s.dtype.categories)
            else:
                levels = sorted(s.dropna().unique().tolist())
            if len(levels) > self.max_levels:
                raise ValueError(
                    f"{c!r}의 레벨이 {len(levels)}개다 (한도 {self.max_levels}). "
                    f"고카디널리티 컬럼은 one-hot이 아니라 임베딩으로 넣어야 한다")
            self.levels_[c] = levels

        self.na_cols_ = []
        for c in self.num_cols_:
            v = X[c].to_numpy(dtype="float64", na_value=np.nan)
            finite = np.isfinite(v)
            if not finite.all():
                self.na_cols_.append(c)
            if not finite.any():
                raise ValueError(f"{c!r}가 학습 데이터에서 전부 결측이다 — "
                                 f"중앙값을 정할 수 없다")
            med = float(np.median(v[finite]))
            filled = np.where(finite, v, med)
            self.median_[c] = med
            self.mean_[c] = float(filled.mean())
            # 상수 컬럼은 std=0이라 그대로 나누면 inf/NaN이 된다. 1로 두면
            # 표준화 결과가 전부 0이 되어 "정보 없음"이 올바르게 표현된다.
            sd = float(filled.std())
            self.std_[c] = sd if sd > 1e-12 else 1.0

        self.feature_names_ = (
            list(self.num_cols_)
            + [f"{c}__isna" for c in self.na_cols_]
            + [f"{c}={lv}" for c in self.cat_cols_ for lv in self.levels_[c]]
        )
        return self

    @property
    def n_features(self) -> int:
        return len(self.feature_names_)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.feature_names_:
            raise RuntimeError("fit을 먼저 호출해야 한다")
        missing = [c for c in self.num_cols_ + self.cat_cols_
                   if c not in X.columns]
        if missing:
            raise KeyError(f"학습 때 있던 컬럼이 없다: {missing}")

        n = len(X)
        out = np.empty((n, self.n_features), dtype="float32")
        j = 0
        for c in self.num_cols_:
            v = X[c].to_numpy(dtype="float64", na_value=np.nan)
            finite = np.isfinite(v)
            v = np.where(finite, v, self.median_[c])
            out[:, j] = (v - self.mean_[c]) / self.std_[c]
            j += 1
        for c in self.na_cols_:
            v = X[c].to_numpy(dtype="float64", na_value=np.nan)
            out[:, j] = (~np.isfinite(v)).astype("float32")
            j += 1
        for c in self.cat_cols_:
            levels = self.levels_[c]
            # 값으로 다시 매핑한다 — X의 dtype categories가 fit 때와 달라도
            # (순서가 섞이거나 일부만 관측돼도) 같은 열에 떨어지게 만든다.
            codes = pd.Categorical(X[c], categories=levels).codes
            block = np.zeros((n, len(levels)), dtype="float32")
            hit = codes >= 0
            block[np.nonzero(hit)[0], codes[hit]] = 1.0
            out[:, j:j + len(levels)] = block
            j += len(levels)
        assert j == self.n_features
        return out


# --- 이하 torch 의존 ---

def _torch():
    import torch
    return torch


def make_id_dropout_mask(shape, p: float, generator=None, device=None):
    """확률 p로 True인 마스크 — True인 자리의 id를 UNK로 바꾼다.

    학습 루프에서 매 배치 새로 뽑는다. 같은 행이 에폭마다 다른 취급을 받으므로
    "이 선수를 알 때"와 "모를 때" 둘 다의 예측을 모델이 배우게 된다.
    """
    torch = _torch()
    if p <= 0.0:
        return torch.zeros(shape, dtype=torch.bool, device=device)
    if p >= 1.0:
        return torch.ones(shape, dtype=torch.bool, device=device)
    r = torch.rand(shape, generator=generator, device=device)
    return r < p


def apply_id_dropout(idx, p: float, generator=None):
    """id 텐서의 일부를 UNK_INDEX로 치환한 새 텐서."""
    torch = _torch()
    if p <= 0.0:
        return idx
    mask = make_id_dropout_mask(idx.shape, p, generator=generator,
                                device=idx.device)
    return torch.where(mask, torch.full_like(idx, UNK_INDEX), idx)


def build_model(n_features: int, n_pitcher: int, n_batter: int, *,
                emb_dim: int = 8, hidden=(128, 64), dropout: float = 0.1,
                seed: int = 42):
    """수치 피처 + 투수/타자 임베딩 -> MLP -> logit.

    임베딩은 **작은 값(std 0.01)으로 초기화**한다. 학습 초기에는 임베딩 기여가
    거의 0이라 모델이 사실상 "수치 피처만 쓰는 MLP"에서 출발하고, 임베딩은
    수치 피처가 설명하지 못하는 잔차를 채우는 방향으로만 자란다. 큰 초기값을 주면
    랜덤 좌표가 초기 손실을 지배해 학습이 그 노이즈를 쫓아간다.
    """
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(seed)

    class EmbedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.pitcher_emb = nn.Embedding(n_pitcher, emb_dim)
            self.batter_emb = nn.Embedding(n_batter, emb_dim)
            nn.init.normal_(self.pitcher_emb.weight, std=0.01)
            nn.init.normal_(self.batter_emb.weight, std=0.01)
            layers, d = [], n_features + 2 * emb_dim
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
                d = h
            layers.append(nn.Linear(d, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x_num, pid, bid):
            z = torch.cat([x_num, self.pitcher_emb(pid), self.batter_emb(bid)],
                          dim=1)
            return self.mlp(z).squeeze(1)

    return EmbedMLP()


def _split_holdout(n: int, val_frac: float, seed: int) -> np.ndarray:
    """학습 행 중 홀드아웃으로 뺄 인덱스의 boolean 마스크 (True=홀드아웃)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)[: int(round(n * val_frac))]
    m = np.zeros(n, dtype=bool)
    m[idx] = True
    return m


def train_embed_nn(X_num: np.ndarray, pid: np.ndarray, bid: np.ndarray,
                   y: np.ndarray, *, n_pitcher: int, n_batter: int,
                   emb_dim: int = 8, hidden=(128, 64), dropout: float = 0.1,
                   id_dropout: float = 0.08, lr: float = 1e-3,
                   weight_decay: float = 0.0, batch_size: int = 4096,
                   max_epochs: int = 30, patience: int = 3,
                   val_frac: float = 0.1, holdout_mask: np.ndarray | None = None,
                   seed: int = 42, device: str = "cpu", verbose: bool = True):
    """BCE + Adam + early stopping. 홀드아웃은 **학습 시즌 안에서만** 뗀다.

    검증 시즌(예: 2024)을 조기종료 기준으로 쓰면 그 시즌이 하이퍼파라미터 선택에
    들어가 검증 점수가 낙관적으로 뜬다. 그래서 학습쪽에서 `val_frac`만큼 무작위로
    떼거나, 호출자가 `holdout_mask`로 직접 지정한다(예: 학습 마지막 시즌 전체를
    떼면 드리프트까지 반영한 조기종료가 된다).

    반환: (model, history) — history는 에폭별 홀드아웃 BCE/Brier.
    """
    torch = _torch()
    n = len(y)
    if holdout_mask is None:
        holdout_mask = _split_holdout(n, val_frac, seed)
    holdout_mask = np.asarray(holdout_mask, dtype=bool)
    if holdout_mask.shape != (n,):
        raise ValueError(f"holdout_mask 길이가 학습 행수와 다르다 "
                         f"({holdout_mask.shape} vs {n})")
    if holdout_mask.all() or not holdout_mask.any():
        raise ValueError("holdout_mask가 전부/전무다 — 조기종료 기준이 성립하지 않는다")

    dev = torch.device(device)
    tr_i = np.nonzero(~holdout_mask)[0]
    ho_i = np.nonzero(holdout_mask)[0]

    def _t(a, dtype):
        return torch.as_tensor(a, dtype=dtype, device=dev)

    Xt = _t(X_num[tr_i], torch.float32)
    Pt = _t(pid[tr_i], torch.long)
    Bt = _t(bid[tr_i], torch.long)
    Yt = _t(np.asarray(y, dtype="float32")[tr_i], torch.float32)
    Xh = _t(X_num[ho_i], torch.float32)
    Ph = _t(pid[ho_i], torch.long)
    Bh = _t(bid[ho_i], torch.long)
    Yh = _t(np.asarray(y, dtype="float32")[ho_i], torch.float32)

    torch.manual_seed(seed)
    model = build_model(X_num.shape[1], n_pitcher, n_batter, emb_dim=emb_dim,
                        hidden=hidden, dropout=dropout, seed=seed).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = torch.nn.BCEWithLogitsLoss()
    gen = torch.Generator(device=dev).manual_seed(seed)

    best = {"bce": float("inf"), "epoch": -1, "state": None}
    history = []
    n_tr = len(tr_i)
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_tr, generator=gen, device=dev)
        total = 0.0
        for s in range(0, n_tr, batch_size):
            b = perm[s:s + batch_size]
            p_in = apply_id_dropout(Pt[b], id_dropout, generator=gen)
            b_in = apply_id_dropout(Bt[b], id_dropout, generator=gen)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Xt[b], p_in, b_in), Yt[b])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(b)

        model.eval()
        with torch.no_grad():
            logit = model(Xh, Ph, Bh)
            ho_bce = float(lossf(logit, Yh))
            ho_brier = float(((torch.sigmoid(logit) - Yh) ** 2).mean())
        history.append({"epoch": epoch, "train_bce": total / n_tr,
                        "ho_bce": ho_bce, "ho_brier": ho_brier})
        if verbose:
            print(f"  epoch {epoch:2d}  train_bce={total / n_tr:.6f}  "
                  f"ho_bce={ho_bce:.6f}  ho_brier={ho_brier:.6f}", flush=True)
        if ho_bce < best["bce"] - 1e-6:
            best = {"bce": ho_bce, "epoch": epoch,
                    "state": {k: v.detach().clone()
                              for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= patience:
            if verbose:
                print(f"  early stop (best epoch {best['epoch']})", flush=True)
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, {"history": history, "best_epoch": best["epoch"],
                   "best_ho_bce": best["bce"], "n_train": int(n_tr),
                   "n_holdout": int(len(ho_i))}


def predict_embed_nn(model, X_num: np.ndarray, pid: np.ndarray, bid: np.ndarray,
                     *, batch_size: int = 16384, device: str = "cpu",
                     force_unk: bool = False) -> np.ndarray:
    """확률 예측. `force_unk=True`면 두 id를 전부 UNK로 강제한다.

    force_unk는 **임베딩 기여 분리**의 도구다. 같은 가중치로 id를 지운 예측과
    비교하면, 성능 차이가 임베딩이 실제로 옮긴 양이 된다 — 다른 모델과의 비교로는
    임베딩 때문인지 신경망이라서인지 구분되지 않는다.
    """
    torch = _torch()
    dev = torch.device(device)
    model = model.to(dev).eval()
    n = len(X_num)
    out = np.empty(n, dtype="float64")
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            xb = torch.as_tensor(X_num[s:e], dtype=torch.float32, device=dev)
            if force_unk:
                pb = torch.full((e - s,), UNK_INDEX, dtype=torch.long, device=dev)
                bb = pb
            else:
                pb = torch.as_tensor(pid[s:e], dtype=torch.long, device=dev)
                bb = torch.as_tensor(bid[s:e], dtype=torch.long, device=dev)
            out[s:e] = torch.sigmoid(model(xb, pb, bb)).cpu().numpy()
    return out


# --- 제출물 이식용 직렬화 ------------------------------------------------------
#
# `submit/script.py`는 의도적으로 `src`를 임포트하지 않는 독립 실행 스크립트다.
# 그래서 TabularPrep/IdVocab 객체도, nn.Module도 그대로 실어 보낼 수 없다
# (pickle에 `src.nn_embed` 모듈 경로가 박혀 평가 서버에서 죽는다 — CatBoost 래퍼와
# 같은 문제다). 아래 두 함수가 그 결합을 끊는다: 전처리 상수는 JSON dict로,
# 가중치는 **numpy 배열 dict**로 나가고 script.py가 인라인으로 재구성한다.
#
# 가중치를 torch 직렬화가 아니라 numpy로 내보내는 이유는 **버전 방향** 때문이다.
# 로컬 torch가 평가 서버(2.7.1)보다 새 버전이라 torch 파일의 전방 호환(새 버전이
# 쓴 파일을 옛 버전이 읽는 것)은 보장되지 않는다. `.npz`는 torch와 무관한 포맷이고
# float32 값이 그대로 보존되므로 비트 동일성도 유지된다.

def prep_meta(prep: TabularPrep) -> dict:
    """`TabularPrep`의 fit 상수 -> meta.json에 실을 수 있는 dict.

    transform이 쓰는 것 전부가 여기 들어간다(컬럼 목록/순서, 중앙값, 평균, 표준편차,
    결측 플래그 대상, 범주 레벨). 재현 쪽에서 dtype 추론을 다시 하지 않도록
    **컬럼 분류 결과까지** 상수로 굳히는 것이 핵심이다 — 평가 데이터에서 dtype이
    조금 달라져도 같은 열이 같은 자리에 떨어져야 한다.
    """
    return {
        "num_cols": list(prep.num_cols_),
        "cat_cols": list(prep.cat_cols_),
        "na_cols": list(prep.na_cols_),
        "levels": {c: list(prep.levels_[c]) for c in prep.cat_cols_},
        "median": {c: float(prep.median_[c]) for c in prep.num_cols_},
        "mean": {c: float(prep.mean_[c]) for c in prep.num_cols_},
        "std": {c: float(prep.std_[c]) for c in prep.num_cols_},
        "feature_names": list(prep.feature_names_),
    }


def state_dict_arrays(model) -> dict:
    """nn.Module -> {파라미터 이름: numpy 배열}. `np.savez`에 그대로 넣는다.

    모듈이 아니라 **state_dict만** 내보낸다. 클래스 정의는 script.py가 따로 갖고
    있고 `load_state_dict(strict=True)`가 이름·모양이 맞는지 검사하므로, 아키텍처가
    어긋나면 조용히 틀린 예측이 나오는 대신 로드에서 즉시 터진다.
    """
    return {k: v.detach().cpu().numpy()
            for k, v in model.state_dict().items()}

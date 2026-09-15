# 인수인계 — 현재 위치와 다음에 할 일

작성 2026-08-21. 이 문서 하나만 읽으면 이어서 작업할 수 있게 쓴다.

> **STATUS: CHAMPION FROZEN**
>
> - Champion: `artifacts/sub_tree_reblend_w0p300.zip`
> - 구성: CatBoost 30% + LightGBM team 70%
> - LB: `1058.074429882`
> - SHA-256: `8ce93a197efea18cf89b7b988c59a8b369d683983152b24b74976a96885da7cb`
> - Latest submission: 22차 two-strike, LB `1056.264418601`, **REJECTED**
> - Current action: **FINAL SUBMISSION OPERATIONS**
> - Active model search: **STOPPED**
> - 최종 감사: `docs/experiments/champion-freeze.md`
> - 제출 상태: `docs/FINAL_SUBMISSION_STATUS.md`

외부 신규 데이터 없이 진행한 구조 탐색은 최종 headroom 감사 결과 deployable `+10 BSS`
근거가 없어 종료했다. 동결 champion 자체는 계속 불변이다. 첫 축인 Cat/team conditional mixture-of-experts는 학습 전 반복성
감사에서 기각했다. 2020~2024 expert 우위 방향이 계속 반전했고, 17개 현재-row 조건의
전 시즌 stable-positive coverage가 모두 0%였다. selector/soft gate/stacking으로
확대하지 않는다. 상세는 `docs/experiments/conditional-moe-screen.md`다.

외부 데이터 재개 감사도 완료했다. 공식 DACON 규칙 2-3과 2026-08-21 운영진 답변은
Phase 2 공식 제공 데이터 외 외부 데이터를 명시적으로 금지한다. KBO 선수/roster/
일정 자료는 존재하지만 official 익명 player/team ID와 direct key가 없고, stadium은
date/game ID가 없어 exact join도 불가능하다. 따라서 **NO ACTIONABLE EXTERNAL DATA
AXIS / DO NOT TRAIN**이다. 수집·학습·ZIP 생성은 하지 않았으며 상세는
`docs/experiments/external-data-reopen-audit.md`다.

공식 TrackMan target-free frozen pitcher representation도 학습 전 감사했다. Repertoire
separation과 mix entropy는 기존 mean/std profile로 잘 복원되지 않아 **PARTIALLY
NOVEL**이지만, strict high-confidence OOF row coverage가 2022/2023/2024
`26.35%/28.61%/30.00%`이고 cold/low-history는 0%다. 신규 factor와 champion residual
상관은 2022·2024 모두 `|r|<=.00667`, loss 상관도 `|r|<=.01356`이며 방향이 반복되지
않았다. 결론은 **TRACKMAN REPRESENTATION IS NOVEL BUT LOW EXPECTED VALUE / DO NOT
TRAIN**이다. PCA/AE/mixture 모델을 실제 학습하거나 ZIP을 만들지 않았다. 상세는
`docs/experiments/trackman-selfsupervised-reopen-audit.md`다.

Regime-robust learning 재개 감사도 완료했다. Season GroupDRO는 선행 저장소에서 이미
동일 objective, temporal OOF, eta grid, 2시드, champion blend와 subset safety까지
검증되어 KILL됐고, leave-one-regime-out은 현재 worst-fold validation policy와 중복된다.
Stable-feature restriction도 실제 shape-stability 분류와 stable-only 학습까지 실패했다.
GroupDRO 0.5% champion blend는 2022/2023/2024 `+0.43/-1.47/+0.99 BSS`, stable
additive는 `-0.76/+2.90/+1.07 BSS`로 2024 `+10` 근거가 없다. 2022·2024 공통
취약군도 count의 약한 방향 외에는 반복되지 않았다. 결론은 **REGIME-ROBUST OBJECTIVE
REDUNDANT — DO NOT TRAIN**이며 모델/ZIP은 만들지 않았다. 상세는
`docs/experiments/regime-robust-reopen-audit.md`다.

Architecture/representation 재개 감사도 완료했다. FT-Transformer, TabM,
player-embedding MLP, exact train-only kNN, prototype/codebook, rank 2/4/8 pair residual
ALS는 모두 실제 temporal OOF까지 검증돼 있다. Exact ID-aware field attention과
DeepFM은 구현 자체는 없어 **PARTIALLY NOVEL**이지만, history `<100`, unseen matchup,
희소 player x count는 2022·2024 모두 champion 전체보다 쉬웠다. Player x count
interaction의 rank-4 singular energy도 pitcher `44.64%`, batter `43.42%`, 독립 시기
효과 상관은 `.0605/.0695`뿐이다. FT residual corr도 2022·2024 `.9989` 수준이고 전
fold standalone 악화다. 결론은 **ARCHITECTURE AXIS IS NOVEL BUT LOW EXPECTED VALUE /
DO NOT TRAIN**, 선택 후보 `NONE`이다. 상세는
`docs/experiments/architecture-reopen-audit.md`다.

Champion recoverable-headroom 통합 감사도 완료했다. 실제 row-level OOF가 남은 16개
prediction을 current 21차 champion과 정렬했다. Per-row oracle은 quality-filter만 써도
2022/2024 `+3824.97/+3831.48 BSS`지만 정답을 보고 고른 **UNDEPLOYABLE** 상한이다.
2022 winner를 2024로 넘긴 cross-fit은 `+4.34`, 반대는 `+5.75`; 사전 legal axis
대부분은 다음 season에서 음수였다. 두 season을 모두 본 retrospective single-axis
상한도 2024 `+3.87`이며 모든 positive route가 실제 LB `-1.8100`으로 실패한
two-strike 하나에 의존한다. Stable route actual changed coverage는 최대 후보에서 8.88%,
low-disagreement/high-entropy irreducible-looking proxy는 약 19.4~19.6% rows다.
`+10/+20/+40` 및 1100 gap을 지지할 deployable evidence가 없다. 최종 판정은
**NO MATERIAL RECOVERABLE HEADROOM — STOP MODEL SEARCH**. 21차 champion을 유지하고
새 공식 정보/버그/+10 구조 증거 전까지 active exploration을 종료한다. 상세는
`docs/experiments/champion-headroom-audit.md`, 산출물은 `artifacts/headroom_audit/`다.

---

## 1. 현재 위치

| | 점수 | 상태 |
| --- | ---: | --- |
| 18차 `artifacts/submit_season_state.zip` | **1051.5537225994** | 이전 최고 |
| 19차 `artifacts/submit_full_state.zip` | **1047.5446166079** | 기각 (18차 대비 −4.0091) |
| 20차 `artifacts/submit_stage_ours_w0p500.zip` | **1048.0613692365** | 기각 (18차 대비 −3.4924) |
| 21차 `artifacts/sub_tree_reblend_w0p300.zip` | **1058.074429882** | 새 최고 (18차 대비 +6.5207) |
| 22차 `artifacts/sub_two_strike_lgbcat_w0p250.zip` | **1056.264418601** | 기각 (21차 대비 −1.8100) |

최종 제한 검증을 통과해 제출했지만 **실측 기각된 후보**가 하나 있다:

- `artifacts/sub_two_strike_lgbcat_w0p250.zip`
- two-strike(0-2/1-2/2-2) 안에서만 `75% champion + 25% × mean(LGB,CAT expert)`
- rolling ΔBSS `+5.75/+24.02/+4.34`; 2024 paired 2SE `4.08` 통과
- seed 42/43/44 모두 2022·2024 양수, outside slice 비트 동일
- CRC/격리 실행/3,000행 6종 독립성 통과, SHA-256
  `48add68ddd754f496256483aa41057de9612592a601f2ea18e5f445a1bfeb297`
- 실제 LB `1056.264418601`, 21차 대비 `-1.810011281`
- 상세 `docs/experiments/two-strike-final-expert.md`

21차 champion을 유지한다. Two-strike는 LB에서 실패했으므로 설정 변경, count 분리,
interaction 또는 weight 추가 탐색 없이 **축을 종료**한다. 제출했던 후보 ZIP은 현재
로컬 artifacts에 없고 builder로 재현 가능하지만 재생성·재제출 금지다.

22차 실패 후 submission gate를 크게 강화한 high-margin inventory도 완료했다.
Transformer/TabM, retrieval·prototype, reliability/shrinkage, masked pretraining,
oblique/additive/latent representation, GroupDRO/direct-Brier 등 선행 구조 실험까지
대조했으나, 기존 축과 중복되지 않으면서 2024 `+10 BSS`를 기대할 근거가 있는 후보는
0개였다. 따라서 Stage 1 학습도, 새 ZIP 생성도 하지 않았다. 결론은
**NO HIGH-MARGIN SUBMISSION CANDIDATE**이며 상세는
`docs/experiments/high-margin-screen.md`다. 앞으로 `+1~5 BSS` 후보는 유의하더라도
제출하지 않고, 2024 `>=+10`, `>=1.5×paired 2SE`, 2022 양수, 3시드 방향 안정성을
모두 요구한다.

### 종료 축 요약 — 재시도 방지

| 실험 | 판정 | 핵심 이유 | 재시도 |
| --- | --- | --- | --- |
| TrackMan historical profile | REJECTED | 잔차 상관과 temporal gain 부족 | 금지 |
| TrackMan LUPI auxiliary / distillation | REJECTED | 2023 의존, 2022·2024 악화 | 금지 |
| TrackMan self-supervised repertoire | CLOSED | partially novel이나 strict coverage 26~30%, residual link 없음 | 금지 |
| pitcher usage / role state | REJECTED | 2023·2024 악화, residual corr `.999764` | 금지 |
| hierarchical empirical-Bayes | REJECTED | 2022·2024 모두 악화 | 금지 |
| hard-example / uncertainty weighting | REJECTED | calibration 붕괴 또는 모든 fold 악화 | 금지 |
| signed/two-strike residual specialist | CLOSED | 22차 실제 LB `-1.810011281` | 금지 |
| forward residual LightGBM | REJECTED | residual corr `.0052`, 효과 크기 부족 | 금지 |
| Histogram XGBoost | REJECTED | 2024 단독·5% blend 모두 악화 | 금지 |
| 기존 NN 재혼합 | REJECTED | 모든 시즌에서 강한 tree보다 약함 | 금지 |
| tree/stage calibration | REJECTED | 무효 또는 실제 LB 악화 | 금지 |
| Cat/team fine weight tuning | CLOSED | 30~40% 평탄, 2024 고비중 악화 | 금지 |
| Cat/team conditional MoE | CLOSED | expert 상대우위 비반복, 전 시즌 stable coverage 0% | 금지 |
| `prev5-current success` | REJECTED | 2024 악화 | 금지 |
| season-state × count/hand | REJECTED | fold·seed 방향 불안정 | 금지 |
| two-strike 0-2/1-2/2-2 expert 전 파생 | CLOSED | 실제 LB 실패 | 재탐색·재튜닝·재제출 금지 |

Two-strike의 LGB/Cat expert, ensemble, interaction, count 세분화, residual specialist,
weight tuning을 포함해 이름이나 구성만 바꾼 파생도 다시 시도하지 않는다.

### 탐색 재개 조건

새 공식 데이터, 규칙/운영진 답변 변경, 데이터 해석 오류, champion 구현 버그,
명확히 누락된 구조적 정보축, 또는 2024 `+10 BSS` 이상을 기대할 근거가 있는 완전히
새로운 방법이 있을 때만 대규모 탐색을 검토한다. 새 알고리즘 이름, 작은 파라미터나
blend weight 변화, 작은 slice는 재개 조건이 아니다.

선두 1197 / 2위 1196 / 3위 1186 / 5위 1170대. 1100 까지 약 48점.

19차는 시즌 상태 피처를 5멤버 전부(hgb·cat·nn·team·team_nn)로 넓힌 것이다.
18차는 cat·team 두 멤버(블렌드의 약 70%)에만 들어가 있었다. 커버리지 비례로 +8을
예상했지만 실제로는 **−4.0091**이었다. 피처가 두 멤버에서 유효하다는 사실은 다른
멤버에도 같은 피처가 유효하다는 뜻이 아니다. **19·20차 당시의 기준선은 18차였고,
21차 이후 최종 기준선은 동결된 21차다.**

빌드·검증은 끝나 있다:
- 실행 245,789행 16초
- 행 독립성 4종 통과 (`tests/test_row_independence.py`)
- hgb pickle 심볼 전수 검증 (`numpy.core...` 만 참조, `numpy._core` 없음)
- 수준 확인 (`scripts/make_proxy_2025.py`): 챔피언 0.472750 / 18차 0.473649 /
  19차 0.474985 → `final_logit_shift = 0.0` (보정 없음)

---

## 2. 작업 환경

```bash
./.venv/bin/python        # 일반 작업 (numpy 2.x)
./.venv312/bin/python     # hgb pickle 을 만들 때만 (Python 3.12 + numpy 1.26.4)
```

`data/` 는 Git 에서 제외돼 있다. `train.csv` / `trackman_history.csv` /
`test.csv` / `sample_submission.csv` 를 대회 페이지에서 받아 넣는다.

`torch` 를 건드리는 모든 실행은 `OMP_NUM_THREADS=1` 이 필요하다. torch·lightgbm·
catboost 가 한 프로세스에 공존하면 macOS 에서 OpenMP 교착이 난다 (CPU 0%, 무한 대기).
학습(스레드 6)과 캘리브레이션(스레드 1)은 프로세스를 나눈다.

```bash
./.venv/bin/python -m src.backtest                    # 베이스라인 3폴드
OMP_NUM_THREADS=1 ./.venv/bin/python -m pytest tests/ # 행 독립성
```

---

## 3. 판정 규칙 — 이걸 어기면 예전 실패를 반복한다

14차까지 14번 중 5번이 "로컬 개선 → 리더보드 하락"이었다. 그래서 판정 도구를 먼저
고쳤다. `src/backtest.py` 를 쓴다.

1. **롤링 3폴드** (~21→22, ~22→23, ~23→24). 빌더가 `train_seasons` 를 인자로
   받으므로 미래 참조가 구조적으로 불가능하다.
2. **세 폴드 전부 개선일 때만 채택.** 평균은 믿지 않는다 — 2023 폴드는 F 구간
   레짐 붕괴(.709→.473)로 극단값을 내 평균을 통째로 끌고 간다.
3. **2σ 노이즈 바닥.** `compare(res_a, res_b, preds_a=..., preds_b=...)` 가
   예측 차이만으로 계산한다. 정답 라벨이 필요 없어 제출 전에 유의성을 본다.
4. **재학습이 끼면 반드시 양쪽 다중 시드.** 같은 피처로 시드만 바꿔도 폴드별
   ±27 이 흔들린다. 단일 시드 비교로 나온 숫자는 전부 무효다.
5. **전이 배율은 상수가 아니다.** 새 정보를 더하는 변경만 전이된다.

   | 변경 성격 | 백테스트 | 실측 LB | 비율 |
   | --- | ---: | ---: | ---: |
   | 보정 3종 — 새 정보 | +24.6 | **+15.48** | 0.63 |
   | 시즌 상태 — 피처 추가 | +44.2 | **+19.6** | 0.44 |
   | 시드 앙상블 — 분산 감소만 | +3.5 | −0.16 | ~0 |

6. **수준에 영향을 주는 변경은 예외 없이 `make_proxy_2025.py` 로 먼저 확인한다.**
   17차에서 이중 보정으로 899.15 (−132.8) 를 받았다. 원인은 검증 도구였다 —
   그때 리허설은 2024 행의 `season` 만 2025 로 바꾼 합성이라 시즌 상태 피처가
   전부 결측이었다. 일반형: **시즌 상태 피처가 있으면 모델이 기저율 드리프트를
   스스로 흡수한다. 그 위에 드리프트 보정을 또 얹으면 과보정이다.**

---

## 4. 반드시 지켜야 하는 대회 규칙

> 평가 데이터의 각 행은 **독립적으로** 예측되어야 한다. 행 A 의 예측값은
> ① 행 A 의 입력 변수 ② 행 A 의 입력 변수만으로 만든 파생변수
> ③ 공식 학습 데이터 ④ 공식 학습 데이터만으로 만든 통계·모델·파생변수
> 로만 생성되어야 한다.

금지: test.csv 내 다른 행을 이용한 누적/rolling/lag, test 전체의 평균·분포·빈도·
순위를 이용한 보정, 같은 선수·팀·월·경기 단위로 평가 데이터를 집계하는 방식.

**자체 검증:** `tests/test_row_independence.py` 가 실제 배포 ZIP 을 풀어 셔플 /
부분집합(50%, 10%) / 단일행 / 다른 행 변조 아래 예측이 **비트 동일**한지 확인한다.
제출 전에 반드시 통과시킨다.

**허용되는 것 (운영진 8/12 공식 답변):** 리더보드 점수를 근거로 한 후보 선택,
그리고 하이퍼파라미터·앙상블 비중의 보간. 위반 판정 기준은 **제출 코드가 추론
시점에 다른 test 행이나 test 분포를 참조하는가** 뿐이다.

사전학습 가중치는 MIT/Apache 2.0 등 비상업 이상 공개 라이선스만. 원격 API 모델
(OpenAI/Gemini) 금지. 공식 Phase-2 데이터 외 외부 데이터 금지.

---

## 5. 후보 A — Trackman 카운트별 구종 구성 → **기각**

이 절은 결론이 이미 났다. 다시 하지 않기 위해 근거를 남긴다.

**가설.** 공식 피처의 `asof_pitcher_{fastball,breaking,offspeed}_rate` 는 투수의
**전체** 구종 구성이다. 실제 배합은 카운트에 따라 크게 달라지고(3-0 직구, 0-2
유인구) 구종마다 제구 난이도가 다르다. `trackman_history.csv` 에는
`balls_before`/`strikes_before` 가 그대로 있으므로 `(투수, 카운트) → 구종 분포`
를 만들 수 있고, 이건 `asof_*` 에 **없는** 정보다.

이전에 기각한 Trackman 물리 피처(릴리스·무브먼트)와는 다른 축이다. 그때는 투수-시즌
**집계**만 썼고 카운트를 무시했다.

**측정.** `scripts/probe_trackman_count_mix.py`. 모델을 새로 학습하지 않고, 현재
최고 구성의 2024 폴드 예측 잔차 `y − p` 와 새 피처의 상관을 본다. 새 피처는
전체 구성 대비 **편차** 로 만든다 (전체 구성은 모델이 이미 아니까):

    dev_g = mix_g(투수, 카운트) − mix_g(투수, 전체)

매핑된 투수 541명, (투수, 카운트) 칸 6,330개, 커버리지 0.769.

**결과.**

```
        피처       r(잔차)     r(R행)
 cm_dev_fastball   -0.0032    -0.0035
 cm_dev_breaking   -0.0056    -0.0056
 cm_dev_offspeed   +0.0125    +0.0129     <- 가장 큼
```

기각선 |r| ≥ 0.02 를 아무것도 넘지 못했다. 그런데 `cm_dev_offspeed` 만 5분위
잔차 평균이 단조에 가까웠다 (−0.0180 → −0.0010, 폭 0.0171). 축소 상수 K 에도
둔감했다 (K=5~200 에서 r=0.0111~0.0127) — 축소 인공물이 아니다.

그래서 교란을 확인했다. 오프스피드 사용은 2스트라이크에서 늘고, 성공률도 카운트마다
다르며, **카운트는 모델이 이미 안다.** 카운트 칸 **안에서** 다시 재면:

```
0-0 +0.0017   0-1 +0.0139   0-2 +0.0150   1-0 +0.0160
1-1 +0.0042   1-2 +0.0263   2-0 +0.0054   2-1 +0.0133
2-2 -0.0013   3-0 -0.0189   3-1 -0.0083   3-2 +0.0026
가중평균 +0.0082   (칸 밖 전체 +0.0125)
```

신호의 1/3 이 카운트 교란이었고, 남은 +0.0082 는 칸마다 부호가 뒤집힌다. 이미
기각된 물리 피처(≤0.016)보다도 낮다. **여기서 멈춘다.**

**그래도 모델 수준까지 가보겠다면** — 백테스트 3폴드 × 양쪽 3시드가 필요하고
(위 판정 규칙 4번), 대략 반나절이다. `src/trackman_features.py` 의
`make_builder` 를 본떠 빌더를 만들고 `compare()` 로 판정한다. 세 폴드 전부
2σ 를 넘지 못하면 제출하지 않는다. 다만 잔차 상관이 이 수준일 때 백테스트가
0 이 나온 전례가 이미 두 번 있다는 걸 알고 시작할 것.

**재사용 가능한 것:** `build_pitcher_map` 이 복원한 `pitcher_id ↔
pitcher_trackman_id` 연결 자체는 검증되었다 (팀 일치율 0.997, 구종 비율 상관
0.69 vs 무작위 −0.03). 피처만 기각이지 매핑은 유효하다.

---

## 6. 다음에 할 일 — 우선순위 순

### 6.1 19차 제출 — 완료, 기각

`artifacts/submit_full_state.zip` 은 **1047.5446166079**, 18차 대비 **−4.0091059915**.
5멤버 전체 확장은 기각하며 아래 LB 튜닝은 18차를 기준선으로 한다.

### 6.2 LB 피드백 튜닝 (가장 확실하게 남은 것)

**기준 ZIP: `artifacts/submit_season_state.zip` (18차, 1051.5537225994).**

백테스트 전이 배율이 0.44 밖에 안 되고, 상위권은 50~73회를 제출했다. 즉
**리더보드가 더 정확한 심판**이다. 로컬에서 "평평하다 / LOFO 로 기각"으로 버린
손잡이들을 한 번에 하나씩 실제로 올려본다. 운영진 답변으로 합법이 확인된 방식이다.

한 번에 **하나만** 바꾸고, 바꾸기 전에 `compare()` 로 2σ 를 재서 "제출할 가치가
있는 크기인지" 부터 확인한다. 2σ 안이면 점수 차이는 노이즈와 구분되지 않는다.

| 손잡이 | 현재값 | 시도 범위 | 로컬 소견 |
| --- | --- | --- | --- |
| ID 비중 | 20% | 20 → 26% | 707.0 vs 704.8, LOFO −1.5 |
| 스테이지 비중 (ours) | 0.4 | 0.5 → 0.7 | 평균 +2.8 |
| 보정 k (β) | 2000 | 1200 | +1.3 |
| 콜드스타트 비중 | 0.4181944104 | 0.25 → 0.35 | LOFO −0.5 |
| season_form α | 현재 off | 부분 재적용 | 미측정 |

14차에서 썼던 방법을 그대로 쓴다: 같은 손잡이의 실측 LB 세 점에 Brier 이차식을
맞춰 최적값을 복원한다. 세 점이 모이기 전에는 외삽하지 않는다.

첫 후보 `artifacts/submit_stage_ours_w0p500.zip` 을 만들었다. 18차에서 메타의
스테이지 비중만 `ours/team = 0.4/0.6 → 0.5/0.5` 로 바꿨다. 프록시 2025에서 평균은
`0.473649 → 0.473153`, 최대 행 변화 `0.006763`, 예측 차이 2σ는 `1.99 BSS`다.
상세는 `docs/experiments/stage-weight-lb.md`.

실제 LB는 **1048.0613692365**로 18차보다 **−3.4923533629** 하락했다. 따라서
`ours=0.5`는 기각하고 `0.6/0.7`도 제출하지 않는다. 이후 기준선은 계속 18차다.

### 6.2.1 다음 제출 후보 — tree-only 재배합

실제 18차 5멤버를 2021~2024 rolling OOF로 모두 재학습했다. NN 두 축이 네 시즌
모두 트리 축보다 약해, 고정 **Cat 30% + LightGBM team 70%** 후보를 만들었다.

- ZIP: `artifacts/sub_tree_reblend_w0p300.zip`
- 2025 구조 프록시: `1535.71 → 1567.60` (**+31.89 BSS**)
- rolling 방향: Cat25/team75 기준 `+94.74 / -0.76 / +165.88 / +1.50`
- 기본 ZIP 대비 변경 파일: `model/meta.json` 하나
- 행 독립성 3,000행 4종 비트 동일, 전체 테스트 `7 passed`
- SHA-256: `8ce93a197efea18cf89b7b988c59a8b369d683983152b24b74976a96885da7cb`

실제 LB는 **1058.074429882**로 18차보다 **+6.5207072826** 개선되어 새 챔피언으로
승격한다. 프록시 +31.89의 LB 전이율은 약 20.4%였다. Cat 비중 30→40%는 프록시와
rolling 모두 차이가 작아 추가 제출 가치가 없고, 50% 이상은 최근 2024 폴드가
악화한다. 상세는 `docs/experiments/high-gain-screen.md`, 재현은
`scripts/build_tree_reblend.py`.

**현재 환경 주의:** 행 독립성 비트 검사는 18차 원본과 후보가 둘 다 실패한다.
Torch 2.8의 `nn`만 셔플 시 7/3,000행에서 최종 예측 최대 `4.82e-9`(18차),
`6.02e-9`(후보) 차이가 난다. 다른 네 멤버는 비트 동일하고 후보는 `meta.json` 외
파일 해시가 원본과 같다. 기존 인수인계 환경에서는 4종 통과 기록이 있으나, 현재
환경에서는 엄격한 비트 동일을 재현하지 못했다.

### 6.3 싼 파생 두 개 — **둘 다 기각**

- **최근 폼 vs 시즌 베이스라인.** `prev5 − cur_p_succ` 한 컬럼. 3시드 R 폴드
  `+14.19 / +2.76 / -1.84`, 평균 `+5.04`. 2023은 2σ 안이고 2024는 악화라 기각.
- **상태 피처 × 카운트/좌우 명시적 상호작용.** 투수 상태 비율 5개를 12개 카운트와
  4개 좌우 셀에 게이팅한 80컬럼. 3시드 R 폴드 `+4.20 / -0.47 / +29.09`, 평균
  `+10.94`. 2022는 2σ 안이고 2023은 악화라 기각.

둘 다 기준선과 후보를 seed 42·43·44로 재학습했다. 재현은
`scripts/validate_state_derivatives.py`, 상세 기록은
`docs/experiments/state-derivatives.md` 를 본다.

---

## 7. 하지 말 것 — 기각 기록

### 2022·2024 residual slice — two-strike 발견, expert 제출은 기각 (2026-08-22)

사전 정의 slice scan에서 0-2/1-2/2-2(`two_strike`, share 약 24%) 하나만 stable했다.
Excess Brier/500-bootstrap CI는 2022 `.001530/[.000900,.002141]`, 2024
`.000974/[.000606,.001336]`. Slice-only LightGBM 3시드 25% gated blend는 전체
ΔBSS `+6.50/+22.75/+3.78`, slice ΔBSS `+26.98/+95.70/+15.67`; outside는 0이고
R/F도 양수였다. 그러나 2024 paired 2SE가 `4.31`로 gain `3.78`보다 커 제출 gate
FAIL. **NO SUBMISSION CANDIDATE**, ZIP 없음. 상세는
`docs/experiments/residual-slice-mining.md`.

### 2023 regime shift 진단 — shift는 있으나 모델 action 없음 (2026-08-22)

2023-only 개선의 주원인은 `game_type=F` target 급락이다. F target/prediction은
2022 `.7087/.7032` → 2023 `.4729/.6688` → 2024 `.4593/.4655`; 2023 OOF만 과거
high-F prior 때문에 calibration error `-.1959`를 냈다. Hard/EB/LUPI auxiliary가
2023 F를 개선한 대신 같은 F 행을 2022·2024에서 모두 악화했다. 2024 champion은
이미 새 F 레짐에 적응했으므로 2025 mixture로 전이할 근거가 없다. 판정은
**SHIFT EXISTS BUT NOT ACTIONABLE / ROBUST VALIDATION POLICY ONLY**. 상세는
`docs/experiments/regime-shift-diagnostic.md`, 산출물은 `artifacts/regime_shift/`.

### Hard-example / uncertainty-aware objective — 기각 (2026-08-22)

Cutoff-safe Cat30/team70 inner OOF로 LightGBM team 학습행을 재가중했다. Hard
q50/q75/q90=`1/1.5/2/3`은 ECE `.09~.12`, prediction corr 음수로 붕괴했다.
Uncertainty λ=.5/1.0은 residual corr `.9999`이고 단독 Brier가 세 폴드 모두 악화했다.
두 seed 모두 모든 후보가 2022·2024를 악화해 seed 44와 signed specialist를 중단했다.
Uncertainty .5의 5% blend BSS는 `-0.20/-4.24/+0.35`; 제출 ZIP 없음. 상세는
`docs/experiments/hard-example-objective.md`, 재현은
`scripts/validate_hard_example_objective.py`.

### Hierarchical Empirical-Bayes prior-only — 기각 (2026-08-22)

공식 train의 cutoff 이전 정답만으로 전역→투수/타자→좌우·카운트→매치업 계층을
부모 posterior에 shrink했다. 현재 21차 챔피언 OOF 대비 대표 `k=500` gain은
2022/2023/2024 `-0.004647 / +0.001228 / -0.003666`; 2.5~15% 블렌드도 2022와
2024가 모두 악화했다. 저표본 투수·타자 구간은 오히려 가장 나빴다. 중단 규칙에
따라 logistic은 실행하지 않았고 제출 ZIP도 만들지 않았다. 셔플/역순/부분집합/삭제/
무관 중복 행 독립성 최대 차이는 0.0. 재현은 `scripts/validate_hierarchical_eb.py`,
상세는 `docs/experiments/hierarchical-eb-prior.md`.

재시도 전에 이 숫자를 먼저 볼 것.

| 기각안 | 근거 |
| --- | --- |
| Trackman 물리 피처 (릴리스·무브먼트) | 잔차 상관 ≤0.016, 백테스트 부호 엇갈림 |
| Trackman 카운트별 구종 구성 | 칸 내부 상관 +0.0082 (5절) |
| 시즌 상태 5멤버 전체 확장 | LB 1047.5446, 18차 대비 −4.0091 |
| `prev5 − cur_p_succ` | 3시드 R 폴드 +14.19 / +2.76 / −1.84, 평균 +5.04 |
| 상태 × 카운트/좌우 80개 | 3시드 R 폴드 +4.20 / −0.47 / +29.09, 평균 +10.94 |
| 당해 시즌 상태 − 커리어 상태 10개 | 3시드 R 폴드 +2.52 / +15.50 / +8.30, 평균 +8.77; 2개 폴드 2σ 안 |
| 성공률 시즌 경계 off-by-one 수정 | 3시드 R 폴드 +17.92 / +2.18 / −6.64, 평균 +4.49 |
| CatBoost 선수 ID CTR | 2024 R: 투수+타자 −61.11 / 투수만 −46.03 |
| 신규 잔차 lookup | 타자×베이스 평균 +2.37, 타자×손·유불리 +2.12, 맞대결 +1.80; 모두 2σ 미만 |
| F 최근 레짐 상수 25% | 실제 18차 2024 프록시 전체 −27.98, F −237.79; 제출 금지 |
| Histogram XGBoost | 2024 단독 619.39, 21차 기준에 5% 혼합도 −1.90 |
| forward residual LightGBM | 2024 잔차 상관 0.0052, 10% +0.55 / 20% −27.14 |
| TrackMan LUPI auxiliary student | seed 42·43 모두 2022/2024 악화; 21차 5% blend +0.88/+25.13/−2.50, 2023 의존 |
| TrackMan teacher→student | 21차 5% blend +0.25/−10.50/−1.09; privileged signal이 student로 전달되지 않음 |
| 공식 feature inventory + pitcher role state | main predictor 47/47 이미 사용; role 5% blend +0.96/−13.59/−0.34, residual corr 0.999764 |
| 다중 창 피처 `window_features` | 3시드 세 폴드 전부 악화, 평균 −9.4, z=−1.98 |
| 구종 엔트로피 | −9.4 |
| asof 오프셋 | −36.6 |
| F 학습 제외 | −25.9 |
| 시즌 감쇠 학습가중 | −14.1 |
| R-only 캘리브레이션 | ~0 |
| season-form 재튜닝 | 2024 폴드 +2.8, 유의하지 않음 |
| L2 목적함수 / CatBoost 멤버 | 개선 없음 |
| 시드 앙상블 | 백테스트 +3.5 → LB −0.16 (분산 감소는 전이 안 됨) |
| 블렌드 비중 재최적화 | LOFO +0.6 |

살아남은 것은 전부 **"이 투수가 상황에 따라 어떻게 달라지는가"** 축이었다.
구장·주자·이닝·홈원정·포수·레버리지 등 상황 자체는 전부 0 이었다.

정보원별 천장 (2024, R 구간): `pitcher_id` 683.5 → 현재 모델 737 →
`pitcher×좌우` **783.9**. 상황 변수는 7~23 에 불과하다. 남은 오라클
(+130~160) 은 `pitcher×월` 같은 **관측 불가능한 동시 시즌 정보**다.

---

## 8. 빌드 파이프라인

```bash
# 챔피언 ZIP 을 사양서로 재현
./.venv/bin/python -c "from src.champion import Champion; ..."

# 멤버별 재학습
./.venv312/bin/python scripts/train_hgb_state.py <base.zip> /tmp/hgb_state.pkl
OMP_NUM_THREADS=1 ./.venv/bin/python scripts/train_nn_state.py <base.zip> /tmp/nn_state

# 조립
./.venv/bin/python scripts/assemble_full_state.py \
    --hgb /tmp/hgb_state.pkl --nn-dir /tmp/nn_state

# 수준 확인 + 행 독립성
./.venv/bin/python scripts/make_proxy_2025.py
OMP_NUM_THREADS=1 ./.venv/bin/python -m pytest tests/test_row_independence.py
```

**hgb 는 반드시 `.venv312` 에서 만든다.** 이유가 두 가지다.
1. fitted 모델이 품는 `_feature_subsample_rng` (numpy Generator) — numpy 2.x 는
   BitGenerator 를 클래스로 직렬화하는데 평가 서버의 1.26.4 는 문자열을 기대한다.
   실제 제출 실패로 확인했다 (`PCG64 is not a known BitGenerator module`).
2. 더 결정적으로, numpy 2.x 는 `numpy._core.multiarray.scalar` 를 참조하는데
   1.26.4 에는 `numpy._core` 자체가 없다. RNG 를 제거해도 이건 남는다.

`train_hgb_state.py` 가 저장 후 pickle 이 요구하는 심볼을 전수 열거해 확인한다.
로컬 리허설로는 절대 못 잡는다 — 같은 numpy 로 pickle 하고 unpickle 하니까.

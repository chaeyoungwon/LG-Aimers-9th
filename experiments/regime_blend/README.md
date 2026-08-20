# Regime-aware blend experiment

이 디렉터리는 production 제출물과 분리된 R16 결합 연구의 실행 계약이다. 설계와
채택 규칙은
`docs/experiments/2026-08-19-regime-aware-blend-design.md`에 있다.

## 입력

`manifest.example.json`을 `artifacts/regime_blend_oof/manifest.json`으로 복사한 뒤,
각 fold의 NPZ를 같은 디렉터리에 둔다. NPZ는 pickle object를 포함하지 않아야 한다.

예측 배열:

- `p_inner_<member>`, `p_val_<member>`: 원시 5개 member와 보정 완료 두 stage
- `p_inner_champion_base`, `p_val_champion_base`: 기존 0.4/0.6 stage blend
- `form_offset_inner`, `form_offset_val`: 고정된 기존 투수+타자 form의 additive offset
- `p_val_champion_form`: 선택. 없으면 base prediction에 form offset을 더해 clip한다.

label 배열은 `y_inner`, `y_val`이다. 행 feature는
`x_inner_<feature>`, `x_val_<feature>` 형식이다. `pitcher_seen`은 각 fold의 train
cutoff 이하 정규시즌 투수 ID 집합만으로 미리 만들어야 한다. validation/test의 ID
빈도나 집합에서 만들면 안 된다.

각 fold의 fitting 입력은 inner prediction/label뿐이다. validation label은 진단과
점수 계산에만 사용한다.

## 실행

```bash
python3 -m pip install -r experiments/regime_blend/requirements-oof.txt

python3 scripts/build_regime_blend_oof.py \
  --train /Users/wooh/Documents/dev/open/data/train.csv \
  --output artifacts/regime_blend_oof

python3 scripts/run_regime_blend.py \
  --manifest artifacts/regime_blend_oof/manifest.json \
  --output-dir artifacts/regime_blend_results

python3 scripts/report_regime_blend_weights.py \
  --manifest artifacts/regime_blend_oof/manifest.json \
  --results artifacts/regime_blend_results/regime_blend_results.json \
  --output-dir artifacts/regime_blend_results
```

출력:

- `member_diagnostics.json`, `member_diagnostics.md`
- `regime_blend_results.json`, `regime_blend_results.md`
- `regime_weight_diagnostics.json`, `regime_weight_diagnostics.md`

기본 grid는 `rho=(0, 1e-6, 1e-5)`, `tau=(2000, 10000, 50000)`이다. 이는 사전
등록된 작은 grid이며 test 분포를 보고 바꾸지 않는다.

## 현재 실행 상태

2026-08-20 공식 `train.csv`에서 2021~2024 forward prediction을 다시 생성해
`artifacts/regime_blend_oof/`에 저장했다. 과거 OOF/cache는 발견되지 않아 `team`과
`team_nn`은 현재 source-exact recipe로, `hgb`/`cat`/`nn`은 champion 직렬화
메타에서 복구한 recipe로 재학습했다. 2024 team Platt 진단 계수의 최대 절대 오차는
`1.797e-4`였고 parity gate를 통과했다.

사전 등록 grid 39개를 실행한 결과 채택 기준을 통과한 후보는 0개다. 가장 높은 pooled
gain은 `count_state,rho=1e-5,tau=10000`의 `+8.685e-5`였지만 2022 fold에서
`-1.254e-4`로 악화했고, fold별 global `ours_stage` 가중치도
`0.102 -> 0.455 -> 1.000`으로 불안정했다. 따라서 production 제출물에는 반영하지
않는다. 상세 수치와 재현 한계는
`docs/experiments/2026-08-20-regime-blend-oof-results.md`에 기록한다.

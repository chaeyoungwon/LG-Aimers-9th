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
python3 scripts/run_regime_blend.py \
  --manifest artifacts/regime_blend_oof/manifest.json \
  --output-dir artifacts/regime_blend_results
```

출력:

- `member_diagnostics.json`, `member_diagnostics.md`
- `regime_blend_results.json`, `regime_blend_results.md`

기본 grid는 `rho=(0, 1e-6, 1e-5)`, `tau=(2000, 10000, 50000)`이다. 이는 사전
등록된 작은 grid이며 test 분포를 보고 바꾸지 않는다.

## 현재 실행 상태

2026-08-19 현재 이 checkout에는 공식 train 데이터, champion member OOF,
`../LG-AIMERS_9TH`, `artifacts/validation_hetero/exp23_cache`가 없다. 따라서 실제
fold 수치와 채택 후보는 아직 생성하지 않았다. synthetic unit test 결과를 성능
근거로 사용하지 않는다.

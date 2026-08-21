# Training-only Current-Pitch TrackMan LUPI Distillation Temporal OOF

실행일: 2026-08-21

## Verdict

**KILL**

current-pitch TrackMan을 추가한 privileged teacher는 legal-only teacher보다 linked-row
cross-fit Brier가 세 outer fold에서 모두 약 `1.8e-3` 좋아졌다. 그러나 legal pre-pitch X만
사용하는 distilled student는 선택된 `alpha=.5`에서 2022만 개선되고 2023·2024가
악화됐다. Pooled도 student baseline보다 `-3.0154e-6` 나빠졌다.

고정 coarse blend에서는 champion과의 보완 신호가 2023·2024 및 pooled에 남았지만,
명시된 kill criterion인 “3개 시즌 중 2개 이상 standalone baseline 악화”와 “teacher
signal이 student로 안정적으로 전달되지 않음”을 모두 충족한다. 따라서 이 prototype을
production화하거나 submission ZIP으로 만들지 않는다.

## 실행 범위

- 공식 `train.csv`, 공식 `trackman_history.csv`만 사용
- 기존 reciprocal pitch anchor와 기존 cutoff별 Hungarian HIGH crosswalk 재사용
- Outer temporal OOF: `<=2021→2022`, `<=2022→2023`, `<=2023→2024`
- Inner teacher 3-fold cross-fit
- Alpha: `0`, `.1`, `.25`, `.5`
- Champion coarse blend weight: `.01`, `.025`, `.05`, `.10`
- 새로운 feature/model/alpha/blend 탐색 없음
- `test.csv` 읽기 없음
- production inference, ET25, champion, 제출 ZIP 변경 없음

## A. Data contract

### Legal pre-pitch X

Teacher의 legal block과 student는 기존 ET25가 재사용한 accepted ID-free 52-feature
contract를 그대로 사용했다. `build_hgb52_features()`가 만드는 current-row/as-of
feature이며 다음은 제외된다.

- `row_id`
- `control_success`
- `pitcher_id`
- `batter_id`

ID는 linkage join과 evaluation subset 생성에만 사용하며 teacher/student model input에는
들어가지 않는다.

### Privileged TrackMan Z

실제 공식 파일에서 컬럼 존재를 확인한 뒤 아래 9개 current-pitch 값만 teacher에 추가했다.

| 종류 | 컬럼 |
|---|---|
| 구속 | `rel_speed`, `zone_speed` |
| 회전 | `spin_rate` |
| 무브먼트 | `induced_vert_break`, `horz_break` |
| 릴리스 | `extension`, `rel_height`, `rel_side` |
| 구종 | `pitch_type_group` |

`trackman_id`, `pitcher_trackman_id`, `batter_trackman_id`, 선수 이름이나 외부 identity는
model input에 없다.

### Teacher와 student

첫 prototype은 repository에 이미 설치된 scikit-learn의
`HistGradientBoostingRegressor` 한 종류만 사용했다.

- Teacher: legal X + privileged Z, 60 iterations, 31 leaves, min leaf 300
- Legal teacher control: legal X only, teacher와 동일 recipe
- Student: legal X only, 60 iterations, 31 leaves, min leaf 1,000
- 고정 seed: 42
- Target:

```text
linked row:   (1-alpha) * real_y + alpha * cross_fitted_teacher_prob
unlinked row: real_y
```

`alpha=0`은 동일 student model과 동일 training rows를 real label로만 학습한 baseline이다.
Student validation prediction은 X만 사용하며 TrackMan lookup이나 test-row aggregate에
의존하지 않는다.

## Linkage coverage와 temporal correction

기존 audit의 374,986 reciprocal anchor와 chronology/physical coverage 수치를 exact하게
재검증했다.

- reciprocal anchors: `374,986`
- chronological adjacent pairs: `374,877`
- TrackMan rank strictly increasing: `99.9672%`
- exact adjacent rank gap: `98.6337%`
- 8개 physical field complete: `99.7592%`

다만 기존 `linkage_summary.json`은 scope를 명시적으로 “full-history 2024 HIGH pitcher
mapping; not temporal OOF mapping”이라고 기록한다. 이를 outer OOF에 직접 사용하면 미래
mapping 정보가 섞일 수 있다. 새로운 linkage를 fitting하지 않고, 기존 reciprocal anchor를
각 cutoff에 이미 계산돼 있던 Hungarian HIGH `main_pitcher_id↔tm_pitcher_id` mapping과
교집합하여 보수적인 cutoff-safe anchor만 teacher에 사용했다.

| Validation | Cutoff | Raw anchors ≤ cutoff | Cutoff-safe anchors | Retention | Pitchers |
|---:|---:|---:|---:|---:|---:|
| 2022 | 2021 | 169,551 | 116,203 | 68.54% | 57 |
| 2023 | 2022 | 238,279 | 182,399 | 76.55% | 71 |
| 2024 | 2023 | 312,336 | 286,144 | 91.61% | 93 |

## B. Leakage audit

### Outer cutoff

| Validation | Student/teacher train max season | Teacher linkage max season | Validation TrackMan model rows |
|---:|---:|---:|---:|
| 2022 | 2021 | 2021 | 0 |
| 2023 | 2022 | 2022 | 0 |
| 2024 | 2023 | 2023 | 0 |

전체 TrackMan 파일은 기존 374,986 linkage artifact의 row/physics parity를 감사하기 위해
읽었지만, teacher target 생성에는 위 cutoff-safe row만 전달했다. Validation season의
TrackMan row는 teacher/student 학습에 사용하지 않았다.

### Inner cross-fit

각 outer training set의 cutoff-safe linked row를 deterministic shuffled 3-fold로 나눴다.
각 row의 teacher probability는 그 row가 들어 있지 않은 나머지 2개 fold로 학습한
teacher에서만 생성했다.

| Outer validation | Inner fit/holdout sizes | Fit-holdout overlap |
|---:|---|---:|
| 2022 | 77,468/38,735 · 77,469/38,734 · 77,469/38,734 | 0 |
| 2023 | 121,599/60,800 · 121,599/60,800 · 121,600/60,799 | 0 |
| 2024 | 190,762/95,382 · 190,763/95,381 · 190,763/95,381 | 0 |

- 같은 row를 학습한 teacher의 in-sample prediction: `0`
- Validation TrackMan 사용: `0`
- Outer cutoff 이후 TrackMan/linkage row: `0`
- Student feature의 TrackMan/ID/test dependency: `0`

상세 fold 증명은 `artifacts/lupi_distill_oof/leakage_audit.json`에 저장했다.

## Teacher privileged signal

Teacher 자체에서는 current-pitch TrackMan이 강한 독립 신호였다.

| Outer validation | Linked rows | Legal-only teacher Brier | Privileged teacher Brier | Gain |
|---:|---:|---:|---:|---:|
| 2022 | 116,203 | 0.2467596577 | 0.2449298168 | +0.0018298409 |
| 2023 | 182,399 | 0.2472399629 | 0.2453464356 | +0.0018935274 |
| 2024 | 286,144 | 0.2472535655 | 0.2454480174 | +0.0018055481 |
| Weighted pooled | 584,746 | — | — | **+0.0018378189** |

즉 current-pitch TrackMan과 outcome 사이 signal 부족이 실패 원인은 아니다. 실패 지점은
privileged Z가 없는 student가 그 signal을 legal X만으로 일반화하는 단계다.

## Alpha stability

각 cell은 동일 alpha student의 `baseline Brier - distilled Brier`다. 양수가 개선이다.

| Alpha | 2022 | 2023 | 2024 | Pooled |
|---:|---:|---:|---:|---:|
| .10 | -3.4172e-6 | -1.9775e-5 | -1.2024e-5 | -1.1720e-5 |
| .25 | +1.8582e-5 | -3.8515e-5 | -1.6822e-5 | -1.2220e-5 |
| .50 | +2.8699e-5 | -3.6083e-6 | -3.3400e-5 | **-3.0154e-6** |

사전 등록 grid 안에서 pooled standalone gain이 가장 큰 alpha를 고르고 동률이면 작은
alpha를 택하는 규칙에 따라 `.5`가 선택됐다. 하지만 `.5`도 pooled가 음수이며 세 alpha
모두 2023·2024 방향이 안정적이지 않다.

## C. Temporal results

선택 alpha `.5` 결과다.

| Season | Student baseline Brier | Distilled Brier | Gain | Current champion Brier | Best coarse blend Brier | Weight |
|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 0.2434823810 | 0.2434536824 | +2.8699e-5 | 0.2431150187 | 0.2431152163 | .01 |
| 2023 | 0.2532554230 | 0.2532590313 | -3.6083e-6 | 0.2530255396 | 0.2530122312 | .10 |
| 2024 | 0.2480597892 | 0.2480931893 | -3.3400e-5 | 0.2480642876 | 0.2480271756 | .10 |
| Pooled | 0.2482511849 | 0.2482542002 | **-3.0154e-6** | 0.2480553204 | 0.2480399564 | .10 |

현재 champion OOF는 frozen `champion_final`과 기존 ET25 25-tree prediction cache를
`0.9775/0.0225`로 결합해 계산했다. LUPI 실행 중 ET25 model, feature, tree order,
`n_jobs`, weight를 변경하지 않았다.

## D. Subset stability

Standalone student는 전체 평균보다 subset 불안정성이 더 크다.

- Worst eligible subset: 2023 `game_type=F`, `-2.67e-4`
- 2024 linkage-covered pitcher: `-1.17e-4`
- Pooled linkage-covered pitcher: `-5.5e-5`
- Pooled linkage-non-covered pitcher: `+1.3e-5`
- Pooled pitcher seen: `-1.4e-5`; unseen: `+4.2e-5`
- Pooled cold-start active: `+5.0e-5`; inactive: `-1.3e-5`
- Pooled game type F: `-1.6e-5`; R: `-1.0e-6`

특히 teacher 학습 coverage와 직접 관련된 pitcher에서 오히려 standalone student가
악화됐다. 단순히 linked-row 수가 부족해서 전체 validation에서 gain이 희석된 형태가
아니며, soft probability를 X-only student target에 섞는 방식 자체의 전달 안정성이 낮다.

## E. Complementarity with current champion

| Season | Corr(student, champion) | Mean abs disagreement | Best blend gain vs champion | Weight |
|---:|---:|---:|---:|---:|
| 2022 | 0.978310 | 0.013875 | -1.9760e-7 | .01 |
| 2023 | 0.973319 | 0.015362 | +1.3308e-5 | .10 |
| 2024 | 0.864163 | 0.016494 | +3.7112e-5 | .10 |
| Pooled | 0.959958 | 0.015253 | **+1.5364e-5** | .10 |

Pooled residual covariance는 `0.2479109194`, RMS disagreement는 `0.0197911092`다.
Champion blend의 양수 gain은 distilled prediction에 보완 잔차가 있음을 보여준다. 그러나
2022에서는 가장 작은 `.01`도 음수이고, standalone student가 두 season에서 악화됐기
때문에 이 coarse scan만으로 후속 production candidate를 정당화하지 않는다.

## F. KILL rationale

적용된 kill criterion:

1. 2022/2023/2024 중 2023·2024 두 season에서 student baseline보다 악화
2. Pooled standalone gain `-3.0154e-6`
3. Teacher privileged gain은 weighted pooled `+1.8378e-3`이지만 student 전달은 음수
4. Alpha `.1/.25/.5` 어느 것도 pooled standalone을 개선하지 못함
5. 주요 subset 방향이 불안정하고 worst subset degradation이 큼

따라서 이 결과는 “TrackMan current-pitch 정보가 유효하지 않다”가 아니라 “단순 soft-label
probability distillation이 legal 52-feature student에 안정적으로 전달되지 않는다”는
결론이다.

## Reproducibility and safety

동일 동결 설정으로 전체 실험을 다시 실행해 기존 OOF cache와 비교했다.

- 2022 row order 및 모든 alpha prediction max diff: `0`
- 2023 row order 및 모든 alpha prediction max diff: `0`
- 2024 row order 및 모든 alpha prediction max diff: `0`
- Submission CSV/ZIP 생성: 없음
- Production inference 수정: 없음
- `test.csv` 읽기: 없음

현재 workspace에는 요청서에 명시된 `artifacts/sub_et25_w0225.zip` 자체가 존재하지 않아
그 파일의 전후 checksum을 다시 계산할 수 없었다. LUPI runner는 ZIP 생성·삭제·수정 코드를
포함하지 않는다. Original champion `artifacts/submit_r10_pitcher_w418194.zip`은 SHA-256
`ce10c428c9b6bd88ebf713fcc05b91ce22fc555573f3d11f8deb283637484c47`로 확인했다.

## Artifacts

- `artifacts/lupi_distill_oof/summary.json`
- `artifacts/lupi_distill_oof/leakage_audit.json`
- `artifacts/lupi_distill_oof/teacher_results.csv`
- `artifacts/lupi_distill_oof/alpha_results.csv`
- `artifacts/lupi_distill_oof/blend_results.csv`
- `artifacts/lupi_distill_oof/temporal_results.csv`
- `artifacts/lupi_distill_oof/subset_results.csv`
- `artifacts/lupi_distill_oof/cache/season_{2022,2023,2024}.npz`

Reproduction:

```bash
cd /Users/wooh/Documents/dev/LG-Aimers-9th
.venv/bin/python scripts/validate_lupi_distill_oof.py
.venv/bin/python -m unittest tests.test_lupi_distill_oof
```

## 다음 한 단계

이 direct probability-distillation 축은 종료한다. LUPI를 한 번만 더 검토한다면 soft
probability를 정답에 직접 섞는 대신, `pitch_type_group`과 물리량을 각각 auxiliary target으로
cross-fit 예측하고 legal-X representation이 어떤 privileged latent를 실제로 복원할 수 있는지
먼저 측정하는 고정 multi-task/auxiliary prototype이 다음 한 단계다. 이 아이디어는 이번
결과에 포함하지 않았고 별도 사전 등록 실험으로만 진행해야 한다.

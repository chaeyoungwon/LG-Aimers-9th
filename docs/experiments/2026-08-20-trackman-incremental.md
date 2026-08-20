# TrackMan incremental signal audit

작성일: 2026-08-20

## Final verdict

**C. No stable incremental TrackMan signal**

이 판정은 TrackMan 물리량 자체가 무의미하다는 뜻이 아니다. 이번 round에서 허용된 공식 파일·익명 ID·cutoff-safe 조건만으로는 `train.pitcher_id`와 `trackman_history.pitcher_trackman_id`를 연결할 수 없었고, exact-key coverage가 모든 temporal fold에서 0%였다. Coverage gate를 통과하지 못했으므로 residual association이나 모델 성능을 계산하는 것은 불가능하며, 검증되지 않은 fuzzy identity mapping으로 결과를 만들어내지 않았다.

## Hypothesis

현재 champion의 HGB, CatBoost, NN, team models, form 및 cold-start endpoint가 설명한 이후에도 공식 `trackman_history.csv`의 과거 투구 품질·안정성·구종 구성 정보가 독립적인 temporal OOF Brier 개선을 제공하는지 검증하려 했다.

검증 순서는 다음으로 고정했다.

1. Champion이 TrackMan을 실제 입력으로 쓰는지 meta 감사
2. 공식 schema와 익명 투수 ID 연결 가능성 확인
3. Cutoff-safe coverage 확인
4. Coverage가 충분한 경우에만 association, member, offset, complementarity, blend 평가

3단계에서 중단 조건이 충족됐다.

## Champion TrackMan usage audit

Champion ZIP의 실제 `model/meta.json`과 `model/coldstart_meta.json`을 기준으로 감사했다. ZIP의 `script.py`에는 과거 schema의 dormant `tm` lookup 구현이 남아 있지만, 실제 member의 `groups` 또는 `feature_columns`에 TrackMan 입력이 선언되어 있는지를 판정 기준으로 사용했다.

| Member | TM feature used | Exact TM columns | Declared groups |
| --- | :---: | --- | --- |
| HGB | no | none | matchup, count, rates |
| CatBoost | no | none | matchup, count, rates |
| ours NN | no | none | matchup, count, rates |
| team LightGBM | no | none | team |
| team NN | no | none | matchup, count, rates |
| cold-start expert | no | none | none |

따라서 TrackMan이 champion에서 사용되지 않는다는 기존 감사를 재확인했다.

## Official TrackMan schema

파일에는 2019–2024의 1,793,078개 투구가 있으며 시즌별 행 수는 다음과 같다.

| Season | Rows | Pitchers |
| ---: | ---: | ---: |
| 2019 | 255,957 | 394 |
| 2020 | 279,126 | 431 |
| 2021 | 301,032 | 425 |
| 2022 | 307,637 | 432 |
| 2023 | 315,100 | 447 |
| 2024 | 334,226 | 460 |

실제 제공된 물리량은 `rel_speed`, `spin_rate`, `induced_vert_break`, `horz_break`, `extension`, `rel_height`, `rel_side`, `zone_speed`다. 구종 관련 컬럼은 `tagged_pitch_type`, `auto_pitch_type`, `pitch_type_group`이며 `pitch_type_group`은 fastball 931,120행, breaking 512,851행, offspeed 326,809행, other 22,298행이다. Plate location 좌표 컬럼은 실제 schema에 없으므로 가정하거나 생성하지 않았다.

사전 정의한 저차원 profile은 다음 23개다.

- Quality: 8개 물리량 mean
- Stability: velocity, horizontal/vertical movement, release height/side std
- Pitch mix: fastball/breaking/offspeed/other share와 entropy
- Evidence: `tm_available`, `tm_pitch_count`, `tm_seasons`, `tm_last_season`, `tm_recency_gap`

Career prior와 recent weighted 두 aggregate만 구현했다. Recent weight는 validation cutoff 기준 직전 시즌 `1.0`, 두 시즌 전 `0.5`, 그보다 과거 `0.25`로 고정했으며 label이나 leaderboard로 조정하지 않았다.

## Connection audit

공식 설명서도 TrackMan 파일이 메인 데이터와 1:1 결합 테이블이 아니라고 명시한다. 실제 key는 다음처럼 서로 다르다.

| Source | Pitcher key |
| --- | --- |
| `train.csv` / evaluation rows | `pitcher_id` |
| `trackman_history.csv` | `pitcher_trackman_id` |

두 파일에 이 ID들을 연결하는 crosswalk 컬럼이나 별도 mapping 파일은 없다. 세 validation fold에서 cutoff 이전 TrackMan ID와 validation `pitcher_id`의 exact value intersection도 모두 0이다. 선수 이름·실명 역추적, 외부 데이터, game-row fuzzy linkage, 물리 profile 유사도를 이용한 신원 추론은 이번 요청에서 금지됐거나 정확도를 검증할 수 없으므로 사용하지 않았다.

## TrackMan coverage

Validation season보다 이전 TrackMan season만 허용한 결과다.

| Fold | TM cutoff | Row coverage | Pitcher coverage |
| ---: | ---: | ---: | ---: |
| 2022 | ≤2021 | 0 / 247,472 (0%) | 0 / 390 (0%) |
| 2023 | ≤2022 | 0 / 245,525 (0%) | 0 / 382 (0%) |
| 2024 | ≤2023 | 0 / 253,507 (0%) | 0 / 391 (0%) |

Seen/unseen pitcher와 R/F subset도 모두 0%다. 탐색을 계속하기 위한 최소 row coverage는 label과 무관하게 20%로 고정했으며, 모든 fold가 이를 통과하지 못했다.

## Stable features

평가할 수 없음. Matched row가 없으므로 Q1–Q4 signed residual 및 squared-error gradient를 계산하지 않았다. `feature_stability.csv`와 `residual_association.csv`에는 모든 사전 정의 feature가 `not_evaluated_zero_exact_key_coverage`로 기록되어 있다. 0개 표본의 방향을 stable로 표시하지 않았다.

## TrackMan member and complementarity

평가할 수 없음. Career prior와 recent weighted 모두 row coverage가 0%여서 소형 HGB/LightGBM member를 학습하지 않았다. 따라서 아래 값도 정의되지 않는다.

- TrackMan member Brier
- `corr(tm_member, champion/ours_stage/team_stage)`
- TM/champion row별 win rate
- Champion high-error row에서의 TM win rate

`member_results.csv`와 `complementarity.csv`에는 champion reference Brier와 명시적인 미평가 상태만 기록했다.

## Blend / incremental correction

독립적인 TrackMan member가 존재할 때만 `w = 0.02, 0.05, 0.10, 0.15` simple blend를 검사하기로 했으므로 blend를 실행하지 않았다. 마찬가지로 strongly regularized no-intercept offset과 feature-group ablation도 실행하지 않았다. 임의의 median/zero 대치로 TrackMan이 있는 것처럼 처리하거나 champion calibration을 재학습하지 않았다.

## Safety

- Cutoff: 2022/2023/2024 lookup은 각각 TrackMan ≤2021/≤2022/≤2023으로 제한
- Future exclusion: cutoff 이후 TrackMan 값을 변경해도 이전 profile이 bit-exact하게 유지됨
- Validation-label invariance: nested offset beta는 evaluation season label 변경에 영향받지 않음
- Unseen fallback: `tm_available=0`, `tm_pitch_count=0`, 물리 profile은 missing으로 명시
- Row independence: single/full batch/shuffle/subset/reverse exact lookup 결과 일치
- Champion identity: zero correction에서 `atol=0`으로 champion 복원
- Numeric safety: NaN/Inf correction input 거부, prediction `[0,1]` bounds 확인
- Test isolation: `test.csv`를 읽지 않음
- Production unchanged: `submission/`과 champion ZIP을 수정하지 않음
- Champion SHA-256 before/after: `ce10c428c9b6bd88ebf713fcc05b91ce22fc555573f3d11f8deb283637484c47`

## Reproduction

```bash
.venv/bin/python scripts/validate_trackman_incremental.py
.venv/bin/python -m unittest tests.test_trackman_incremental
```

정량 결과와 미평가 사유는 `artifacts/trackman_incremental/`의 `coverage.csv`, `feature_stability.csv`, `residual_association.csv`, `member_results.csv`, `complementarity.csv`, `blend_results.csv`, `ablation.csv`, `summary.json`에 저장된다.

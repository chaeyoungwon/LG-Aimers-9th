# Training-only TrackMan LUPI pitch-linkage feasibility audit

실행일: 2026-08-21

## 결론

**PASS — teacher/student 실험으로 진행할 수 있는 수준의 보수적 pitch-level linkage가 존재한다.**

이 감사는 모델 학습이나 제출 ZIP 생성을 하지 않았다. `test.csv`를 사용하지 않았고, 공식 `train.csv`와 `trackman_history.csv`만 사용했다.

## 공식 허용 근거

공식 Q&A에서 다음 세 방식이 모두 가능하다고 답변했다.

1. train↔TrackMan 매칭 부분집합에서 현재 투구 구종·TrackMan을 입력한 teacher를 학습하고, pre-pitch만 입력받는 student에 distillation
2. 현재 투구 구종·구속 등을 auxiliary target으로 사용하는 학습
3. privileged 정보가 일부 train 행에만 존재하는 부분 매칭 학습

조건은 추론 시 평가 데이터의 다른 행 정보를 사용하지 않는 것이다.

## linkage 방법

1. 기존 official-data-only pitcher fingerprint crosswalk의 2024 HIGH mapping을 재구성했다.
2. `train.csv`에서 `asof_pitcher_n == asof_pitcher_pitchmix_n`가 전 행에서 성립하고, 각 pitcher의 `asof_pitcher_n`이 중복/누락 없이 1씩 증가하는 것을 확인했다.
3. 다음 main row의 누적 pitch-mix count와 현재 row의 차이를 이용해 training-only audit anchor로 현재 pitch의 `pitch_type_group`을 복원했다. 이 값은 inference feature로 사용하지 않는다.
4. main team ID와 TrackMan team namespace는 HIGH-mapped pitcher-season의 official-data-only vote로 매핑했다.
5. 다음 키가 모두 같은 TrackMan 후보를 찾았다.
   - mapped pitcher
   - season / month / weekday
   - inning / top-bottom
   - balls / strikes / outs
   - batter hand
   - derived current pitch group
   - mapped pitcher team / batter team
6. main row 기준 후보가 정확히 1개인 경우만 남긴 뒤, 같은 TrackMan row가 둘 이상의 main row에 쓰이는 경우를 전부 제거하여 reciprocal one-to-one anchor만 인정했다.

## 결과

- 전체 train rows: 1,475,092
- HIGH-mapped main rows: 561,608
- main-key unique anchors: 377,287
- reciprocal one-to-one anchors: **374,986**
- 전체 train coverage: **25.4212%**
- HIGH-mapped row 내부 coverage: **66.7701%**
- anchor가 존재하는 main pitchers: 109

### 시간순 self-check

reciprocal anchors를 main `asof_pitcher_n` 순서와 TrackMan chronological rank로 비교했다.

- 비교 가능한 adjacent anchor pairs: 374,877
- TrackMan rank strictly increasing: **99.9672%**
- main rank gap == TrackMan rank gap: **98.6337%**
- TrackMan gap >= main gap: **99.7887%**

### privileged physical signal availability

8개 물리 측정값이 모두 non-null인 reciprocal anchors: **99.7592%**

- rel_speed: 99.9976%
- spin_rate: 99.7888%
- induced_vert_break: 99.9973%
- horz_break: 99.9763%
- extension: 99.9893%
- rel_height: 99.9976%
- rel_side: 99.9976%
- zone_speed: 99.9976%

### season별 reciprocal anchors

| season | anchors | full-train coverage |
|---:|---:|---:|
| 2019 | 44,711 | 18.83% |
| 2020 | 53,656 | 21.98% |
| 2021 | 71,184 | 28.81% |
| 2022 | 68,728 | 27.77% |
| 2023 | 74,057 | 30.16% |
| 2024 | 62,650 | 24.71% |

## 해석

이 결과는 pitcher crosswalk가 실명 identity ground truth라는 의미가 아니다. 하지만 teacher 학습용 privileged subset을 만드는 목적에서는 다음 세 조건이 동시에 강하게 만족된다.

- exact shared-state uniqueness
- reciprocal one-to-one linkage
- chronological sequence consistency

따라서 TrackMan historical profile을 test feature로 붙였던 기존 실패축과 달리, 현재 pitch TrackMan을 **training-only supervision**으로 사용하는 LUPI/distillation 실험은 기술적으로 실행할 가치가 있다.

## 다음 실험 gate

다음 단계는 temporal OOF를 지키는 teacher/student prototype이다.

- fold 2022: train/linkage <= 2021 only
- fold 2023: train/linkage <= 2022 only
- fold 2024: train/linkage <= 2023 only
- validation/test prediction input: accepted legal pre-pitch features only
- privileged TrackMan: teacher training rows에서만 사용
- test row aggregation/groupby/rank/rolling 없음
- champion/ET25 immutable

첫 prototype에서는 production ZIP을 만들지 않는다. 먼저 original champion 대비 temporal OOF와 2022/2023/2024 fold consistency만 평가한다.

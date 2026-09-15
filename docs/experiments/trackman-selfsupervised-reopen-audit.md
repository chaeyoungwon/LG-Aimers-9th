# TrackMan Self-Supervised Representation Reopen Audit

실행일: 2026-08-25

## 결론

**TRACKMAN REPRESENTATION IS NOVEL BUT LOW EXPECTED VALUE**

Pitch-shape mixture/repertoire 표현은 기존 historical mean/std profile이나 LUPI와 완전히
같지는 않다. 특히 구종별 centroid separation과 mix entropy는 기존 profile에서 잘
복원되지 않아 **PARTIALLY NOVEL**이다. 그러나 strict high-confidence linkage의 OOF row
coverage가 2022/2023/2024 `26.35%/28.61%/30.00%`이고 cold/low-history coverage는
0%다. 사전 고정한 신규 structural factor와 champion residual의 상관은 2022·2024
모두 절대값 `.0067` 이하이며 방향도 반복되지 않았다.

따라서 target-free representation 모델, champion integration, 제출 ZIP을 만들지 않는다.
21차 champion은 변경하지 않았다.

## A. Existing TrackMan experiments

### Historical profile

`src/trackman_features.py`는 cutoff 이전 공식 TrackMan으로 다음을 이미 계산했다.

- 7개 물리량: `rel_speed`, `spin_rate`, `induced_vert_break`, `horz_break`,
  `extension`, `rel_height`, `rel_side`
- 투수별 weighted mean
- 구종군 내부 weighted std
- TrackMan pitch count와 pitch-type count
- release height/side spread

결과는 R fold `+5.3/-12.0/+12.2 BSS`, 2024 전체 `+7.3 BSS`로 모두 noise floor 안이며,
각 profile과 residual 상관은 `|r|<=.016`이었다. 릴리스 산포 quintile target rate도
`.4931/.4920`으로 평평했다.

### LUPI auxiliary

Legal pre-pitch X가 current-pitch TrackMan 물리량을 auxiliary target으로 예측하도록
학습했다. 이는 physics를 직접 inference input으로 쓰지 않지만 shared control
representation에 TrackMan supervision을 넣는다. Seed 42·43 모두 2022·2024가
악화했고 champion 5% blend는 `+0.88/+25.13/-2.50 BSS`였다.

### Distillation

Legal X + current-pitch TrackMan teacher에서 legal-X-only student로 soft target을
전달했다. Champion 5% blend는 `+0.25/-10.50/-1.09 BSS`로 실패했다.

### 이번 후보와 중복도

```text
cutoff 이전 공식 TrackMan physics
  -> target-free pitch/repertoire encoder
  -> pitcher aggregate
  -> frozen pitcher lookup
```

Control target을 representation pretraining에 쓰지 않고, validation/current pitch
TrackMan도 쓰지 않는다. LUPI/distillation과 목적·data flow가 다르다. 다만 PCA mean/spread
및 covariance는 historical profile의 선형 재표현과 중복되고, repertoire mixture만
실질적으로 새롭다. 최종 novelty 판정은 **PARTIALLY NOVEL**이다.

## B. Linkage와 coverage 재검증

### Reciprocal current-pitch anchor artifact

실제 `reciprocal_pitch_anchors.csv`와 crosswalk를 다시 읽어 확인했다.

| 항목 | 값 |
| --- | ---: |
| 전체 train rows | 1,475,092 |
| strict reciprocal matched rows | 374,986 |
| row coverage | 25.4212% |
| 전체 pitcher | 792 |
| matched main/TM pitcher | 109 / 109 |
| pitcher coverage | 13.7626% |

Season별 strict anchor는 2019 `44,711`, 2020 `53,656`, 2021 `71,184`, 2022
`68,728`, 2023 `74,057`, 2024 `62,650`이며 기존 문서와 일치한다.

### 두 mapping contract의 차이

Historical profile 코드는 season/month/day-of-week pitch-count fingerprint, mutual match,
hand filter, team-consensus를 사용한다. 이 wider contract는 cutoff별 422/486/541명을
매핑하고 validation row의 `77.79%/83.30%/77.05%`에 embedding을 붙일 수 있다.

LUPI audit의 strict HIGH Hungarian crosswalk는 cutoff 2021/2022/2023에서
90/91/108명만 허용한다. Frozen physical embedding을 새 submission candidate로 판단할
때는 오매핑 위험이 낮은 strict contract를 기준으로 삼았다.

| Validation | HIGH mapped IDs | covered validation pitchers | pitcher coverage | row coverage |
| ---: | ---: | ---: | ---: | ---: |
| 2022 | 90 | 72/390 | 18.46% | **26.35%** |
| 2023 | 91 | 65/382 | 17.02% | **28.61%** |
| 2024 | 108 | 78/391 | 19.95% | **30.00%** |

Strict embedding은 세 시즌 모두 cold-start와 `asof_pitcher_n<200` row coverage가 0%다.
2022/2023/2024 high-history row 안에서만 `27.99%/30.00%/31.83%`를 덮는다. 즉 기존
champion이 이미 가장 많은 pitcher history를 가진 population에만 적용된다.

## C. Covered-subset champion error

Wider historical-profile mapping으로 availability만 진단했을 때:

| Season | coverage | all Brier | covered Brier | uncovered Brier | covered excess vs all |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 77.79% | .243132 | .244775 | .237376 | **-659.58 BSS** |
| 2023 | 83.30% | .252347 | .251616 | .255992 | +292.30 BSS |
| 2024 | 77.05% | .247655 | .247757 | .247313 | **-40.76 BSS** |

표의 excess는 `all Brier - covered Brier`라 양수가 covered 구간이 쉽다는 뜻이다.
2022·2024 covered pitcher는 champion의 반복 취약구간이 아니다. 2023만 쉬운 방향이며,
이를 미래 일반화 근거로 쓰지 않는다.

## D. Representation candidate 정의와 ranking

점수는 0~5, redundancy는 높을수록 기존 profile과 중복, cost는 높을수록 비싸다.

| Rank | Representation | Novelty | Coverage | Temporal safety | Redundancy | Residual potential | +10 plausibility | Cost |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Pitch-shape mixture/repertoire | 4 | 2 | 5 | 2 | 1 | 1 | 3 |
| 2 | PCA/linear latent mean+spread | 1 | 2 | 5 | 5 | 1 | 1 | 1 |
| 3 | Target-free AE/contrastive | 3 | 2 | 5 | 3 | 1 | 1 | 5 |

### PCA / linear latent

Pitch-level standardized 8개 physics(`zone_speed` 포함 가능)를 PCA로 줄인 뒤 pitcher별
latent mean/spread를 저장하는 구조다. Linear transform의 mean/covariance는 기존 7개
mean/std로 대부분 결정되므로 신규성이 낮다. 학습 대상으로 보내지 않는다.

### Pitch-shape mixture / repertoire

Pitch-type group별 centroid와 weight, centroid separation, normalized mix entropy를
보존한다. 기존 profile이 구종별 centroid/weight를 최종 feature로 남기지 않아 세 후보
중 유일하게 명확한 신규성이 있다.

### Target-free autoencoder / contrastive

Physics reconstruction 또는 augmentation consistency로 pitch encoder를 만들 수 있다.
그러나 input information은 mixture 후보와 같고, 복잡도를 늘려도 residual 근거를
새로 만들지는 않는다. Deep architecture search는 하지 않는다.

## E. Novelty 정량화

Target을 전혀 사용하지 않고 사전 고정한 세 구조를 계산한 뒤, 기존 historical profile로
복원하는 5-fold Ridge CV R²를 측정했다. Strict mapping 결과:

| Factor | 2022 cutoff CV R² | 2024 cutoff CV R² | 판정 |
| --- | ---: | ---: | --- |
| repertoire separation | -.017 | -.170 | **NOVEL** |
| normalized mix entropy | +.326 | -.205 | **NOVEL/PARTIAL** |
| within-type covariance magnitude | +.805 | +.779 | **HIGHLY REDUNDANT** |

따라서 “평균 profile이 multimodality를 잃었다”는 가설 자체는 사실이다. 하지만 신규성이
곧 control residual predictiveness를 뜻하지 않는다.

## F. Residual evidence

Strict high-confidence covered rows에서 사전 고정한 factor와 champion residual/loss를
진단했다. Rule을 만들거나 threshold를 선택하지 않았다.

| Season | Factor | corr(factor, residual) | corr(factor, squared error) | Q1→Q5 loss 방향 |
| ---: | --- | ---: | ---: | --- |
| 2022 | repertoire separation | -.00143 | +.00899 | Q5 worse |
| 2024 | repertoire separation | +.00268 | +.01356 | Q5 worse, 비단조 |
| 2022 | mix entropy | +.00022 | -.01190 | Q5 better |
| 2024 | mix entropy | -.00340 | +.00643 | **방향 반전** |
| 2022 | within-type covariance | -.00068 | +.00827 | 거의 평평 |
| 2024 | within-type covariance | +.00667 | -.00875 | **방향 반전** |

절대 residual correlation 최대가 `.00667`, squared-error correlation 최대가 `.01356`에
불과하다. Separation의 quintile error는 양쪽 모두 Q5가 나쁘지만 중간 quintile이 크게
비단조이고 residual signed relation은 반전한다. Entropy/covariance는 2022·2024 방향도
유지하지 못한다. Common deployable structural evidence는 **없음**이다.

## G. Historical-profile failure diagnosis

| 원인 | 판정 |
| --- | --- |
| LOW COVERAGE | strict mapping에는 해당; wider profile은 77~83%라 단독 원인은 아님 |
| REDUNDANCY | mean/std/PCA/covariance에 해당 |
| TEMPORAL INSTABILITY | 해당 — historical profile fold 부호 불일치 |
| OVER-SMOOTHING / multimodality loss | 실제 존재 — mixture factor R²가 낮음 |
| NO RESIDUAL LINK | **primary failure** |

Mixture representation은 over-smoothing을 해결하지만 primary failure인 residual link
부재는 해결하지 못한다. 따라서 historical profile 실패 원인을 충분히 뒤집지 못한다.

## H. Coverage × upper bound

Official metric denominator를 전체 validation prevalence로 고정할 때, strict-covered row만
바꿔 overall `+10 BSS`를 얻기 위해 필요한 covered-subset 평균 개선은:

| Season | strict coverage | required covered-subset gain | perfect covered prediction theoretical upper bound |
| ---: | ---: | ---: | ---: |
| 2022 | 26.35% | **+37.95 BSS** | +25,799.89 BSS |
| 2024 | 30.00% | **+33.33 BSS** | +29,717.07 BSS |

수학적 ceiling은 충분하므로 coverage만으로 +10이 불가능한 것은 아니다. 그러나 실제
신규 factor의 residual correlation이 사실상 0이고 기존 profile model gain도 noise
수준이어서 `+33~38 BSS`를 안정적으로 얻을 경험적 근거는 없다. 현실성 판정은
**UNREALISTIC GIVEN OBSERVED SIGNAL**이다.

## I. Fallback 및 ID memorization

- global zero embedding: 유일하게 명확하고 안전한 fallback. Strict contract에서는
  70~74% row가 동일 fallback이 된다.
- team-average embedding: train-only면 규칙상 가능하지만 team ID proxy가 되어 물리
  similarity보다 identity/context를 학습할 위험이 크다.
- official-history inferred embedding: 기존 LUPI/student 축과 중복되므로 금지.
- test 등장 빈도/roster/groupby로 fallback 생성: 행 독립성 위반으로 금지.

Champion의 team LightGBM은 이미 `pitcher_id`를 사용한다. Frozen embedding은 같은
pitcher 안의 고정 code로만 작동하면 ID의 다른 표현에 불과하다. 유사 투수 간 physical
transfer가 가치의 핵심이어야 하지만, separation/entropy factor에서 그런 residual
repeatability가 관찰되지 않았다.

## J. Rule safety contract

공식 TrackMan 사용 자체와 LUPI 1~3은 [DACON 운영진 Q&A](https://dacon.io/en/competitions/official/236743/talkboard/417082)에서
허용된 것으로 확인된다. 안전한 self-supervised flow는 다음뿐이다.

```text
outer cutoff 이전 공식 TrackMan
  -> target-free fit/aggregate
  -> frozen main pitcher_id lookup
  -> validation/test current row pitcher_id
```

- 2022 embedding: TrackMan `<=2021`
- 2023 embedding: TrackMan `<=2022`
- 2024 embedding: TrackMan `<=2023`
- 2025 inference: official TrackMan `2019~2024`
- validation/current pitch TrackMan: 사용 금지
- validation season future TrackMan: 사용 금지
- `control_success` during representation fit: 사용 금지
- test-wide mapping/frequency/statistics: 사용 금지

이 contract면 row independence는 충족 가능하다. 이번 기각은 rule safety가 아니라
expected value 때문이다.

## 최종 판정

- Novelty: **PARTIALLY NOVEL**
- Best representation family: pitch-shape mixture/repertoire
- Decision: **TRACKMAN REPRESENTATION IS NOVEL BUT LOW EXPECTED VALUE**
- Next action: **DO NOT TRAIN**
- Champion: 21차 artifact 그대로 유지

새로운 공식 TrackMan ID linkage가 제공되거나, target을 보지 않은 사전 factor가
2022·2024 residual과 반복적으로 연결된다는 새 근거가 생길 때만 재감사한다.

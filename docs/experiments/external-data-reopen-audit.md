# External-Data Reopen Audit

실행일: 2026-08-24

## 결론

**NO ACTIONABLE EXTERNAL DATA AXIS**

외부 선수·구장·팀 메타데이터는 공개적으로 존재하지만, Phase 2 공식 규칙이 외부
데이터를 명시적으로 금지한다. 운영진도 2026-08-21 ABS/리그 정보를 외부 데이터로
사용할 수 있는지 묻는 질문에 “외부 데이터는 사용 불가능”이라고 답했다. 따라서
public, free, pre-pitch, manually collected 여부와 무관하게 모델 feature나 학습 label로
사용할 수 없다.

규칙과 별개로 linkage도 실패한다. 공식 main table은 이름/KBO player code/game date/
game ID/stadium을 제공하지 않고 익명 `pitcher_id` 792개, `batter_id` 830개, team ID
13개만 제공한다. KBO의 이름 기반 선수·로스터 데이터와 direct exact key가 없고,
월·요일·팀만으로 stadium schedule을 붙이는 것도 경기 단위로 모호하다.

Champion ZIP은 수정·재생성하지 않았고 외부 데이터를 내려받거나 학습하지 않았다.

## A. Rule status

| 질문 | 판정 | 근거 |
| --- | --- | --- |
| publicly available external data allowed? | **NO** | 공식 규칙 2-3: Phase 2 공식 데이터 외 외부 데이터 금지 |
| competition-period restriction? | 해당 없음 | 기간 제한보다 blanket prohibition이 우선 |
| licensing restriction? | 외부 데이터에는 해당 없음 | 외부 데이터 자체가 금지; 공개 pretrained weight에는 별도 공개·비상업 이상 license 조건 |
| manual data collection allowed? | **NO** | 수집 방식 예외 없음 |
| API allowed during training? | **NO for external data** | 외부 API 모델 금지이며 외부 데이터도 별도 금지 |
| internet allowed during inference? | **NO** | 코드 제출 환경은 package 설치 외 오프라인 |
| 2025 pre-pitch metadata allowed? | **NO if external** | timestamp가 안전해도 공식 제공 데이터가 아니므로 금지 |
| post-game information allowed? | **NO** | 투구 이전 정보만 사용 가능; 외부 데이터 금지와 별개로 temporal leakage |
| unclear restrictions | **NONE material** | 외부 데이터 금지는 규칙·운영진 답변 모두 명시적 |

공식 근거:

- [DACON 대회 규칙](https://www.dacon.io/en/competitions/official/236743/overview/rules):
  외부 데이터 사용 금지, 평가 행 독립성, 외부 API 제한을 각각 명시한다.
- [DACON 운영진 Q&A](https://dacon.io/en/competitions/official/236743/talkboard/417082):
  2026-08-21 운영진이 ABS 관련 외부 정보 사용 질문에 외부 데이터 사용 불가라고 답했다.
- [대회 설명](https://www.dacon.io/en/competitions/official/236743/overview/description):
  투구 이전 정보만 사용하며 평가 서버는 package 설치 외 오프라인이다.

기존 HANDOFF의 “공식 Phase-2 데이터 외 외부 데이터 금지” 요약은 현재 공개된 공식
규칙 원문과 일치한다.

## B. Candidate source inventory

대규모 scraping/API 호출 없이 공개 page/schema만 확인했다.

| Source | Information | Years/availability | Stable identifier | Linkage | Rule safety |
| --- | --- | --- | --- | --- | --- |
| [KBO 선수 조회](https://web1.koreabaseball.com/Player/Search.aspx) | 이름, 팀, 포지션, 생년월일, 체격, 출신교 | current/historical player search | 이름/선수 page 중심 | official anonymous ID와 direct key 없음 | **PROHIBITED** |
| [KBO 선수 등록 현황](https://www.koreabaseball.com/player/register.aspx) | 날짜별 1군 roster, 포지션, 생년월일, 체격 | 날짜 선택형 current/history 가능 | 이름·팀·등번호 | 이름/날짜가 official row에 없음 | **PROHIBITED** |
| KBO 선수 이동 현황 | 이적·등록 이동 | KBO player section에 공개 | 이름·팀·일자 | direct key 없음 | **PROHIBITED** |
| [2025 KBO 소속 선수 명단](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=11371) | 시즌 개막 전 roster, 신인, 구단 | 2025-02-11 공개 | 이름·구단 | direct key 없음 | **PROHIBITED** |
| [2025 KBO 일정](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=10341) | 경기일, 상대팀, 개최 구장 | 2024-12-20 사전 공개 | date/team/venue | official row에 date/game/venue 없음 | **PROHIBITED** |

KBO 페이지 footer는 “All Rights Reserved”를 표시하며 별도의 machine-readable reuse
license는 이번 소량 schema 확인에서 찾지 못했다. 하지만 license 판단 전에 대회 규칙이
사용을 금지하므로 추가 terms 조사나 수집을 진행하지 않았다.

## C. ID linkage audit

### Player

- official: 익명 numeric `pitcher_id` 792개, `batter_id` 830개
- external KBO: 선수명, 팀, 등번호, 생년월일 또는 site 내부 page
- 공통 direct join key: 없음
- exact match coverage from available keys: **0%**
- ambiguous/manual reconciliation: 필요, 하지만 규칙상 수행 금지
- unmatched under direct-key contract: **100%**

팀·손·누적 투구 통계로 실제 선수를 역추정하는 방식은 exact identifier linkage가 아니며,
동명이인·이적·외국인·퓨처스/1군 이동에 취약하다. 또한 금지된 외부 데이터를 official
행에 연결하기 위한 작업이므로 시도하지 않았다.

### Team

- official team ID: 13개 익명 값(12~25 범위)
- KBO 1군 구단: 이름 기반 10개 구단
- direct key: 없음
- 통계/상대구성으로 team identity를 역추정할 수 있어도 direct exact mapping이 아니고
  external feature 사용 금지를 우회하지 못한다.

### Venue/game

- official row에는 season, month, day-of-week, inning/half, pitcher/batter team은 있지만
  calendar date, game ID, stadium은 없다.
- 동일 month/day-of-week/team 조합에 여러 경기가 가능하므로 schedule exact join 불가.
- neutral venue, shared stadium, doubleheader, postponed game을 안전하게 구분할 수 없다.

## D. Champion redundancy

Champion은 test-visible predictor 47/47과 23개 derived 표현을 사용한다.

| External axis | Novelty classification | Existing overlap |
| --- | --- | --- |
| age / birth date | **NOVEL** | `asof_*_n`과 career stage가 일부 중복하지만 생물학적 age는 없음 |
| debut/service/rookie | **PARTIALLY REDUNDANT** | `asof_pitcher_n`, `asof_batter_n`, cold-start state와 강한 중복 |
| height/weight | **NOVEL** | hand 외 신체 정보 없음 |
| position/role/foreign status | **PARTIALLY REDUNDANT** | hand, team, usage/history count가 일부 proxy |
| transfer/roster continuity | **PARTIALLY REDUNDANT** | current team ID와 as-of history가 일부 변화 흡수 |
| stadium dimensions/roof/surface | **NOVEL** | home/away expectancy와 team context가 일부 간접 proxy |
| park factor | **HIGHLY REDUNDANT / temporally risky** | outcome-derived season statistic이며 team/context와 중복; current-season 값은 post-game 누수 |
| manager/rule environment | **PARTIALLY REDUNDANT** | season/game_type가 broad regime을 이미 표현 |

정보 novelty가 있어도 legality와 linkage가 0이면 actionable candidate가 아니다.

## E. Coverage 및 missingness

직접 join key가 없으므로 모든 player/venue 후보의 **verified exact row coverage는 0%**다.
이 상태에서 fuzzy/manual mapping을 만들면 missing indicator가 선수 cohort·리그·팀을
암묵적으로 외우게 된다.

규칙이 향후 변경된다는 가정 아래의 사전 최소 기준은 다음이 합리적이다.

- player metadata: row coverage `>=90%`, unique pitcher/batter exact linkage `>=95%`,
  ambiguous match `<0.5%`
- roster/transfer: row coverage `>=90%`, availability timestamp가 각 row 이전임을 증명
- stadium: row coverage `>=95%`, date+game+venue exact key 필수

현재 contract는 세 축 모두 이 기준을 만족하지 않는다.

## F. Candidate external axes — expected value

점수는 0~5이며 Safety는 현재 대회 규칙 기준이다. Cost는 높을수록 비싸다.

| Rank | Axis | Novelty | Coverage | Safety | Linkage reliability | Temporal safety | Residual potential | Expected 2024 gain | Cost | +10 plausibility |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | player age/body/career stage | 4 | 0 | 0 | 0 | 5 | 3 | 0 | 4 | **NO** |
| 2 | stadium fixed metadata | 4 | 0 | 0 | 0 | 4 | 3 | 0 | 4 | **NO** |
| 3 | preseason roster/transfer continuity | 3 | 0 | 0 | 0 | 4 | 2 | 0 | 5 | **NO** |

세 항목은 “남긴 실험 후보”가 아니라 source inventory에서 상대적으로 novel한 순서다.
모두 legality·linkage gate에서 탈락하므로 실제 Top candidate는 0개다.

## G. Rejected axes

### Player age / physical metadata

- availability timestamp: birth date/height/weight는 선수 등록 전부터 알려진 고정 정보
- rejection: external data prohibited; anonymous ID linkage 없음; age 효과가 2022·2024
  champion residual에 반복된다는 프로젝트 내부 근거도 없음

### Debut/service/rookie/role/foreign status

- availability timestamp: 시즌 개막 전 또는 선수 등록 시점
- rejection: prohibited; direct linkage 없음; official history counts/cold-start와 중복 큼

### Transfer/roster status

- availability timestamp: 공시 시점 이후만 안전
- rejection: prohibited; official row에 real team/player identity와 date가 없어 point-in-time
  join 불가; test 등장 선수 목록으로 roster를 재구성하는 것도 행 독립성 위반

### Stadium/dimensions/roof/surface

- availability timestamp: fixed venue facts는 경기 전 안전
- rejection: prohibited; official row에 game date/ID/stadium 없음; home-team proxy는 neutral/
  shared venue를 구분하지 못함

### Park factor

- availability timestamp: 직전 시즌 고정 factor만 temporal-safe; 2025 full-season factor는
  post-game leakage
- rejection: prohibited, outcome-derived, team/home expectancy와 중복, venue linkage 없음

### Manager/coaching/rule change

- availability timestamp: 공식 발표 이후 또는 시즌 개막 전
- rejection: prohibited; team/season broad proxy와 중복; row-level residual을 `+10 BSS`
  개선할 구조적 근거 없음

## H. +10 BSS gate

외부 axis가 학습 단계로 가려면 legality, exact linkage, high coverage, champion과의 낮은
중복, 2022·2024 공통 residual 근거가 모두 필요하다. 현재는 가장 앞단의 legality와
linkage가 모두 0이다. 따라서 효과 추정, sample collection, residual join, 모델 학습,
multi-seed validation을 수행하지 않는다.

## 최종 판정

- Best external axis: **NONE**
- Decision: **NO ACTIONABLE EXTERNAL DATA AXIS**
- Next action: **DO NOT TRAIN**
- Champion: 21차 그대로 유지

규칙이 명시적으로 변경되고 주최 측이 익명 ID↔외부 stable identifier mapping까지
제공하는 경우에만 이 감사를 다시 연다.

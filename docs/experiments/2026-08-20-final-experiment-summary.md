# Final experiment summary

작성일: 2026-08-20

## Frozen champion

- LB BSS: `1016.4442212358`
- artifact: `artifacts/submit_r10_pitcher_w418194.zip`
- SHA-256: `ce10c428c9b6bd88ebf713fcc05b91ce22fc555573f3d11f8deb283637484c47`
- size: `3,345,014 bytes`
- cold-start expert weight: `0.4181944103545871`

이 문서는 최종 연구 종료 기록이다. 아래 축은 모두 닫혔으며, positive local signal도 새
candidate나 leaderboard probe를 정당화하지 않는다. 최종 champion의 모델, feature,
calibration, seed, form coefficient 및 ensemble weight는 변경하지 않는다.

## Closed research axes

`gain`은 명시한 reference보다 낮아진 Brier이므로 양수가 개선이다. `—`는 LB 제출을 만들지
않았다는 뜻이다.

| Research axis | Hypothesis | Local / temporal OOF result | LB result | Closure / rejection reason |
| --- | --- | --- | --- | --- |
| Regime-aware blend | game type, hand, count 등 row-local regime별로 ours/team 비중을 바꾸면 고정 `0.4/0.6`보다 안정적이다. | 가장 좋은 count-state 후보는 2022/2023/2024 `-1.254e-4 / +2.389e-5 / +3.550e-4`, pooled `+8.685e-5`; 기존 form+cold endpoint 후 pooled `+6.430e-5`였다. | — | 2022 손실이 사전 정의 worst-fold 한도 `-1e-5`를 한 자릿수 이상 초과했다. 시즌 간 방향이 반전되어 package를 만들지 않았다. |
| Global ensemble weight optimization | 모든 행에 하나의 ours/team 최적 비중을 적용하면 fixed blend를 개선한다. | 최선 global 후보는 2022/2023/2024 `-1.359e-4 / +2.679e-5 / +3.580e-4`, pooled `+8.536e-5`. validation별 ours weight가 `0.102 → 0.455 → 1.000`으로 이동했다. | — | 단일 optimum조차 계절 drift가 크고 2022를 악화했다. 현재 `0.4/0.6`을 유지한다. |
| Hierarchical shrinkage blend | sparse regime optimum을 global optimum으로 `n/(n+tau)` shrink하면 분산을 줄일 수 있다. | game type, hand, count, game-type×count family를 사전 정의한 `tau`로 평가했다. pooled 최대는 count-state `tau=10000`의 `+8.685e-5`였으나 worst fold는 `-1.254e-4`; game-type family는 pooled `-1.051e-5`였다. | — | Shrinkage가 최신 fold gain과 과거 fold loss의 충돌을 제거하지 못했다. 통과 family가 0개였다. |
| General residual correction | row context, as-of reliability, form/cold flag로 champion residual을 직접 예측할 수 있다. | Ridge/Huber/depth-2 tree의 pooled residual correlation은 각각 `-0.0152 / -0.0249 / -0.0046`; 모든 fold/model R²가 음수였다. 가장 덜 나쁜 bounded correction도 pooled `-3.295e-5 / -5.898e-5 / -8.504e-6`였다. | — | Residual mean보다도 일반화가 나빴고 passing correction이 0개였다. |
| Residual cell correction | count×hand cell의 OOF residual mean을 shrink해 기존 form endpoint를 보완한다. | 로컬 OOF proxy에서는 개선했다. | `983.5132453353`, 당시 champion 대비 `-11.7877` | 대규모 LB 하락으로 local-to-private 전이가 부정되었다. 같은 cell correction을 다시 열지 않는다. |
| Model disagreement gating | five-member std/range 또는 ours-team gap이 높은 행에 다른 blend/correction을 적용한다. | High-low Brier가 member std `-0.018554 / +0.011294 / -0.002698`, range `-0.020320 / +0.012028 / -0.002571`, ours-team `-0.013962 / +0.026770 / -0.004041`로 fold마다 반전했다. | — | Disagreement가 반복 가능한 difficulty/reliability 신호가 아니어서 후보를 만들지 않았다. |
| Pitcher/batter form retuning | 기존 season-to-date form의 alpha를 다시 맞추면 cold endpoint 위에서도 개선한다. | 기존 pitcher/batter form은 champion에 유지했다. 더 강한 batter `alpha=0.20`은 local proxy와 LB가 불일치했고, cold 위 joint optimum `alpha=0.09491537`도 전이가 실패했다. | batter `alpha=.20`: `994.9143132327` (`-12.5901`); joint alpha: `1015.7752537314` (`-0.6690`) | 기존 form endpoint는 동결하되 추가 alpha 재튜닝은 LB에서 두 번 실패했으므로 종료한다. |
| Pitcher nonlinear form shape | positive/negative pitcher form에 별도 slope를 주면 남은 비대칭 residual을 포착한다. | cap `.005`: 2023 `+3.25017e-5`, 2024 `+9.65431e-7`, pooled `+1.64813e-5`. 하지만 2024 full-count `-4.71133e-5`, negative-form `-1.13056e-5`; fitted slope 크기도 크게 이동했다. | — | 전체 gain이 미미하고 사전 정의 subset safety를 위반했다. production candidate를 만들지 않았다. |
| Cold-start scalar weight tuning | unseen regular-season pitcher에 ID-free expert를 일정 비율 혼합하면 개선한다. | Forward OOF에서 3개 fold 중 2개 개선, pooled gain 약 `+3.62e-4`. | `w=.50`: `1016.102132613`; `w=.75`: `1010.8164029281`; 이차 복원 `w=.4181944104`: champion `1016.4442212358` | `w=.4181944103545871`만 최종 채택·동결했다. 더 큰 weight는 과혼합이었고 추가 scalar 탐색/LB probing은 종료한다. |
| Cold-start segment boosting | runners=2 또는 특정 same-hand cold row에서 expert weight를 `.60`으로 높이면 개선한다. | Row-local forward validation에서 반복 개선한 segment만 선택했다. | `1015.0710565406` (`-1.0311`) | Segment 효과가 private distribution으로 전이되지 않았다. 전역 champion weight를 유지한다. |
| Cold-start seed ensemble | cold expert seed `42/43/44` 평균이 단일 seed 분산을 줄인다. | 로컬 OOF에서 단일 seed 42보다 소폭 개선했다. | `1015.4525572004` (`-0.6496`) | Seed 평균의 local gain이 LB에서 재현되지 않았다. 단일 seed 42를 유지한다. |
| TrackMan incremental feature | cutoff 이전 구종 품질·안정성이 champion에 독립적인 정보를 더한다. | 공식 `train.pitcher_id`와 `trackman_history.pitcher_trackman_id` 사이 crosswalk가 없어 exact cutoff-safe coverage가 2022/2023/2024 모두 `0%`; member/correction score는 미평가했다. | — | ID namespace가 다르며 이름·외부 데이터·fuzzy matching은 규정상 사용하지 않았다. TrackMan은 의도적으로 배제한다. |

## Final decision

연구 상태는 **종료**다. 현재 제출 대상은 위 frozen champion 하나이며, 추가 alpha/weight/seed,
feature, residual/calibration, TrackMan 연결 또는 leaderboard probe를 수행하지 않는다. 최종
제출 가능성은 점수 실험이 아니라 `docs/final/champion-code-audit.md`의 재현·규정 감사 결과로만
판정한다.

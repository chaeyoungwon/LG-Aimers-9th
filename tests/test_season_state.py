import numpy as np
import pandas as pd

from src.season_state import (
    context_interaction_features,
    exact_success_features_by_season,
    recent_form_feature,
    state_vs_career_features,
)


def test_recent_form_feature_uses_current_season_baseline_row_by_row():
    df = pd.DataFrame(
        {"asof_pitcher_prev5_game_success_rate": [0.61, np.nan, 0.40]},
        index=[10, 20, 30],
    )
    state = pd.DataFrame(
        {"cur_p_succ": [0.51, 0.48, np.nan]}, index=df.index
    )

    got = recent_form_feature(df, state)

    assert got.index.equals(df.index)
    assert got.columns.tolist() == ["prev5_vs_cur_p_succ"]
    assert np.isclose(got.loc[10, "prev5_vs_cur_p_succ"], 0.10)
    assert np.isnan(got.loc[20, "prev5_vs_cur_p_succ"])
    assert np.isnan(got.loc[30, "prev5_vs_cur_p_succ"])


def test_recent_form_feature_is_order_independent():
    df = pd.DataFrame(
        {"asof_pitcher_prev5_game_success_rate": [0.55, 0.42, 0.63]},
        index=[1, 2, 3],
    )
    state = pd.DataFrame({"cur_p_succ": [0.50, 0.47, 0.60]}, index=df.index)
    expected = recent_form_feature(df, state)
    order = [3, 1, 2]

    shuffled = recent_form_feature(df.loc[order], state.loc[order])

    pd.testing.assert_frame_equal(shuffled.sort_index(), expected.sort_index())


def test_context_interactions_are_fixed_row_local_gates():
    df = pd.DataFrame(
        {
            "balls_before": [1, 3],
            "strikes_before": [2, 0],
            "pitcher_hand": [1, 2],
            "batter_hand": [2, 2],
        },
        index=[10, 20],
    )
    state = pd.DataFrame(
        {
            "cur_p_succ": [0.51, 0.48],
            "cur_p_mid": [0.10, 0.11],
            "cur_p_ball": [0.20, 0.21],
            "cur_p_rev": [0.09, 0.08],
            "cur_p_strk": [0.60, 0.59],
        },
        index=df.index,
    )

    got = context_interaction_features(df, state)

    assert got.shape == (2, 80)
    assert got.loc[10, "p_succ_x_count_1_2"] == np.float32(0.51)
    assert np.isnan(got.loc[10, "p_succ_x_count_3_0"])
    assert got.loc[10, "p_succ_x_hand_1_2"] == np.float32(0.51)
    assert np.isnan(got.loc[10, "p_succ_x_hand_2_2"])
    assert got.loc[20, "p_succ_x_count_3_0"] == np.float32(0.48)


def test_state_vs_career_features_are_row_local_differences():
    df = pd.DataFrame(
        {
            "asof_pitcher_success_rate": [0.45, 0.52],
            "asof_pitcher_middle_rate": [0.10, 0.11],
            "asof_pitcher_ball_rate": [0.20, 0.19],
            "asof_pitcher_reverse_rate": [0.25, 0.18],
            "asof_pitcher_strike_rate": [0.60, 0.62],
            "asof_pitcher_fastball_rate": [0.50, 0.48],
            "asof_pitcher_breaking_rate": [0.30, 0.32],
            "asof_pitcher_offspeed_rate": [0.20, 0.20],
            "asof_batter_success_rate": [0.47, 0.49],
            "asof_batter_middle_rate": [0.12, 0.10],
        },
        index=[10, 20],
    )
    state = pd.DataFrame(
        {
            "cur_p_succ": [0.50, 0.50],
            "cur_p_mid": [0.08, 0.12],
            "cur_p_ball": [0.18, 0.20],
            "cur_p_rev": [0.24, 0.16],
            "cur_p_strk": [0.64, 0.61],
            "cur_mix_fb": [0.55, 0.45],
            "cur_mix_br": [0.25, 0.35],
            "cur_mix_off": [0.20, 0.20],
            "cur_b_succ": [0.51, np.nan],
            "cur_b_mid": [0.09, 0.11],
        },
        index=df.index,
    )

    got = state_vs_career_features(df, state)

    assert got.shape == (2, 10)
    assert np.isclose(got.loc[10, "cur_p_succ_vs_career"], 0.05)
    assert np.isclose(got.loc[10, "cur_mix_fb_vs_career"], 0.05)
    assert np.isnan(got.loc[20, "cur_b_succ_vs_career"])
    shuffled = state_vs_career_features(df.iloc[::-1], state.iloc[::-1])
    pd.testing.assert_frame_equal(shuffled.sort_index(), got.sort_index())


def test_exact_success_boundary_includes_last_prior_season_result():
    df = pd.DataFrame(
        {
            "season": [2023, 2023, 2024, 2024],
            "pitcher_id": [1, 1, 1, 1],
            "batter_id": [7, 7, 7, 7],
            "asof_pitcher_n": [10, 11, 12, 13],
            "asof_pitcher_success_rate": [0.5, 6 / 11, 7 / 12, 8 / 13],
            "asof_batter_n": [20, 21, 22, 23],
            "asof_batter_success_rate": [0.5, 11 / 21, 12 / 22, 13 / 23],
            "control_success": [1, 1, 1, 0],
        }
    )

    got = exact_success_features_by_season(df)

    # 2023 마지막 행 직전 count=6/11, 마지막 결과=1 -> 완결 경계 7/12.
    # 2024 첫 행 as-of도 7/12라 당해 시즌 표본은 0개이고 결측이어야 한다.
    assert np.isnan(got.loc[2, "exact_p_n"])
    assert np.isnan(got.loc[2, "exact_p_succ"])
    # 다음 행에서는 2024 첫 결과 하나만 정확히 복원된다.
    assert got.loc[3, "exact_p_n"] == np.float32(1.0)
    assert got.loc[3, "exact_p_succ"] == np.float32(1.0)

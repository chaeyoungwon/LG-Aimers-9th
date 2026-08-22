import pandas as pd

from scripts.validate_role_state import role_features


def fixture():
    return pd.DataFrame({
        "inning": [1, 5, 9], "asof_pitcher_n": [0, 500, 5000],
        "asof_pitcher_fastball_rate": [.7, .2, None],
        "asof_pitcher_breaking_rate": [.2, .6, None],
        "asof_pitcher_offspeed_rate": [.1, .2, None],
        "asof_pitcher_prev1_game_success_rate": [.5, .6, None],
        "asof_pitcher_prev5_game_success_rate": [.51, .4, None],
        "pitcher_team_id": [1, 2, 3], "game_type": ["R", "R", "F"],
    })


def test_role_features_are_row_local_under_shuffle_and_subset():
    rows = fixture()
    full = role_features(rows).astype(str)
    order = [2, 0]
    part = role_features(rows.iloc[order]).astype(str)
    pd.testing.assert_frame_equal(part.reset_index(drop=True), full.iloc[order].reset_index(drop=True))

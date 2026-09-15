import numpy as np
import pandas as pd

from scripts.validate_hierarchical_eb import fit_lookups, prepare, transform


def fixture():
    return prepare(pd.DataFrame({
        "season": [2020, 2020, 2021, 2022],
        "pitcher_id": [1, 1, 2, 1], "batter_id": [10, 11, 10, 10],
        "pitcher_hand": ["R", "R", "L", "R"], "batter_hand": ["L", "R", "L", "L"],
        "balls_before": [0, 1, 0, 2], "strikes_before": [0, 1, 2, 2],
        "control_success": [1, 0, 1, 0],
    }))


def test_validation_target_is_not_used_and_rows_are_independent():
    data = fixture()
    lookup = fit_lookups(data[data.season < 2022])
    rows = pd.concat([data[data.season == 2022], data[data.season == 2022]], ignore_index=True)
    base = transform(rows.drop(columns="control_success"), lookup, 20)
    changed = rows.copy()
    changed["control_success"] = 1 - changed["control_success"]
    pd.testing.assert_frame_equal(base, transform(changed.drop(columns="control_success"), lookup, 20))
    shuffled = rows.sample(frac=1, random_state=3)
    got = transform(shuffled.drop(columns="control_success"), lookup, 20).sort_index()
    pd.testing.assert_frame_equal(base.sort_index(), got)
    pd.testing.assert_frame_equal(base.iloc[[0]].reset_index(drop=True),
                                  transform(rows.iloc[[0]].drop(columns="control_success"), lookup, 20).reset_index(drop=True))


def test_unseen_groups_shrink_exactly_to_parent():
    data = fixture()
    lookup = fit_lookups(data[data.season < 2022])
    row = data.iloc[[3]].drop(columns="control_success").copy()
    row["pitcher_id"], row["batter_id"] = 999, 888
    out = transform(row, lookup, 100).iloc[0]
    assert np.isclose(out.pitcher_rate, out.global_rate, rtol=0, atol=1e-15)
    assert np.isclose(out.batter_rate, out.global_rate, rtol=0, atol=1e-15)
    assert np.isclose(out.pitcher_batter_rate, out.global_rate, rtol=0, atol=1e-15)
    assert np.isfinite(out.to_numpy(float)).all()

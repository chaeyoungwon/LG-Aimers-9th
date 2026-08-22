import pandas as pd

from scripts.validate_two_strike_expert import slice_mask


def test_two_strike_slice_is_fixed_and_row_local():
    rows = pd.DataFrame({"balls_before": [0, 1, 2, 3, 2], "strikes_before": [2, 2, 2, 2, 1]})
    expected = [True, True, True, False, False]
    assert slice_mask(rows).tolist() == expected
    order = [3, 0, 4]
    assert slice_mask(rows.iloc[order]).tolist() == [expected[i] for i in order]

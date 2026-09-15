import numpy as np
import pandas as pd

from scripts.f_expert_runtime import apply_f_expert


def test_f_expert_only_changes_f_rows_and_is_order_independent():
    rows = pd.DataFrame({"game_type": ["R", "F", "F"]}, index=[10, 20, 30])
    pred = np.array([0.40, 0.60, 0.20])
    spec = {"weight": 0.25, "prior": 0.46, "segment": "F"}

    got = apply_f_expert(pred, rows, spec)

    assert got[0] == pred[0]
    assert np.isclose(got[1], 0.75 * 0.60 + 0.25 * 0.46)
    assert np.isclose(got[2], 0.75 * 0.20 + 0.25 * 0.46)
    order = [2, 0, 1]
    shuffled = apply_f_expert(pred[order], rows.iloc[order], spec)
    np.testing.assert_array_equal(shuffled, got[order])

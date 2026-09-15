import numpy as np
import pandas as pd

from scripts.validate_hard_example_objective import objective_weights


def test_weights_only_use_train_labels_and_frozen_inner_oof():
    y = np.array([0., 1., 0., 1., 1., 0.])
    oof = np.array([.2, .4, .8, .6, np.nan, np.nan])
    base = objective_weights(y, oof, "hard")
    changed_future = y.copy(); changed_future[4:] = 1-changed_future[4:]
    np.testing.assert_array_equal(base, objective_weights(changed_future, oof, "hard"))
    assert np.all(base[~np.isfinite(oof)] == 1.)


def test_uncertainty_weight_is_row_local_and_label_free():
    y = np.array([0., 1., 0.]); p = np.array([.1, .5, .9])
    first = objective_weights(y, p, "uncertainty_1p0")
    second = objective_weights(1-y, p, "uncertainty_1p0")
    np.testing.assert_array_equal(first, second)
    assert first[1] > first[0] and first[1] > first[2]

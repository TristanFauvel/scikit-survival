import os.path

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
import pytest

from sksurv.datasets import load_gbsg2
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    cook_ridker_test,
    dagostino_nam_test,
    greenwood_dagostino_nam_test,
)
from sksurv.preprocessing import OneHotEncoder
from sksurv.svm import FastSurvivalSVM

def test_hl_coxph():
    X, y = load_gbsg2()
    X.loc[:, "tgrade"] = X.loc[:, "tgrade"].map(len).astype(int)

    Xt = OneHotEncoder().fit_transform(X)

    est = CoxPHSurvivalAnalysis(ties="efron").fit(Xt, y)

    ck_res = cook_ridker_test(est, Xt, y)
    assert round(abs(ck_res['p_value'] - 0.30012243125428406), 5) == 0
    assert round(abs(ck_res['test_statistic'] - 9.522866759277584), 5) == 0

    dan_res = dagostino_nam_test(est, Xt, y)
    assert round(abs(dan_res['p_value'] - 0.39047238656286276), 5) == 0
    assert round(abs(dan_res['test_statistic'] - 9.522866759277584), 5) == 0

    gdan_res = greenwood_dagostino_nam_test(est, Xt, y)
    assert round(abs(gdan_res['p_value'] - 0.8702849971045289), 5) == 0
    assert round(abs(gdan_res['test_statistic'] - 4.567248742271165), 5) == 0


def test_hl_groups():
    X, y = load_gbsg2()
    X.loc[:, "tgrade"] = X.loc[:, "tgrade"].map(len).astype(int)

    Xt = OneHotEncoder().fit_transform(X)

    est = CoxPHSurvivalAnalysis(ties="efron").fit(Xt, y)

    with pytest.raises(ValueError,
                       match="The number of groups must be larger than 2"):
        cook_ridker_test(est, Xt, y, Q=2)

    with pytest.raises(ValueError,
                       match="The number of groups must be larger than 1"):
        dagostino_nam_test(est, Xt, y, Q=1)

    with pytest.raises(ValueError,
                       match="The number of groups must be larger than 1"):
        greenwood_dagostino_nam_test(est, Xt, y, Q=1)


def test_hl_time():
    X, y = load_gbsg2()
    X.loc[:, "tgrade"] = X.loc[:, "tgrade"].map(len).astype(int)

    Xt = OneHotEncoder().fit_transform(X)

    est = CoxPHSurvivalAnalysis(ties="efron").fit(Xt, y)

    with pytest.raises(ValueError,
                       match="Time must be between 0 and 1."):
        cook_ridker_test(est, Xt, y, time=1.1)

    with pytest.raises(ValueError,
                       match="Time must be between 0 and 1."):
        dagostino_nam_test(est, Xt, y, time=1.1)

    with pytest.raises(ValueError,
                       match="Time must be between 0 and 1."):
        greenwood_dagostino_nam_test(est, Xt, y, time=1.1)

    with pytest.raises(ValueError,
                       match="Time must be between 0 and 1."):
        cook_ridker_test(est, Xt, y, time=0)

    with pytest.raises(ValueError,
                       match="Time must be between 0 and 1."):
        dagostino_nam_test(est, Xt, y, time=0)

    with pytest.raises(ValueError,
                       match="Time must be between 0 and 1."):
        greenwood_dagostino_nam_test(est, Xt, y, time=0)


def test_hl_predict_survival():
    X, y = load_gbsg2()
    X.loc[:, "tgrade"] = X.loc[:, "tgrade"].map(len).astype(int)

    Xt = OneHotEncoder().fit_transform(X)

    with pytest.raises(
        AttributeError,
        match=r"FastSurvivalSVM\(\) object has no attribute 'predict_survival_function'"
    ):
        cook_ridker_test(FastSurvivalSVM(), Xt, y)

    with pytest.raises(
        AttributeError,
        match=r"FastSurvivalSVM\(\) object has no attribute 'predict_survival_function'"
    ):
        dagostino_nam_test(FastSurvivalSVM(), Xt, y)

    with pytest.raises(
        AttributeError,
        match=r"FastSurvivalSVM\(\) object has no attribute 'predict_survival_function'"
    ):
        greenwood_dagostino_nam_test(FastSurvivalSVM(), Xt, y)

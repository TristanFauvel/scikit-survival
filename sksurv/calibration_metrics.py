from .util import check_y_survival
import numpy as np
import pandas as pd
from .nonparametric import SurvivalFunctionEstimator
from scipy.stats import chi2 
import warnings
import numbers
from scipy import stats

def variance_kaplan_meier_prob_survival(survival_prob, ci, conf_type="greenwood", conf_level=0.95):
    """Variance of the Kaplan-Meier estimator of survival function. Computed using Greenwood's approximation
    """
    if conf_type not in {"greenwood"}:
        raise ValueError(f"conf_type must be None or a str among {{'greenwood'}}, but was {conf_type!r}")

    if not isinstance(conf_level, numbers.Real) or not np.isfinite(conf_level) or conf_level <= 0 or conf_level >= 1.0:
        raise ValueError(f"conf_level must be a float in the range (0.0, 1.0), but was {conf_level!r}")

    z = stats.norm.isf((1.0 - conf_level) / 2.0)
    var_survival_prob =    ((ci[1,:] - survival_prob)/z)**2
    return var_survival_prob



def cook_ridker_test(model, X, y, time=3 / 4, Q=10):
    """Cook-Ridker goodness-of-fit test

    The Cook-Ridker test is a translation of the Hosmer-Lemeshow goodness-of-fit test to survival analysis.
    It is the same as the D'Agostino-Nam test, the difference being that the number of degrees of freedom
    is Q-2 for the Cook-Ridker test.

    See [1]_ and [2]_ for further description.


    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        Data matrix.

    y : structured array, shape = (n_samples,)
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    time : float, optional, default : 3/4
        Time at which the test statistic is computed, in fraction of the maximum time.

    Q : int, optional, default: 10
        The number of groups. Data is divided into groups based on predicted probabilities.

    Returns
    -------
    statistic : float or array
        The calculated test statistic.

    pvalue : float or array
        The p-value.

    References
    ----------
    .. [1] Cook, N. R., & Ridker, P. M. (2009). Advances in Measuring the Effect of Individual Predictors
           of Cardiovascular Risk: The Role of Reclassification Measures. Annals of Internal
           Medicine, 150(11), 795-802.
    .. [2] Guffey D., Hosmer-Lemeshow goodness-of-fit test: Translations to the Cox Proportional Hazards Model (2013)
    """
    df = 2
    test_result = _hosmer_lemeshow_survival(model, X, y, df, time, Q, "CR")
    return test_result


def dagostino_nam_test(model, X, y, time=3 / 4, Q=10):
    """D'Agostino-Nam goodness-of-fit test

    The D'Agostino-Nam test is a translation of the Hosmer-Lemeshow goodness-of-fit test to survival analysis.
    It is the same as the Cook-Ridker test, the difference being that the number of degrees of freedom is Q-1 for the D'Agostino-Nam test.

    See [1]_ and [2]_ for further description.


    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        Data matrix.

    y : structured array, shape = (n_samples,)
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    time : float, optional, default : 3/4
        Time at which the test statistic is computed, in fraction of the maximum time.

    Q : int, optional, default: 10
        The number of groups. Data is divided into groups based on predicted probabilities.

    Returns
    -------
    statistic : float or array
        The calculated test statistic.

    pvalue : float or array
        The p-value.

    References
    ----------
    .. [1] D'Agostino, R., & Nam, B. (2004). Evaluation of the performance of survival analysis models:
           Discrimination and Calibration measure. In N. Balakrishnan, & C. Rao (Eds.), Handbook
           of Statistics, Survival Methods. Volume 23. (pp. 1-25). Amsterdam: Elsevier.
    .. [2] Guffey D., Hosmer-Lemeshow goodness-of-fit test: Translations to the Cox Proportional Hazards Model (2013)
    """

    df = 1
    test_result = _hosmer_lemeshow_survival(model, X, y, df, time, Q, "DN")
    return test_result


def greenwood_dagostino_nam_test(model, X, y, time=3 / 4, Q=10):
    """Greenwood-D'Agostino-Nam test

    The Greenwood-D'Agostino-Nam test is an improvement over the D'Agostino-Nam test with lower Type 1 error at higher censoring rates.

    Parameters
    ----------
    X: array-like, optional, shape = (n_samples, n_features)
    Data matrix.

    y: structured array, shape = (n_samples,)
    A structured array containing the binary event indicator
    as first field, and time of event or time of censoring as
    second field.

    time : float, optional, default : 3/4
        Time at which the test statistic is computed, in fraction of the maximum time.

    Q: int, optional, default: 10
    The number of groups. Data is divided into groups based on predicted probabilities.

    Returns
    -------
    statistic: float or array
    The calculated test statistic.

    pvalue: float or array
    The p-value.

    References
    ----------
    .. [1] Demler OV, Paynter np, Cook NR. Tests of calibration and goodness-of-fit in the survival setting.
           Stat Med. 2015 May 10;34(10):1659-80. doi: 10.1002/sim.6428. Epub 2015 Feb 11.
           PMID: 25684707; PMCID: PMC4555993.
    """

    df = 1
    test_result = _hosmer_lemeshow_survival(model, X, y, df, time, Q, "GDN")
    return test_result


def _check_hl_test_inputs(model, df, time, Q, test):
    if df < 1:
        raise ValueError("Degrees of freedom must be at least 1.")
    if Q < 1:
        raise ValueError("Number of groups must be at least 1.")
    if test not in ["CR", "DN", "GDN"]:
        raise ValueError("The test name should be either ''CR'', ''DN'', or ''GDN''")
    if time > 1 or time <= 0:
        raise ValueError("Time must be between 0 and 1.")
    if not hasattr(model, "predict_survival_function"):
        raise AttributeError("{!r} object has no attribute {!r}".format(model, "predict_survival_function"))


def _hosmer_lemeshow_survival(model, X, y, df, time, Q, test):
    """Translation of the Hosmer-Lemeshow goodness-of-fit test to survival analysis.

    See [1]_ and [2]_ for further description.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        Data matrix.

    y : structured array, shape = (n_samples,)
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    df : int,
        The number of degrees of freedom of the test is Q-df.

    time : float
        Time at which the test statistic is computed, in fraction of the maximum time.

    Q : int,
        The number of groups. Data is divided into groups based on predicted probabilities.

    test : str,
        Name of the test, either "CR", "DN", or "GDN"

    Returns
    -------
    statistic : float or array
        The calculated test statistic.

    pvalue : float or array
        The p-value.

    References
    ----------
    .. [1] Guffey D., Hosmer-Lemeshow goodness-of-fit test: Translations to the Cox Proportional Hazards Model (2013)
           (http://hdl.handle.net/1773/22648)
    .. [2] Demler OV, Paynter np, Cook NR. Tests of calibration and goodness-of-fit in the survival setting.
           Stat Med. 2015 May 10;34(10):1659-80. doi: 10.1002/sim.6428. Epub 2015 Feb 11.
           PMID: 25684707; PMCID: PMC4555993.
    """
    _check_hl_test_inputs(model, df, time, Q, test)

    event, event_time = check_y_survival(y)

    time = time * event_time.max()

    predictions = model.predict_survival_function(X)
    pred_surv_prob = np.row_stack([fn(time) for fn in predictions])

    # Split the data into Q groups based on the risk scores (for example into deciles).
    group_data = True
    min_group_size = 5
    while group_data:
        categories = pd.cut(
            pred_surv_prob[:, -1],
            np.percentile(pred_surv_prob[:, -1], np.linspace(0, 100, Q + 1)),
            labels=False,
            include_lowest=True,
            duplicates="drop",
        )
        if any(np.array([sum(categories == i) for i in range(Q)]) < min_group_size):
            warnings.warn("Some group sizes are less than 5, lowering the number of groups", UserWarning)
            Q = Q - 1
        else:
            group_data = False

    Q = categories.max() + 1

    if Q <= df:
        raise ValueError(f"The number of groups must be larger than {df}")

    prob = np.zeros(Q)
    N = np.zeros(Q)
    KM = np.zeros(Q)
    ci_KM = np.zeros((2, Q))
    for i in range(Q):
        surv = SurvivalFunctionEstimator(conf_level=0.95, conf_type="greenwood")
        surv.fit(y[categories == i])

        max_t_i = event_time[categories == i].max()
        # 1 - KM is the Kaplan-Meyer failure probabiliy in the g-th Q-quantile at time time
        if time > max_t_i:
            KM[i], ci_temp = surv.predict_proba([max_t_i], return_conf_int=True)
        else:
            KM[i], ci_temp = surv.predict_proba([time], return_conf_int=True)
        ci_KM[:, i] = ci_temp.flatten()

        N[i] = np.sum(categories == i)

        # Compute the mean predicted probability of failure for subjects ion the g-th Q-quantile.
        prob[i] = (1 - pred_surv_prob[categories == i]).sum(axis=0) / N[i]

    # Determine how well the model fits the data in each decile, using Chi-squared statistics
    if test in ["CR", "DN"]:
        chisq_value = np.sum(
            np.divide(
                (N * (1 - KM) - N * prob) ** 2,
                (N * prob * (1 - prob)),
                out=np.zeros(Q, dtype=float),
                where=(1 - KM) != prob,
            ),
            axis=0,
        )
    elif test == "GDN":
        # Variance of the Kaplan-Meier estimator of the survival function
        Var_KM = variance_kaplan_meier_prob_survival(KM, ci_KM, conf_level=0.95, conf_type = "greenwood") 

        chisq_value = np.sum(
            np.divide(
                (1 - KM - prob) ** 2,
                Var_KM,
                out=np.zeros(Q, dtype=float),
                where=Var_KM != 0,
            ),
            axis=0,
        )

    obsevents = N * (1 - KM)  # "Observed" number of failures (based on the Kaplan-Meier estimator)
    expevents = N * prob  # Predicted number of failures

    # Summarize informations for each group.
    groupings = pd.DataFrame(
        {
            "Groups": np.arange(Q),
            "N": N.reshape(
                -1,
            ),
            "Observations": obsevents.reshape(
                -1,
            ),
            "Expectations": expevents.reshape(
                -1,
            ),
        }
    )

    pvalue = 1 - chi2.cdf(chisq_value, Q - df)

    test_result = {
        "test_statistic": chisq_value,
        "p_value": pvalue,
        "groupings": groupings,
    }

    return test_result

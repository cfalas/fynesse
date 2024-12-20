# This file contains code for suporting addressing questions in the data
from . import access

"""# Here are some of the imports we might expect
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

import statsmodels.api as sm
import numpy as np
import numpy.typing as npt
import pandas as pd

import scipy.stats as stats
from sqlalchemy.sql import func, select, text
import matplotlib.pyplot as plt

"""Address a particular question that arises from the data"""


def random_voters_design_matrix(sample: pd.DataFrame):
    return np.concatenate(
        (
            *[(sample.elections_vote == i).to_numpy().reshape(-1, 1) for i in range(4)],
            *[
                (sample.country_of_birth == i).to_numpy().reshape(-1, 1)
                for i in range(2)
            ],
            *[(sample.nssec == i).to_numpy().reshape(-1, 1) for i in range(8)],
            *[(sample.age == i).to_numpy().reshape(-1, 1) for i in range(3)],
            np.ones((len(sample), 1)),
        ),
        axis=1,
    )


def voting_areas_design_matrix(random_voters: access.datasets.Dataset):
    all_cols = random_voters.query(
        select(
            func.avg(random_voters.c.referendum_vote),
            *[func.avg(random_voters.c.elections_vote == i) for i in range(4)],
            *[func.avg(random_voters.c.country_of_birth == i) for i in range(2)],
            *[func.avg(random_voters.c.nssec == i) for i in range(8)],
            *[func.avg(random_voters.c.age == i) for i in range(3)],
            text("1"),
        ).group_by(random_voters.c.referendum_results_Area_Code)
    ).to_numpy()
    all_cols = all_cols[~np.isnan(all_cols).any(axis=1)]
    return all_cols[:, 0], all_cols[:, 1:]


def random_voters_logistic_regression(
    design: npt.NDArray[np.float64], y: npt.ArrayLike
):
    model = sm.Logit(y, design)
    result = model.fit()
    return result


def random_voters_logistic_regression_regularised(
    design: npt.NDArray[np.float64], y: npt.ArrayLike, alpha: float, L1_wt: float
):
    model = sm.GLM(y, design, family=sm.families.Binomial())
    result = model.fit_regularized(alpha=alpha, L1_wt=L1_wt)
    return result


def random_voters_accuracy(
    results: sm.regression.linear_model.RegressionResults, sample: pd.DataFrame
):
    X = random_voters_design_matrix(sample)
    y = sample.referendum_vote.to_numpy()
    y_pred = results.predict(X) > 0.5
    accuracy = np.mean(y == y_pred)

    global_av = sample.referendum_vote.mean()
    p_value = 1 - stats.binom.cdf(
        int(accuracy * len(y)), len(y), global_av
    )  # one-sided p-value

    if p_value < 0.05:
        print(
            f"The model is significantly different from choosing at random (p = {p_value})"
        )

    return accuracy


def correlation(a: npt.ArrayLike, b: npt.ArrayLike):
    return stats.pearsonr(a, b)


def plot_correlation(a: npt.ArrayLike, b: npt.ArrayLike):
    plt.scatter(a, b)
    plt.xlabel("Actual % Leave")
    plt.ylabel("Predicted % Leave")
    plt.show()
    print(f"Correlation: {correlation(a, b)}")

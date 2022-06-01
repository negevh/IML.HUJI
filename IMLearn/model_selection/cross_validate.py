from __future__ import annotations

import math
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    X_y = np.c_[X, y]
    np.random.shuffle(X_y)
    n_samples = float(X.shape[0])
    n_subset = round(n_samples / 5)
    end_ind = 0
    train_scores = []
    valid_scores = []
    for i in range(cv):
        start_ind = end_ind
        end_ind = start_ind + n_subset if i < cv - 1 else int(n_samples)
        X_train = np.delete(X_y[:, :-1], np.s_[start_ind:end_ind], 0)
        y_train = np.delete(X_y[:, -1], np.s_[start_ind:end_ind])
        X_valid = X_y[start_ind:end_ind, :-1]
        y_valid = X_y[start_ind:end_ind, -1]
        estimator.fit(X_train, y_train)
        train_scores.append(scoring(estimator.predict(X_train), y_train))
        valid_scores.append(scoring(estimator.predict(X_valid), y_valid))

    return np.average(train_scores), np.average(valid_scores)

from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product

from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self):
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_features = X.shape[1]
        best_thr_err = 1
        best_thr = None
        best_sign = None
        best_j = None
        for j in range(n_features):
            for cur_sign in [1, -1]:
                cur_thr, cur_thr_err = self._find_threshold(X[:, j], y, cur_sign)
                if cur_thr_err < best_thr_err:
                    best_thr_err = cur_thr_err
                    best_thr = cur_thr
                    best_sign = cur_sign
                    best_j = j
        self.threshold_ = best_thr
        self.sign_ = best_sign
        self.j_ = best_j

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.array([-self.sign_ if x < self.threshold_ else self.sign_ for x in X[:, self.j_]])

    @staticmethod
    def _find_threshold(values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        n_samples = len(values)
        values_and_labels = sorted(list(zip(values, labels)), key=lambda x: x[0])
        sorted_labels = [x[1] for x in values_and_labels]
        neg_err = np.cumsum((np.sign(sorted_labels) != -sign) * np.abs(sorted_labels))
        pos_err = np.flipud(np.cumsum(np.flipud((np.sign(sorted_labels) != sign) * np.abs(sorted_labels))))
        thr_err = np.insert(neg_err, 0, 0) + np.insert(pos_err, n_samples, 0)
        index = np.argmin(thr_err)
        max_err = np.max(thr_err)
        while index < n_samples - 1 and values_and_labels[index][0] == values_and_labels[index + 1][0]:
            thr_err[index] = max_err
            index = np.argmin(thr_err)
        if index == n_samples:
            best_thr = values_and_labels[-1][0] + 1
        else:
            best_thr = values_and_labels[index][0]
        return best_thr, thr_err[index]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.predict(X))

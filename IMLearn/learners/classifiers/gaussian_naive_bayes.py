from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, n = np.unique(y, return_counts=True)
        n_classes = self.classes_.shape[0]
        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.pi_ = n / n_samples
        for i in range(n_classes):
            row = (1 / n[i]) * np.sum(X, where=(np.c_[y, y] == self.classes_[i]), axis=0)
            if i == 0:
                self.mu_ = row
            else:
                self.mu_ = np.vstack([self.mu_, row])
        self.vars_ = np.zeros((n_classes, n_features))
        for i in range(n_classes):
            row = (1 / n[i]) * np.sum((X - self.mu_[i]) ** 2, where=(np.c_[y, y] == self.classes_[i]), axis=0)
            if i == 0:
                self.vars_ = row
            else:
                self.vars_ = np.vstack([self.vars_, row])

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
        """
        return np.argmax(self.likelihood(X), axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        n_classes = self.classes_.shape[0]
        likelihood = None
        for i in range(n_classes):
            col = self.pi_[i] * np.prod(np.exp(-((X - self.mu_[i]) ** 2) / (2 * self.vars_[i])) / np.sqrt(
                2 * np.pi * self.vars_[i]), axis=1)
            if i == 0:
                likelihood = col
            else:
                likelihood = np.c_[likelihood, col]
        return likelihood

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

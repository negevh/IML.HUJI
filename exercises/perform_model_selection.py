from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.linspace(-1.2, 2, n_samples)
    y_noiseless = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)
    y = y_noiseless + np.random.normal(0, noise, n_samples)
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), .67)
    train_X = train_X.to_numpy().flatten()
    train_y = train_y.to_numpy().flatten()
    test_X = test_X.to_numpy().flatten()
    test_y = test_y.to_numpy().flatten()
    go.Figure([
        go.Scatter(x=X, y=y_noiseless, mode='markers', name='True Model (Noiseless)'),
        go.Scatter(x=train_X, y=train_y, mode='markers', name='Train samples'),
        go.Scatter(x=test_X, y=test_y, mode='markers', name='Test samples')]).update_layout(
        title=f'Model and Samples. Samples amount: {n_samples}, Noise level: {noise}', xaxis=dict(title='Samples'),
        yaxis=dict(title='Responses')).show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    avg_train_errs = []
    avg_valid_errs = []
    for k in range(11):
        avg_train_err, avg_valid_err = cross_validate(PolynomialFitting(k), train_X, train_y, mean_square_error)
        avg_train_errs.append(avg_train_err)
        avg_valid_errs.append(avg_valid_err)
    go.Figure([
        go.Scatter(x=list(range(11)), y=avg_train_errs, mode='markers + lines', name='Average Train Error'),
        go.Scatter(x=list(range(11)), y=avg_valid_errs, mode='markers + lines', name='Average Validation Error')]).\
        update_layout(title=f'Average Errors For Polynomial Fitting. Samples amount: {n_samples}, Noise level: {noise}',
                      xaxis=dict(title='Polynomial Degree')).show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(avg_valid_errs)
    poly_fit = PolynomialFitting(best_k)
    poly_fit.fit(train_X, train_y)
    test_error = poly_fit.loss(test_X, test_y)
    print(f'Samples amount: {n_samples}, Noise level: {noise}\n'
          f'--------------------------------------------------\n'
          f'Best polynomial degree: {best_k}\n'
          f'Test error: {round(test_error, 2)}\n')


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = load_diabetes(return_X_y=True)
    train_X = X[:n_samples, :]
    train_y = y[:n_samples]
    test_X = X[n_samples:, :]
    test_y = y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    models = [RidgeRegression, Lasso]
    model_names = ['Ridge Regression', 'Lasso']
    model_best_lams =[]
    for i in range(len(models)):
        avg_train_errs = []
        avg_valid_errs = []
        lams = np.linspace(0, 1, num=n_evaluations)
        for lam in lams:
            avg_train_err, avg_valid_err = cross_validate(models[i](lam), train_X, train_y, mean_square_error)
            avg_train_errs.append(avg_train_err)
            avg_valid_errs.append(avg_valid_err)
        go.Figure([
            go.Scatter(x=lams, y=avg_train_errs, mode='markers + lines',
                       name='Average Train Error'),
            go.Scatter(x=lams, y=avg_valid_errs, mode='markers + lines',
                       name='Average Validation Error')]). \
            update_layout(title=f'Average Errors For {model_names[i]}',
                          xaxis=dict(title='Regularization Parameter')).show()
        model_best_lams.append(lams[np.argmin(avg_valid_errs)])

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    for i in range(len(models)):
        print(f'Model: {model_names[i]}\n'
              f'------------------------------------\n'
              f'Best regularization parameter: {model_best_lams[i]}\n')
    for i in range(len(models)):
        model = models[i](model_best_lams[i])
        model.fit(train_X, train_y)
        print(f'Model: {model_names[i]}\n'
              f'------------------------------------\n'
              f'Test error over best regularization parameter: {mean_square_error(model.predict(test_X), test_y)}\n')
    lin_reg = LinearRegression()
    lin_reg.fit(train_X, train_y)
    print(f'Model: Linear Regression\n'
          f'------------------------------------\n'
          f'Test error: {lin_reg.loss(test_X, test_y)}\n')


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()

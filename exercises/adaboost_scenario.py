import numpy as np
from typing import Tuple
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metalearners import AdaBoost
from IMLearn.metrics import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    train_loss = np.zeros(n_learners)
    test_loss = np.zeros(n_learners)
    for i in range(n_learners):
        train_loss[i] = adaboost.partial_loss(train_X, train_y, i + 1)
        test_loss[i] = adaboost.partial_loss(test_X, test_y, i + 1)
    go.Figure([
        go.Scatter(x=np.arange(n_learners) + 1, y=train_loss, mode='markers + lines', name='Train Error'),
        go.Scatter(x=np.arange(n_learners) + 1, y=test_loss, mode='markers + lines', name='Test Error')]) \
        .update_layout(title=f'Train and test errors of AdaBoost. Noise level = {noise}',
                       xaxis=dict(title='Number of Learners')).show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    symbols = np.array(["circle", "x"])
    figs = []
    for i, t in enumerate(T):
        fig = go.Figure([decision_surface(lambda X: adaboost.partial_predict(X, t), lims[0], lims[1], showscale=False),
                         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                    marker=dict(color=test_y.astype(int), symbol=symbols[(test_y == 1).astype(int)],
                                                colorscale=[custom[0], custom[-1]],
                                                line=dict(color="black", width=1)))])
        fig.update_layout(title=f'Decision Surface - {t} learners. Noise level = {noise}')
        fig.show()
        figs.append(fig)

    # Question 3: Decision surface of best performing ensemble
    lowest_err_index = np.argmin(test_loss[np.array(T) - 1])
    ensemble_accuracy = accuracy(test_y, adaboost.partial_predict(test_X, T[lowest_err_index]))
    figs[lowest_err_index].update_layout(title=f'Decision Surface - {T[lowest_err_index]} learners. '
                                               f'Noise level = {noise}. Accuracy: {ensemble_accuracy}').show()

    # Question 4: Decision surface with weighted samples
    fig = go.Figure([decision_surface(lambda X: adaboost.predict(X), lims[0], lims[1], showscale=False),
                     go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                marker=dict(color=train_y.astype(int), symbol=symbols[(train_y == 1).astype(int)],
                                            colorscale=[custom[0], custom[-1]],
                                            size=(adaboost.D_ / np.max(adaboost.D_)) * 70,
                                            line=dict(color="black", width=1)))])
    fig.update_layout(title=f'Decision surface with weighted samples. Noise level = {noise}')
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)

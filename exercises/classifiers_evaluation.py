from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from math import atan2, pi

from utils import class_colors, decision_surface, class_symbols

pio.templates.default = "simple_white"

DATASETS_PATH = 'C:\\Users\\user\\Documents\\university\\IML\\IML.HUJI\\datasets\\'


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X_train, y_train = load_dataset(DATASETS_PATH + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def perceptron_callback(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit._loss(X_train, y_train))

        perceptron = Perceptron(callback=perceptron_callback)
        perceptron.fit(X_train, y_train)

        # Plot figure
        fig = go.Figure([go.Scatter(x=np.arange(len(losses)), y=losses, mode='markers+lines')],
                        layout=go.Layout(title='fit progression of the Perceptron algorithm over {} dataset'.format(n),
                                         xaxis_title='iteration', yaxis_title='loss'))
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X_train, y_train = load_dataset(DATASETS_PATH + f)

        # Fit models and predict over training set
        gnb = GaussianNaiveBayes()
        gnb.fit(X_train, y_train)
        y_gnb = gnb.predict(X_train)

        lda = LDA()
        lda.fit(X_train, y_train)
        y_lda = lda.predict(X_train)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        models = [gnb, lda]
        model_names = ["Gaussian Naive Bayes", "LDA"]
        predicts = [y_gnb, y_lda]
        accuracies = [accuracy(y_train, y_pred) for y_pred in predicts]
        fig = make_subplots(rows=1, cols=2, subplot_titles=["{} over {}. Accuracy: {}".format(m, f[:-4], ac) for
                                                            m, ac in zip(model_names, accuracies)],
                            horizontal_spacing=0.01, vertical_spacing=.03)

        # Add traces for data-points setting symbols and colors
        lims = np.array([X_train.min(axis=0), X_train.max(axis=0)]).T + np.array([-.4, .4])
        for i, m in enumerate(models):
            fig.add_traces([decision_surface(m.predict, lims[0], lims[1], showscale=False),
                            go.Scatter(x=X_train[:, 0], y=X_train[:, 1], mode="markers", showlegend=False,
                                       marker=dict(color=predicts[i], symbol=class_symbols[y_train],
                                                   colorscale=class_colors(m.classes_.size),
                                                   line=dict(color="black", width=1)))],
                           rows=1, cols=i+1)

        # Add `X` dots specifying fitted Gaussians' means
        for i, m in enumerate(models):
            fig.add_traces([go.Scatter(x=m.mu_[:, 0], y=m.mu_[:, 1], mode="markers", showlegend=False,
                                       fillcolor="black", marker=dict(symbol='x', size=20, color="black"))],
                           rows=1, cols=i+1)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i, m in enumerate(models):
            if m == lda:
                cov = m.cov_
            for j in range(m.classes_.size):
                if m == gnb:
                    cov = np.diag(m.vars_[j])
                fig.add_traces([get_ellipse(m.mu_[j, :], cov)], rows=1, cols=i+1)

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()

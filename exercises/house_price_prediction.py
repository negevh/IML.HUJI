import numpy

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

HOUSE_PRICES_CSV_FILENAME = 'C:\\Users\\user\\Documents\\university\\IML\\IML.HUJI\\datasets\\house_prices.csv'


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)

    df.drop(df[df.price < 0].index, inplace=True)
    df.drop(df[df.bedrooms < 0].index, inplace=True)
    df.drop(df[df.bathrooms < 0].index, inplace=True)
    df.drop(df[df.sqft_lot < 0].index, inplace=True)
    df.drop(df[df.sqft_lot < df.sqft_living].index, inplace=True)
    df.drop(df[df.floors < 0].index, inplace=True)
    df.drop(df[df.waterfront < 0].index, inplace=True)
    df.drop(df[df.waterfront > 1].index, inplace=True)
    df.drop(df[df.view < 0].index, inplace=True)
    df.drop(df[df.view > 1].index, inplace=True)
    df.drop(df[df.condition < 0].index, inplace=True)
    df.drop(df[df.grade < 0].index, inplace=True)
    df.drop(df[df.sqft_lot < df.sqft_above].index, inplace=True)
    df.drop(df[df.sqft_lot < df.sqft_basement].index, inplace=True)
    df.drop(df[df.yr_built < 1500].index, inplace=True)
    df.drop(df[df.yr_renovated < 0].index, inplace=True)
    df.drop(df[df.zipcode < 98000].index, inplace=True)
    df.drop(df[df.zipcode > 98999].index, inplace=True)
    df.drop(df[df.lat < 40].index, inplace=True)
    df.drop(df[df.lat > 50].index, inplace=True)
    df.drop(df[df.long < -130].index, inplace=True)
    df.drop(df[df.long > -120].index, inplace=True)
    df.drop(df[df.sqft_lot15 < 0].index, inplace=True)
    df.drop(df[df.sqft_lot15 < df.sqft_living15].index, inplace=True)

    df['date'] = df['date'].apply(process_date)
    df.drop(df[df.date < 19000000].index, inplace=True)

    df = df.dropna()

    prices = df[['price']].squeeze()

    dummies_zip = pd.get_dummies(df.zipcode)
    df.drop(columns=['id', 'price', 'zipcode'], inplace=True)
    df = df.join(dummies_zip)

    return df, prices


def process_date(date):
    date = str(date)[:str(date).find('T')]
    if date.isdigit():
        return float(date)
    return 0


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    y = y.to_numpy(dtype=float)
    mu_y = np.mean(y)
    sigma_y = np.std(y)
    for (col_name, col_val) in X.iteritems():
        col_val = col_val.to_numpy(dtype=float)
        pearson_correlation = np.mean((col_val - np.mean(col_val)) * (y - mu_y)) / (sigma_y * np.std(col_val))
        fig = go.Figure([go.Scatter(x=col_val, y=y, mode='markers', marker=dict(color="black"))],
                        layout=go.Layout(title='Feature: {}. Pearson Correlation: {}'.format(col_name,
                                                                                             pearson_correlation),
                                         xaxis={"title": "x - {}".format(col_name)},
                                         yaxis={"title": "y - Response"}))
        fig.write_image('{}/{}.png'.format(output_path, col_name))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    dataset, responses = load_data(HOUSE_PRICES_CSV_FILENAME)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(dataset, responses, 'C:\\Users\\user\\Documents\\university\\IML\\ex2\\correlations')

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(dataset, responses, .75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    estimator = LinearRegression()
    x = numpy.arange(10, 101)
    train_X_y = train_X.assign(price=train_y)
    test_X = test_X.to_numpy()
    test_y = test_y.to_numpy()
    loss = np.zeros((91, 10))
    for p in x:
        for i in range(10):
            sampled_X_y = train_X_y.sample(frac=(.01 * p))
            sampled_y = (sampled_X_y[['price']].squeeze()).to_numpy()
            sampled_X = (sampled_X_y.drop(columns=['price'])).to_numpy()
            estimator.fit(sampled_X, sampled_y)
            loss[p - 10][i] = estimator.loss(test_X, test_y)
    loss_mean = loss.mean(axis=1)
    loss_std = loss.std(axis=1)
    fig = go.Figure([go.Scatter(x=x, y=loss_mean, mode='lines+markers', marker=dict(color="blue")),
                     go.Scatter(x=x, y=loss_mean - 2 * loss_std, fill=None, mode="lines", line=dict(color="lightgrey"),
                                showlegend=False),
                     go.Scatter(x=x, y=loss_mean + 2 * loss_std, fill='tonexty', mode="lines",
                                line=dict(color="lightgrey"), showlegend=False)],
                    layout=go.Layout(title='Mean loss',
                                     xaxis={"title": "x - p% of the sample"},
                                     yaxis={"title": "y - mean loss"}))
    fig.show()



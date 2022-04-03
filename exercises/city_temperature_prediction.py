import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

CITY_TEMPERATURE_CSV_FILENAME = 'C:\\Users\\user\\Documents\\university\\IML\\IML.HUJI\\datasets\\City_Temperature.csv'


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df = df.dropna()
    df['Date'] = df['Date'].apply(lambda date: date.timetuple().tm_yday)
    df.rename(columns={'Date': 'DayOfYear'}, inplace=True)
    df.drop(df[df.Year < 1900].index, inplace=True)
    df.drop(df[df.Year > 2100].index, inplace=True)
    df.drop(df[df.Month < 1].index, inplace=True)
    df.drop(df[df.Month > 12].index, inplace=True)
    df.drop(df[df.Day < 1].index, inplace=True)
    df.drop(df[df.Day > 31].index, inplace=True)
    df.drop(df[df.Temp < -50].index, inplace=True)
    df.drop(df[df.Temp > 50].index, inplace=True)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data(CITY_TEMPERATURE_CSV_FILENAME)

    # Question 2 - Exploring data for specific country
    israel_data = data[data['Country'] == 'Israel']
    israel_data = israel_data.astype({'Year': str})
    fig = px.scatter(israel_data, x='DayOfYear', y='Temp', color='Year')
    fig.show()
    israel_data_months_std = israel_data.groupby('Month').agg('std')
    fig = px.bar(israel_data_months_std, x=israel_data_months_std.index, y='Temp',
                 labels={'israel_data_months_std.index': 'Month', 'Temp': 'Standard deviation of temperature'})
    fig.show()

    # Question 3 - Exploring differences between countries
    grouped_data = data.groupby(['Country', 'Month']).Temp.agg(['mean', 'std'])
    fig = px.line(grouped_data, x=grouped_data.index.get_level_values("Month"), y='mean',  error_y='std',
                  color=grouped_data.index.get_level_values("Country"),
                  labels={'x': 'Month', 'mean': 'Average temperature', 'color': 'Country'})
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    responses = israel_data[['Temp']].squeeze()
    dataset = israel_data[['DayOfYear']]
    train_X, train_y, test_X, test_y = split_train_test(dataset, responses, .75)
    train_X = train_X.to_numpy()
    train_y = train_y.to_numpy()
    test_X = test_X.to_numpy()
    test_y = test_y.to_numpy()
    ks = np.arange(1, 11)
    loss = np.zeros(10)
    for k in ks:
        estimator = PolynomialFitting(k)
        estimator.fit(train_X, train_y)
        loss[k - 1] = round(estimator.loss(test_X, test_y), 2)
        print('Loss for degree {}: {}\n'.format(k, loss[k - 1]))
    res = pd.DataFrame({'k': ks, 'Loss': loss})
    fig = px.bar(res, x='k', y='Loss',
                 labels={'k': 'k - degree of polynomial fitting'})
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    estimator = PolynomialFitting(4)
    estimator.fit(train_X, train_y)
    countries = ['Israel', 'Jordan', 'South Africa', 'The Netherlands']
    loss = np.zeros(4)
    test_Xs = [test_X]
    test_ys = [test_y]
    for i in range(1, 4):
        test_country = data[data['Country'] == countries[i]]
        test_Xs.append(test_country[['DayOfYear']].to_numpy())
        test_ys.append(test_country[['Temp']].squeeze().to_numpy())
    for i in range(4):
        loss[i] = round(estimator.loss(test_Xs[i], test_ys[i]), 2)
    res = pd.DataFrame({'Country': countries, 'Loss': loss})
    fig = px.bar(res, x='Country', y='Loss')
    fig.show()

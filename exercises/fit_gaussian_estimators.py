from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, size=1000)
    gaussian = UnivariateGaussian()
    gaussian.fit(samples)
    print("\n({}, {})\n".format(gaussian.mu_, gaussian.var_))

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.arange(10, 1001, 10)

    estimated_mean = []
    for m in ms:
        X = samples[:m + 1]
        gaussian.fit(X)
        estimated_mean.append(gaussian.mu_)

    fig = go.Figure([go.Scatter(x=ms, y=estimated_mean, mode='markers+lines', name=r'$\widehat\mu$'),
                     go.Scatter(x=ms, y=[10] * len(ms), mode='lines', name=r'$\mu$')],
                    layout=go.Layout(title=r"$\text{Estimation of Expectation As Function Of Number Of Samples}$",
                    xaxis_title="$m\\text{ - number of samples}$",
                    yaxis_title="r$\hat\mu$",
                    height=300))
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    gaussian.fit(samples)
    pdfs = gaussian.pdf(np.linspace(5, 14, 1000))
    fig = go.Figure([go.Scatter(x=np.linspace(5, 14, 1000), y=pdfs, mode='markers', marker=dict(size=3), name='PDF'),
                     go.Scatter(x=samples, y=[0] * 1000, mode='markers', marker=dict(size=3), name='samples')],
                    layout=go.Layout(title=r"$\text{Empirical PDF of fitted model}$",
                    xaxis_title='samples',
                    yaxis_title='PDF',
                    height=300))
    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mu, sigma, 1000)
    gaussian = MultivariateGaussian()
    gaussian.fit(samples)
    print('\n{}\n{}\n'.format(gaussian.mu_, gaussian.cov_))

    # Question 5 - Likelihood evaluation
    log_likelihood = np.zeros((200, 200))
    f1 = f3 = np.linspace(-10, 10, 200)
    for i in range(200):
        for j in range(200):
            mu = np.array([f1[i], 0, f3[j], 0])
            log_likelihood[i][j] = gaussian.log_likelihood(mu, sigma, samples)
    fig = go.Figure(go.Heatmap(x=f1, y=f3, z=log_likelihood), layout=go.Layout(
        title='Likelihood evaluation', xaxis_title='f3', yaxis_title='f1'))
    fig.show()

    # Question 6 - Maximum likelihood
    argmax = int(log_likelihood.argmax())
    print('\nf1: {}\nf3: {}\n'.format(f1[int(argmax / 200)], f3[(argmax % 200)]))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

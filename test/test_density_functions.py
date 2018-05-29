import numpy as np
from scalarmix import BivariateGaussian

def single_gaussian_densities_explicit(X, mean, variance):
    n_rows, n_cols = X.shape
    assert mean.shape == (n_rows,)
    assert variance.shape == (n_rows,)
    diff = (X - mean[:, np.newaxis])
    diff *= diff
    z_score = diff / variance[:, np.newaxis]
    normalizer = np.sqrt(2 * np.pi * variance)
    unnormalized_likelihoods = np.exp(-0.5 * z_score)
    return unnormalized_likelihoods / normalizer[:, np.newaxis]


def test_single_gaussian_densities():
    mu = np.array([1.0])
    variance = np.array([0.5])
    X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    print(mu.shape)
    print(variance.shape)
    print(X.shape)
    bivariate_gaussian = BivariateGaussian()

    densities = bivariate_gaussian.single_gaussian_densities(X, mu, variance)
    assert np.allclose(
        single_gaussian_densities_explicit(X, mu, variance),
        densities)

    # first entry is same as mean so should be highest
    assert (densities[0] > densities[1:]).all()
    # last entry is furthest from mean so should be lower
    assert (densities[-1] < densities[:-1]).all()

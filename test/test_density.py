import numpy as np
from sbgm import BivariateGaussian

def test_density():
    mu = np.array([1.0])
    stds = np.array([0.5])
    X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    print(mu.shape)
    print(stds.shape)
    print(X.shape)

    bivariate_gaussian = BivariateGaussian()
    assert np.allclose(
        bivariate_gaussian.single_gaussian_densities_explicit(X, mu, stds),
        bivariate_gaussian.single_gaussian_densities(X, mu, stds))

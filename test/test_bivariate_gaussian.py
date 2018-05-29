import numpy as np
from numpy.testing import assert_almost_equal

# from nose.tools import eq_

from scalarmix import BivariateGaussian

np.random.seed(0)

def generate_data(
        num_rows=10,
        num_points=10,
        mean_scale=100,
        variance_scale=10):
    num_rows = 10
    num_points = 100000

    means = np.random.uniform(-1, 1, num_rows * 2).reshape((-1, 2)) * mean_scale
    variances = np.random.uniform(1, 2, num_rows * 2).reshape((-1, 2)) * variance_scale

    assignments = np.random.randint(0, 2, (len(means), num_points))
    samples = np.array([
        [
            np.random.normal(
                means[row, assignments[row, point]],
                np.sqrt(variances[row, assignments[row, point]]))
            for point in range(num_points)
        ]
        for row in range(len(means))])
    return samples, means, variances, assignments


def test_simple():
    samples, means, variances, assignments = generate_data(
        num_rows=10,
        num_points=100,
        mean_scale=100,
        variance_scale=1)
    print("means", means.shape)
    print("variances", variances.shape)
    print("samples", samples.shape)

    assert not np.isnan(samples).any()

    m = BivariateGaussian()
    m.fit(samples)

    # Attempt to get identifiability by comparing min and max of the two component means.
    assert_almost_equal(m.mean_.min(1), means.min(1), decimal=1)
    assert_almost_equal(m.mean_.max(1), means.max(1), decimal=1)
    # Todo: we should check stds and assignments (weights) too.


def test_larger_means():

    samples, means, variances, assignments = generate_data(
        num_rows=10,
        num_points=10000,
        mean_scale=1e6,
        variance_scale=1e3)

    print("means", means[:2])
    print("stds", variances[:2])

    assert not np.isnan(samples).any()

    m = BivariateGaussian()
    m.fit(samples)

    # Attempt to get identifiability by comparing min and max of the two component means.
    assert_almost_equal(m.mean_.min(1), means.min(1), decimal=0)
    assert_almost_equal(m.mean_.max(1), means.max(1), decimal=0)

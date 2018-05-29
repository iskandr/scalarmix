import numpy as np
from numpy.testing import assert_almost_equal

# from nose.tools import eq_

import sbgm

np.random.seed(0)


def close_enough(true_param, estimated_param, tolerance=0.001):
    return np.abs(true_param - estimated_param) / estimated_param < tolerance


def test_simple():
    num_rows = 10
    num_points = 100000

    means = np.random.uniform(-100, 100, num_rows * 2).reshape((-1, 2))
    stds = np.random.uniform(1, 2, num_rows * 2).reshape((-1, 2))

    assignments = np.random.randint(0, 2, (len(means), num_points))
    samples = np.array([
        [
            np.random.normal(
                means[row, assignments[row, point]],
                stds[row, assignments[row, point]])
            for point in range(num_points)
        ]
        for row in range(len(means))])

    print("means", means.shape)
    print("stds", stds.shape)
    print("samples", samples.shape)

    assert not np.isnan(samples).any()

    m = sbgm.BivariateGaussian()
    m.fit(samples)

    # Attempt to get identifiability by comparing min and max of the two component means.
    assert_almost_equal(m.mu_.min(1), means.min(1), decimal=1)
    assert_almost_equal(m.mu_.max(1), means.max(1), decimal=1)
    # Todo: we should check stds and assignments (weights) too.


def test_larger_means():
    num_rows = 10
    num_points = 10000

    means = np.random.uniform(-10, 10, num_rows * 2).reshape((-1, 2))
    stds = np.random.uniform(1, 2, num_rows * 2).reshape((-1, 2))

    means *= 1e6
    print("!!!")
    print(means)
    print("!!!")
    assignments = np.random.randint(0, 2, (len(means), num_points))
    samples = np.array([
        [
            np.random.normal(
                means[row, assignments[row, point]],
                stds[row, assignments[row, point]])
            for point in range(num_points)
        ]
        for row in range(len(means))])

    assert not np.isnan(samples).any()

    m = sbgm.BivariateGaussian()
    m.fit(samples)

    # Attempt to get identifiability by comparing min and max of the two component means.
    assert_almost_equal(m.mu_.min(1), means.min(1), decimal=1)
    assert_almost_equal(m.mu_.max(1), means.max(1), decimal=1)

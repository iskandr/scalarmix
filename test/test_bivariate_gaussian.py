import numpy
import numpy.testing
numpy.random.seed(0)

from nose.tools import eq_

import sbgm


def test_simple():
    num_rows = 10
    num_points = 100000

    means = numpy.random.uniform(-100, 100, num_rows * 2).reshape((-1, 2))
    stds = numpy.random.uniform(1, 2, num_rows * 2).reshape((-1, 2))

    assignments = numpy.random.randint(0, 2, (len(means), num_points))
    samples = numpy.array([
	[
	    numpy.random.normal(means[row, assignments[row, point]], stds[row, assignments[row, point]])
	    for point in range(num_points)
	]
	for row in range(len(means))
    ])
    samples

    assert not numpy.isnan(samples).any()

    m = sbgm.BivariateGaussian()
    m.fit(samples)

    # Attempt to get identifiability by comparing min and max of the two component means.
    numpy.testing.assert_almost_equal(m.mu_.min(1), means.min(1), decimal=1)
    numpy.testing.assert_almost_equal(m.mu_.max(1), means.max(1), decimal=1)
    # Todo: we should check stds and assignments (weights) too.


def test_multiple_scales():
    num_rows = 100
    num_points = 10000

    means = numpy.random.uniform(-10, 10, num_rows * 2).reshape((-1, 2))
    stds = numpy.random.uniform(1, 2, num_rows * 2).reshape((-1, 2))

    # Make problem slightly harder by changing the scale of a few of the means and stds.
    means[1] *= 1e6
    stds[1] *= 1e3
    stds[3] *= 1e4

    assignments = numpy.random.randint(0, 2, (len(means), num_points))
    samples = numpy.array([
	[
	    numpy.random.normal(means[row, assignments[row, point]], stds[row, assignments[row, point]])
	    for point in range(num_points)
	]
	for row in range(len(means))
    ])
    samples

    assert not numpy.isnan(samples).any()

    m = sbgm.BivariateGaussian()
    m.fit(samples)

    # Attempt to get identifiability by comparing min and max of the two component means.
    numpy.testing.assert_almost_equal(m.mu_.min(1), means.min(1))
    numpy.testing.assert_almost_equal(m.mu_.max(1), means.max(1))

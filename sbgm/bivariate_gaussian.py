# Copyright (c) 2018. Mount Sinai School of Medicine
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from serializable import Serializable


class BivariateGaussian(Serializable):
    """
    Fits one bivariate Gaussian mixture per row of data
    """

    def __init__(
            self,
            max_iters=200,
            assignment_smoothing=10.0 ** -10,
            mean_bias=0.0,
            variance_bias=10.0 ** -10,
            min_improvement=0.0001,
            numeric_error_checking=True):
        self.max_iters = max_iters
        self.assignment_smoothing = assignment_smoothing
        self.mean_bias = mean_bias
        self.variance_bias = variance_bias
        self.min_improvement = min_improvement
        self.numeric_error_checking = numeric_error_checking

        self.mean_ = None
        self.variance_ = None
        self.cluster_weights_ = None

    def _e_step(self, X, mean, variance, cluster_weights):
        m1, m2 = mean[:, 0], mean[:, 1]
        s1, s2 = variance[:, 0], variance[:, 1]
        w1, w2 = cluster_weights, 1.0 - cluster_weights
        l1 = self.single_gaussian_densities(X, m1, s1)
        if self.numeric_error_checking:
            assert not np.isnan(l1).any()
            assert np.isfinite(l1).all()

        l1 *= w1[:, np.newaxis]
        if self.numeric_error_checking:
            assert not np.isnan(l1).any()
            assert np.isfinite(l1).all()

        l1 += self.assignment_smoothing

        l2 = self.single_gaussian_densities(X, m2, s2)
        if self.numeric_error_checking:
            assert not np.isnan(l2).any()
            assert np.isfinite(l2).all()

        l2 *= w2[:, np.newaxis]
        if self.numeric_error_checking:
            assert not np.isnan(l2).any()
            assert np.isfinite(l2).all()

        l2 += self.assignment_smoothing

        assignments = l1 / (l1 + l2)
        if self.numeric_error_checking:
            assert not np.isnan(assignments).any()

        return assignments

    def _m_step(self, X, assignments):
        n_rows, n_cols = X.shape
        assert assignments.shape == (n_rows, n_cols)
        a1 = assignments
        a2 = 1.0 - assignments
        if self.numeric_error_checking:
            assert (a1 >= 0).all(), a1
            assert (a1 <= 1).all(), a1

        assert X.shape == a1.shape
        assert X.shape == a2.shape
        mean = np.empty((len(X), 2), dtype="float64")
        mean.fill(self.mean_bias)
        mean[:, 0] = (X * a1).sum(axis=1) / a1.sum(axis=1)
        mean[:, 1] = (X * a2).sum(axis=1) / a2.sum(axis=1)

        assert mean.shape == (n_rows, 2), \
            "Got mu.shape=%s but expected (%d, 2)" % (mean.shape, n_rows)

        m1, m2 = mean[:, 0], mean[:, 1]

        # squared distances to both centers
        diff1 = X - m1[:, np.newaxis]
        diff1 *= diff1

        diff2 = X - m2[:, np.newaxis]
        diff2 *= diff2

        assert diff1.shape == a1.shape
        assert diff2.shape == a2.shape

        # estimate of variance is the weighted average of
        # squared distances from means
        variance = np.empty_like(mean)
        variance.fill(self.variance_bias)
        variance[:, 0] = (diff1 * a1).sum(axis=1) / a1.sum(axis=1)
        variance[:, 1] = (diff2 * a2).sum(axis=1) / a2.sum(axis=1)

        assert variance.shape == (n_rows, 2)
        if self.numeric_error_checking:
            assert (variance > 0).all(), "Found %d/%d sigma values<=0" % (
                (variance <= 0).sum(),
                variance.shape[0] * variance.shape[1])

        weights = a1.mean(axis=1)
        assert weights.shape == (n_rows,)
        if self.numeric_error_checking:
            assert (weights >= 0).all(), "Found %d/%d weights<0" % (
                (weights < 0).sum(),
                len(weights))
            assert (weights <= 1).all(), "Found %d/%d weights>1" % (
                (weights > 1).sum(),
                len(weights))
        return mean, variance, weights

    def single_gaussian_densities_explicit(self, X, mean, variance):
        n_rows, n_cols = X.shape
        assert mean.shape == (n_rows,)
        assert variance.shape == (n_rows,)
        diff = (X - mean[:, np.newaxis])
        diff *= diff
        z_score = diff / variance[:, np.newaxis]
        normalizer = np.sqrt(2 * np.pi * variance)
        unnormalized_likelihoods = np.exp(-0.5 * z_score)
        return unnormalized_likelihoods / normalizer[:, np.newaxis]

    def _check_gaussian_params(self, mean, variance, cluster_weights):
        if self.numeric_error_checking:
            n_rows = mean.shape[0]
            assert mean.shape == (n_rows, 2), mean.shape
            assert variance.shape == (n_rows, 2), variance.shape
            assert cluster_weights.shape == (n_rows,), cluster_weights.shape
            assert (variance > 0).all()
            assert np.isfinite(mean).all()
            assert np.isfinite(variance).all()
            assert np.isfinite(cluster_weights).all()

    def single_gaussian_densities(self, X, mean, variance):
        return np.exp(
            self.single_gaussian_log_densities(X, mean, variance))

    def single_gaussian_log_densities(self, X, mean, variance):
        n_rows, n_cols = X.shape
        assert mean.shape == (n_rows,)
        assert variance.shape == (n_rows,)
        diff = (X - mean[:, np.newaxis])
        diff *= diff
        z_score = diff / variance[:, np.newaxis]
        normalizer = 1.0 / np.sqrt(2 * np.pi * variance[:, np.newaxis])
        log_normalizer = np.log(normalizer)
        return -0.5 * z_score + log_normalizer

    def mixture_densities(self, X, mean=None, variance=None, cluster_weights=None):
        """
        Returns Gaussian density of each observation under the
        mean, std, and mixture coefficients for each row.
        """
        if mean is None:
            mean = self.mean_
        if variance is None:
            variance = self.variance_
        if cluster_weights is None:
            cluster_weights = self.cluster_weights_

        if mean is None or variance is None or cluster_weights is None:
            raise ValueError("You must call fit() before log_likelihood()")
        self._check_gaussian_params(mean, variance, cluster_weights)

        n_rows, n_cols = X.shape

        m1, m2 = mean[:, 0], mean[:, 1]
        s1, s2 = variance[:, 0], variance[:, 1]
        w1, w2 = cluster_weights, 1.0 - cluster_weights
        return (
            w1[:, np.newaxis] * self.single_gaussian_densities(X, m1, s1) +
            w2[:, np.newaxis] * self.single_gaussian_densities(X, m2, s2))

    def log_mixture_densities(self, X, mean=None, variance=None, cluster_weights=None):
        return np.log(self.mixture_densities(
            X,
            mean=mean,
            variance=variance,
            cluster_weights=cluster_weights))

    def log_likelihood(
            self, X, mean=None, variance=None, cluster_weights=None):
        log_densities = self.log_mixture_densities(
            X,
            mean=mean,
            variance=variance,
            cluster_weights=cluster_weights)
        return np.sum(log_densities, axis=1)

    def negative_log_likelihood(
            self, X, mean=None, variance=None, cluster_weights=None):
        return -self.log_likelihood(
            X, mean=mean, variance=variance, cluster_weights=cluster_weights)

    def normalized_log_likelihood(
            self, X, mean=None, variance=None, cluster_weights=None):
        n_samples_per_row = X.shape[1]
        log_likelihood = self.log_likelihood(
            X, mean=mean, variance=variance, cluster_weights=cluster_weights)
        return log_likelihood / n_samples_per_row

    def normalized_negative_log_likelihood(
            self, X, mean=None, variance=None, cluster_weights=None):
        return -self.normalized_log_likelihood(
            X, mean=mean, variance=variance, cluster_weights=cluster_weights)

    def initialize_mixture_params(self, X):
        mean = np.empty((len(X), 2), dtype="float64")
        mean.fill(self.mean_bias)

        variance = np.empty_like(mean)
        variance.fill(self.variance_bias)

        for i in range(len(X)):
            row = X[i, :]
            median = np.median(row)
            mean[i, 0] += np.mean(row[row < median])
            mean[i, 1] += np.mean(row[row >= median])
            variance[i, 0] += np.std(row[row < median]) ** 2
            variance[i, 1] += np.std(row[row >= median]) ** 2
        weights = np.ones(len(X)) * 0.5
        self._check_gaussian_params(mean, variance, weights)
        return mean, variance, weights

    def fit(self, X, verbose=True):
        n_rows, n_cols = X.shape
        mean, variance, cluster_weights = self.initialize_mixture_params(X)

        best_likelihoods = 10.0 ** 30 * np.ones(n_rows, dtype="float64")

        for iter_number in range(self.max_iters):
            assignments = self._e_step(
                X,
                mean=mean,
                variance=variance,
                cluster_weights=cluster_weights)
            new_mean, new_variance, new_cluster_weights = \
                self._m_step(X, assignments)
            per_row_normalized_neg_log_likelihood = \
                self.normalized_negative_log_likelihood(
                    X,
                    mean=new_mean,
                    variance=new_variance,
                    cluster_weights=new_cluster_weights)
            improvement = (best_likelihoods - per_row_normalized_neg_log_likelihood)
            improvement_fraction = improvement / best_likelihoods

            improved = improvement_fraction > self.min_improvement

            # best_likelihoods = per_row_normalized_neg_log_likelihood
            # mean = new_mean
            # variance = new_variance
            # cluster_weights = new_cluster_weights

            best_likelihoods[improved] = per_row_normalized_neg_log_likelihood[improved]
            mean[improved] = new_mean[improved]
            variance[improved] = new_variance[improved]
            cluster_weights[improved] = cluster_weights[improved]

            n_improved = improved.sum()
            if verbose:
                print(
                    "-- Epoch %d: log likelihood mean=%f (%d improved)" % (
                        iter_number + 1,
                        per_row_normalized_neg_log_likelihood.mean(),
                        n_improved))

            if n_improved == 0:
                break

        self.mean_ = mean
        self.variance_ = variance
        self.cluster_weights = cluster_weights
        return assignments

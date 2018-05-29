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
            std_smoothing=10.0 ** -10,
            mean_bias=0.0,
            min_percent_improvement=10.0 ** -5):
        self.max_iters = max_iters
        self.assignment_smoothing = assignment_smoothing
        self.std_smoothing = std_smoothing
        self.mean_bias = mean_bias
        self.min_percent_improvement = min_percent_improvement

        self.mu_ = None
        self.sigma_ = None
        self.weights_ = None

    def _e_step(self, X, mu, sigma, weights):
        l1 = self.single_gaussian_densities(X, mu[:, 0], sigma[:, 0])
        assert not np.isnan(l1).any()
        assert np.isfinite(l1).all()

        l1 *= weights[:, np.newaxis]
        assert not np.isnan(l1).any()
        assert np.isfinite(l1).all()

        l1 += self.assignment_smoothing

        l2 = self.single_gaussian_densities(X, mu[:, 1], sigma[:, 1])
        assert not np.isnan(l2).any()
        assert np.isfinite(l2).all()

        l2 *= (1.0 - weights)[:, np.newaxis]
        assert not np.isnan(l2).any()
        assert np.isfinite(l2).all()

        l2 += self.assignment_smoothing

        return l1 / (l1 + l2)

    def _m_step(self, X, assignments):
        n_rows, n_cols = X.shape
        assert assignments.shape == (n_rows, n_cols)
        a1 = assignments
        a2 = 1.0 - assignments

        assert (a1 >= 0).all(), a1
        assert (a1 <= 1).all(), a1

        mu = self.mean_bias + np.column_stack([
            (X * a1).sum(axis=1) / a1.sum(axis=1),
            (X * a2).sum(axis=1) / a2.sum(axis=1)
        ])
        assert mu.shape == (n_rows, 2), "Got mu.shape=%s but expected (%d, 2)" % (mu.shape, n_rows)

        diff1 = X - mu[:, 0][:, np.newaxis]
        diff1_squared = diff1 ** 2

        diff2 = X - mu[:, 1][:, np.newaxis]
        diff2_squared = diff2 ** 2

        sigma = self.std_smoothing + np.column_stack([
            (diff1_squared * a1).sum(axis=1) / a1.sum(axis=1),
            (diff2_squared * a2).sum(axis=1) / a2.sum(axis=1)
        ])
        assert sigma.shape == (n_rows, 2)
        assert (sigma > 0).all(), "Found %d/%d sigma values<=0" % (
            (sigma <= 0).sum(),
            sigma.shape[0] * sigma.shape[1])

        weights = a1.mean(axis=1)
        assert weights.shape == (n_rows,)
        assert (weights >= 0).all(), "Found %d/%d weights<0" % (
            (weights < 0).sum(),
            len(weights))
        assert (weights <= 1).all(), "Found %d/%d weights>1" % (
            (weights > 1).sum(),
            len(weights))
        return mu, sigma, weights

    def single_gaussian_densities_explicit(self, X, mu, sigma):
        n_rows, n_cols = X.shape
        assert mu.shape == (n_rows,)
        assert sigma.shape == (n_rows,)
        diff_squared = (X - mu[:, np.newaxis]) ** 2
        sigma_squard = sigma ** 2
        normalizer = 1.0 / np.sqrt(2 * np.pi * sigma_squard)
        z_scores = diff_squared / sigma_squard[:, np.newaxis]
        unnormalized_likelihoods = np.exp(-0.5 * z_scores)
        return normalizer[:, np.newaxis] * unnormalized_likelihoods

    def single_gaussian_densities(self, X, mu, sigma):
        return np.exp(
            self.single_gaussian_log_densities(X, mu, sigma))

    def single_gaussian_log_densities(self, X, mu, sigma):
        n_rows, n_cols = X.shape
        assert mu.shape == (n_rows,)
        assert sigma.shape == (n_rows,)
        diff_squared = (X - mu[:, np.newaxis]) ** 2
        sigma_squard = sigma ** 2
        normalizer = 1.0 / np.sqrt(2 * np.pi * sigma_squard)
        z_scores = diff_squared / sigma_squard[:, np.newaxis]
        return -0.5 * z_scores + np.log(normalizer)[:, np.newaxis]

    def likelihood(self, X, mu, sigma, weights):
        """
        Returns likelihood of each observation under the
        mean, std, and mixture coefficients for each row.
        """
        n_rows, n_cols = X.shape
        assert mu.shape == (n_rows, 2), mu.shape
        assert sigma.shape == (n_rows, 2), sigma.shape
        assert weights.shape == (n_rows,), weights.shape
        assert (sigma > 0).all()
        m1, m2 = mu[:, 0], mu[:, 1]
        s1, s2 = sigma[:, 0], sigma[:, 1]
        w1, w2 = weights, 1.0 - weights
        return (
            w1[:, np.newaxis] * self.single_gaussian_densities(X, m1, s1) +
            w2[:, np.newaxis] * self.single_gaussian_densities(X, m2, s2))

    def initialize(self, X):
        mu = np.ones((len(X), 2))
        sigma = np.ones_like(mu) * self.std_smoothing
        for i in range(len(X)):
            row = X[i, :]
            median = np.median(row)
            mu[i, 0] = np.mean(row[row < median])
            mu[i, 1] = np.mean(row[row >= median])
            sigma[i, 0] = np.std(row[row < median])
            sigma[i, 1] = np.std(row[row >= median])
        weights = np.ones(len(X)) * 0.5
        return mu, sigma, weights

    def fit(self, X, verbose=True):
        n_rows, n_cols = X.shape
        mu, sigma, weights = self.initialize(X)

        best_likelihoods = 10 ** 30 * np.ones(n_rows, dtype="float64")

        min_improvement = self.min_percent_improvement

        for iter_number in range(self.max_iters):
            assignments = self._e_step(X, mu, sigma, weights)
            print(np.around(mu, decimals=2))
            assert not np.isnan(assignments).any()
            prev_mu, prev_sigma, prev_weights = mu, sigma, weights
            mu, sigma, weights = self._m_step(X, assignments)
            per_sample_likelihood = self.likelihood(X, mu, sigma, weights)
            mean_log_likelihoods = (-np.log(per_sample_likelihood)).mean(axis=1)
            improvement = (best_likelihoods - mean_log_likelihoods)
            improvement_percent = improvement / best_likelihoods
            print("Improvement percent: %s" % (improvement_percent * 100))
            improved = improvement_percent > min_improvement
            best_likelihoods[improved] = mean_log_likelihoods[improved]
            mu[~improved] = prev_mu[~improved]
            sigma[~improved] = prev_sigma[~improved]
            weights[~improved] = prev_weights[~improved]
            if verbose:
                print(
                    "-- Epoch %d: log likelihood mean=%f (%d improved)" % (
                        iter_number + 1,
                        mean_log_likelihoods.mean(),
                        improved.sum()))

            if not improved.any():
                break

        self.mu_ = mu
        self.sigma_ = sigma
        self.weights_ = weights
        return assignments

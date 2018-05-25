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
            max_iters=200):
        self.max_iters = max_iters

        self.mu_ = None
        self.sigma_ = None
        self.weights_ = None

    def _e_step(self, X, mu, sigma, weights):
        l1 = self.single_gaussian_likelihood(X, mu[:, 0], sigma[:, 0])
        l1 *= weights[:, np.newaxis]
        l2 = self.single_gaussian_likelihood(X, mu[:, 1], sigma[:, 1])
        l2 *= (1.0 - weights)[:, np.newaxis]
        return l1 / (l1 + l2)

    def _m_step(self, X, assignments, eps=0.00001):
        n_rows, n_cols = X.shape
        assert assignments.shape == (n_rows, n_cols)
        a1 = assignments
        a2 = 1.0 - assignments

        assert (a1 >= 0).all()
        assert (a1 <= 1).all()

        mu = np.column_stack([
            (X * a1).sum(axis=1) / a1.sum(axis=1),
            (X * a2).sum(axis=1) / a2.sum(axis=1)
        ])
        assert mu.shape == (n_rows, 2), "Got mu.shape=%s but expected (%d, 2)" % (mu.shape, n_rows)

        diff1 = X - mu[:, 0][:, np.newaxis]
        diff1_squared = diff1 ** 2

        diff2 = X - mu[:, 1][:, np.newaxis]
        diff2_squared = diff2 ** 2

        sigma = eps + np.column_stack([
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

    def single_gaussian_likelihood(self, X, mu, sigma):
        n_rows, n_cols = X.shape
        assert mu.shape == (n_rows,)
        assert sigma.shape == (n_rows,)
        diff_squared = (X - mu[:, np.newaxis]) ** 2
        normalizer = 1.0 / np.sqrt(2 * np.pi * sigma)
        z_scores = diff_squared / sigma[:, np.newaxis]
        unnormalized_likelihoods = np.exp(-0.5 * z_scores)
        return normalizer[:, np.newaxis] * unnormalized_likelihoods

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
        l1 = self.single_gaussian_likelihood(X, m1, s1)
        l2 = self.single_gaussian_likelihood(X, m2, s2)
        return w1[:, np.newaxis] * l1 + w2[:, np.newaxis] * l2

    def initialize(self, X):
        m1 = np.percentile(X, q=5.0, axis=1)
        m2 = np.percentile(X, q=95.0, axis=1)
        mu = np.column_stack([m1, m2])
        sigma = np.zeros_like(mu)
        for i in range(len(X)):
            row = X[i, :]
            median = np.median(row)
            sigma[i, 0] = np.std(row[row < median])
            sigma[i, 1] = np.std(row[row >= median])
        weights = np.ones(len(X)) * 0.5
        return mu, sigma, weights

    def fit(self, X, min_improvement=10.0 ** -6, verbose=True):
        n_rows, n_cols = X.shape
        mu, sigma, weights = self.initialize(X)
        best_likelihoods = 10 ** 20 * np.ones(n_rows, dtype="float64")
        for iter_number in range(self.max_iters):
            assignments = self._e_step(X, mu, sigma, weights)
            prev_mu, prev_sigma, prev_weights = mu, sigma, weights
            mu, sigma, weights = self._m_step(X, assignments)
            per_sample_likelihood = self.likelihood(X, mu, sigma, weights)
            mean_log_likelihoods = (-np.log(per_sample_likelihood)).mean(axis=1)
            improved = (best_likelihoods - mean_log_likelihoods) > min_improvement
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

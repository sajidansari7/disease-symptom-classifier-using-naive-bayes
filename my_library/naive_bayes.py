"""
Naive Bayes Classifiers — implemented from scratch using NumPy only.

Three variants:
    - BernoulliNB  : for binary features (symptom present/absent)
    - MultinomialNB: for count/frequency features
    - GaussianNB   : for continuous features

Key design decisions:
    - All probability computations in log-space to prevent underflow.
    - Laplace (additive) smoothing to handle zero-probability problem.
    - Supports top-k predictions with calibrated probabilities.

Author: Sajid Ansari
"""

import numpy as np


class BernoulliNB:
    """
    Bernoulli Naive Bayes for binary feature vectors.

    Best suited for symptom data where each feature is 0 (absent) or 1 (present).

    Parameters
    ----------
    alpha : float
        Laplace smoothing parameter (default=1.0).
        Higher alpha → more smoothing → less sensitive to rare symptoms.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_log_prior_ = None    # log P(class)
        self.feature_log_prob_ = None   # log P(feature=1 | class)
        self.classes_ = None
        self.n_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Estimate model parameters from training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Binary feature matrix.
        y : np.ndarray, shape (n_samples,)
            Class labels (integers 0..K-1).
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, self.n_features_ = X.shape

        # --- Class priors: log P(c) ---
        class_counts = np.array([(y == c).sum() for c in self.classes_], dtype=np.float64)
        self.class_log_prior_ = np.log(class_counts / n_samples)

        # --- Feature likelihoods: log P(x_j = 1 | c) ---
        # Shape: (n_classes, n_features)
        # Laplace smoothing: (count of 1s + alpha) / (total + 2*alpha)
        self.feature_log_prob_ = np.zeros((n_classes, self.n_features_))
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            feature_counts = X_c.sum(axis=0)               # sum of 1s per feature
            n_c = X_c.shape[0]
            log_prob_1 = np.log((feature_counts + self.alpha) / (n_c + 2 * self.alpha))
            self.feature_log_prob_[i] = log_prob_1

        return self

    def _compute_log_posterior(self, X: np.ndarray) -> np.ndarray:
        """
        Compute unnormalised log-posterior for each class.

        log P(c|x) ∝ log P(c) + Σ_j [ x_j * log P(x_j=1|c)
                                       + (1-x_j) * log P(x_j=0|c) ]

        Returns shape (n_samples, n_classes).
        """
        X = np.array(X, dtype=np.float64)
        log_prob_1 = self.feature_log_prob_              # (n_classes, n_features)
        log_prob_0 = np.log(1 - np.exp(log_prob_1))     # log(1 - P(x=1|c))

        # For each sample: dot product with indicator and complement
        log_likelihood = X @ log_prob_1.T + (1 - X) @ log_prob_0.T  # (n_samples, n_classes)
        return log_likelihood + self.class_log_prior_

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Return normalised log-probabilities (log-softmax)."""
        log_post = self._compute_log_posterior(X)
        # Log-sum-exp for numerical stability
        log_sum = np.log(np.exp(log_post - log_post.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True))
        return log_post - log_post.max(axis=1, keepdims=True) - log_sum

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probabilities for each class. Shape (n_samples, n_classes)."""
        log_post = self._compute_log_posterior(X)
        # Numerically stable softmax
        shifted = log_post - log_post.max(axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / exp_vals.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class label for each sample."""
        return self.classes_[self._compute_log_posterior(X).argmax(axis=1)]

    def top_k_predictions(self, X: np.ndarray, k: int = 3):
        """
        Return top-k (disease, probability) pairs for each sample.

        Returns
        -------
        list of list of tuples: [[(disease, prob), ...], ...]
        """
        proba = self.predict_proba(X)
        results = []
        for row in proba:
            top_idx = np.argsort(row)[::-1][:k]
            results.append([(self.classes_[i], round(float(row[i]), 4)) for i in top_idx])
        return results


class MultinomialNB:
    """
    Multinomial Naive Bayes for count/frequency features.

    Parameters
    ----------
    alpha : float
        Laplace smoothing parameter (default=1.0).
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None
        self.n_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, self.n_features_ = X.shape

        class_counts = np.array([(y == c).sum() for c in self.classes_], dtype=np.float64)
        self.class_log_prior_ = np.log(class_counts / n_samples)

        # log P(feature | class) with Laplace smoothing
        self.feature_log_prob_ = np.zeros((n_classes, self.n_features_))
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            feature_counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(feature_counts / feature_counts.sum())

        return self

    def _compute_log_posterior(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64)
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_post = self._compute_log_posterior(X)
        shifted = log_post - log_post.max(axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / exp_vals.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self._compute_log_posterior(X).argmax(axis=1)]

    def top_k_predictions(self, X: np.ndarray, k: int = 3):
        proba = self.predict_proba(X)
        results = []
        for row in proba:
            top_idx = np.argsort(row)[::-1][:k]
            results.append([(self.classes_[i], round(float(row[i]), 4)) for i in top_idx])
        return results


class GaussianNB:
    """
    Gaussian Naive Bayes for continuous features.

    Assumes P(x_j | c) ~ Normal(mu_jc, sigma_jc^2).

    Parameters
    ----------
    var_smoothing : float
        Portion of the largest variance added to all variances for stability.
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.class_log_prior_ = None
        self.theta_ = None      # class-wise feature means
        self.sigma_ = None      # class-wise feature variances
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        class_counts = np.array([(y == c).sum() for c in self.classes_], dtype=np.float64)
        self.class_log_prior_ = np.log(class_counts / n_samples)

        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[i] = X_c.mean(axis=0)
            self.sigma_[i] = X_c.var(axis=0)

        self.sigma_ += self.var_smoothing * self.sigma_.max()
        return self

    def _compute_log_posterior(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_post = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            # log N(x; mu, sigma^2) = -0.5 * [log(2pi*sigma^2) + (x-mu)^2/sigma^2]
            log_likelihood = -0.5 * (
                np.log(2 * np.pi * self.sigma_[i])
                + ((X - self.theta_[i]) ** 2) / self.sigma_[i]
            ).sum(axis=1)
            log_post[:, i] = self.class_log_prior_[i] + log_likelihood

        return log_post

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_post = self._compute_log_posterior(X)
        shifted = log_post - log_post.max(axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / exp_vals.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self._compute_log_posterior(X).argmax(axis=1)]

    def top_k_predictions(self, X: np.ndarray, k: int = 3):
        proba = self.predict_proba(X)
        results = []
        for row in proba:
            top_idx = np.argsort(row)[::-1][:k]
            results.append([(self.classes_[i], round(float(row[i]), 4)) for i in top_idx])
        return results

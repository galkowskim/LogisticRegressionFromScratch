import numpy as np


class LogisticRegression:
    def __init__(
        self,
        learning_rate=0.01,
        max_iter=1000,
        tolerance=1e-4,
        add_interactions=False,
        optimizer="sgd",
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.add_interactions = add_interactions
        self.optimizer = optimizer
        self.interactions = []

    def _add_interactions(self, X):
        n_samples, n_features = X.shape
        if self.add_interactions:
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    interaction = X[:, i] * X[:, j]
                    self.interactions.append(interaction.reshape(-1, 1))
            X = np.hstack((X, *self.interactions))
        return X

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _compute_gradient(self, X, y, weights):
        pass

    def _optimize(self, X, y):
        pass

    def fit(self, X, y):
        if self.add_interactions:
            X = self._add_interactions(X)

        self.weights = np.zeros(X.shape[1])

        for _ in range(self.max_iter):
            old_weights = np.copy(self.weights)
            self._optimize(X, y)
            if np.linalg.norm(self.weights - old_weights) < self.tolerance:
                break

    def predict(self, X):
        if self.add_interactions:
            X = self._add_interactions(X)

        z = np.dot(X, self.weights)
        probabilities = self._sigmoid(z)
        predictions = np.where(probabilities >= 0.5, 1, 0)
        return predictions

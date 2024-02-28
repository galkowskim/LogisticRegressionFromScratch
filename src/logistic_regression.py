import numpy as np
from src.optimization_algorithms import sgd_optimization


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

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _compute_gradient(self, X, y):
        pass

    def _optimize(self, X, y):
        if self.optimizer == "sgd":
            sgd_optimization(self, X, y)
        pass

    def fit(self, X, y):
        
        self.weights = np.zeros(X.shape[1])
        self.biases = 0

        for _ in range(self.max_iter):
            old_weights = np.copy(self.weights)
            self._optimize(X, y)
            if np.linalg.norm(self.weights - old_weights) < self.tolerance:
                break
        return 

    def predict(self, X):
        z = np.dot(X, self.weights.reshape(-1,1)) + self.biases
        probabilities = self._sigmoid(z)
        predictions = np.where(probabilities >= 0.5, 1, 0)
        return predictions

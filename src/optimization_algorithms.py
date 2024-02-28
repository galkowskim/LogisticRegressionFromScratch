import numpy as np

def sgd_optimization(self, X, y):
    combined = list(zip(X, y))
    # Shuffle the combined array
    np.random.shuffle(combined)
    # Unpack the shuffled array back into X and y
    X, y = zip(*combined)
    for i in range(len(X)):
        y_pred = self._sigmoid(np.dot(self.weights.T, X[i]) + self.biases)
        self.weights += self.learning_rate * X[i] * (y[i] - y_pred)
        self.biases += self.learning_rate * (y[i] - y_pred)
    return
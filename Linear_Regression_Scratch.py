import numpy as np

class YusufLinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.epochs):
            y_pred = X @ self.weights + self.bias

            error = y_pred - y
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

    def predict(self, X):
        return X @ self.weights + self.bias

    @property
    def coef_(self):
        return self.weights

    @property
    def intercept_(self):
        return self.bias
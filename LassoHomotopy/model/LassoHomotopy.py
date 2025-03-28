import numpy as np


class LassoHomotopyModel:
    def __init__(self):
        self.beta_path = []
        self.lambdas = []

    def fit(self, X, y):
        # Validate input: check for empty data
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("Input data cannot be empty")

        # Validate input: check for NaN values
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Input data contains NaN values")

        n, p = X.shape
        lambda_max = np.max(np.abs(X.T @ y))
        lambda_val = lambda_max
        beta = np.zeros(p)
        active_set = set()

        while lambda_val > 1e-3:
            correlations = X.T @ (y - X @ beta)
            j_max = np.argmax(np.abs(correlations))

            if j_max not in active_set:
                active_set.add(j_max)

            # Calculate the step size using standard NumPy indexing
            step_size = min(lambda_val / max(abs(correlations[j_max]), 1e-6), 0.1)
            beta[j_max] += step_size * np.sign(correlations[j_max])

            # Store the current beta and lambda values
            self.beta_path.append(beta.copy())
            self.lambdas.append(lambda_val)

            # Decrease lambda for the next iteration
            lambda_val *= 0.9

        return LassoHomotopyResults(beta)


class LassoHomotopyResults:
    def __init__(self, beta):
        self.beta = beta

    def predict(self, X):
        # Make predictions
        return X @ self.beta

import pytest
import numpy as np
import pandas as pd
import os
import csv
from model.LassoHomotopy import LassoHomotopyModel


def get_test_path(filename):
    """Get absolute path to test files"""
    return os.path.join(os.path.dirname(__file__), filename)


def load_test_data(filename):
    """Load CSV data with format detection"""
    X, y = [], []
    with open(get_test_path(filename)) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "x_0" in row:  # small_test.csv format
                X.append([float(row["x_0"]), float(row["x_1"]), float(row["x_2"])])
                y.append(float(row["y"]))
            else:  # collinear_data.csv format
                feats = [float(v) for k, v in row.items() if k.startswith("X_")]
                X.append(feats)
                y.append(float(row["target"]))
    return np.array(X), np.array(y)


def test_model_fitting():
    model = LassoHomotopyModel()
    X = np.random.rand(100, 10)
    y = np.random.rand(100)

    results = model.fit(X, y)

    # Check if the learned coefficients have the right shape
    assert results.beta.shape[0] == X.shape[1], "Coefficient shape mismatch."


def test_prediction():
    model = LassoHomotopyModel()
    X = np.random.rand(10, 3)
    y = np.random.rand(10)

    results = model.fit(X, y)
    predictions = results.predict(X)

    # Check if predictions have the same shape as y
    assert predictions.shape[0] == y.shape[0], "Prediction shape mismatch."


def test_empty_input():
    model = LassoHomotopyModel()
    X = np.empty((0, 10))
    y = np.empty((0,))

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_nan_values():
    model = LassoHomotopyModel()
    X = np.random.rand(100, 10)
    y = np.random.rand(100)

    # Introduce NaN value
    X[0, 0] = np.nan

    with pytest.raises(ValueError):
        model.fit(X, y)


# New dataset tests
def test_small_test():
    """Test basic functionality with small_test.csv"""
    X, y = load_test_data("small_test.csv")
    model = LassoHomotopyModel()
    results = model.fit(X, y)

    # Verify basic functionality
    assert results.beta.shape == (3,), "Should learn 3 coefficients"
    predictions = results.predict(X)
    assert predictions.shape == y.shape, "Prediction shape mismatch"
    assert not np.allclose(results.beta, 0), "Model should learn non-zero coefficients"


def test_collinear_data():
    df = pd.read_csv("D:/Uni/Raga/LassoHomotopy/tests/collinear_data.csv")

    # Prepare data for modeling
    X = df.drop(columns=["target"]).values
    y = df["target"].values

    model = LassoHomotopyModel()

    results = model.fit(X, y)
    print("Model fit completed successfully. Results:", results)


def test_highly_collinear_data_sparsity():
    np.random.seed(42)
    n_samples = 100

    # Generate 3 collinear features:
    X_base = np.random.randn(n_samples, 1)
    noise = np.random.normal(0, 1e-3, (n_samples, 1))
    X_collinear = np.hstack([X_base, X_base + noise, X_base + noise])

    # Add 2 more independent features.
    X_other = np.random.randn(n_samples, 2)

    X = np.hstack([X_collinear, X_other])

    # Define target:
    true_beta = np.array([1.0, 0.0, 0.0, 2.0, -1.0])
    y = X @ true_beta + np.random.normal(0, 0.1, n_samples)

    model = LassoHomotopyModel()
    results = model.fit(X, y)

    non_zero = np.sum(np.abs(results.beta) > 1e-3)

    # Expect fewer than 5 non-zero coefficients because of collinearity.
    assert (
        non_zero < 5
    ), f"Expected fewer than 5 non-zero coefficients due to collinearity, but got {non_zero}."

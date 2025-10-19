import numpy as np
from sklearn.model_selection import train_test_split

def get_data(N=100):
    np.random.seed(42)
    X = np.random.rand(N, 2).reshape(-1, 2) * 100  # between 0 and 100
    y = (X[:, 0] + np.random.normal(0, 5, size=N)).reshape(-1, 1)  # add noise
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def get_concept_drift_data(N=100):
    np.random.seed(42)
    X = np.random.rand(N, 2).reshape(-1, 2) * 100  # between 0 and 100
    y = (X[:, 0] + np.random.normal(10, 5, size=N)).reshape(-1, 1)  # add noise+10 concept drift
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def get_data_drift_data(N=100):
    np.random.seed(42)
    X = 25 + np.random.rand(N, 2).reshape(-1, 2) * 100  # between 10 and 110
    y = (X[:, 0] + np.random.normal(0, 5, size=N)).reshape(-1, 1)  # add noise
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
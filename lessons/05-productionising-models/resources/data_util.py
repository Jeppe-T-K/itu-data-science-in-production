import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def get_breast_cancer_data():
    """The breast cancer dataset with the same stratified 70/15/15
    train/validation/test split as example_model.py: the test set is
    held out first, then the validation set is carved out of the rest."""
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=15 / 85, random_state=42, stratify=y_train)
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_breast_cancer_drift_data():
    """Data drift: the same patients and outcomes as before, but the
    measurement device has drifted - every reading comes out 25% higher.
    The input distributions have changed; the labels have not."""
    X, y = load_breast_cancer(return_X_y=True)
    return X * 1.25, y

def get_breast_cancer_concept_data():
    """Concept drift: the same measurements as before, but the labeling
    guideline changed and every diagnosis is the opposite of what it was.
    The input distributions are unchanged; the relationship X -> y flipped."""
    X, y = load_breast_cancer(return_X_y=True)
    return X, 1 - y

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


def get_classification_data(N=200):
    np.random.seed(42)
    X = np.random.rand(N, 2).reshape(-1, 2) * 100  # between 0 and 100
    logit = 0.08 * (X[:, 0] - 50)  # feature 0 drives the outcome
    p = 1 / (1 + np.exp(-logit))
    y = (np.random.rand(N) < p).astype(int)
    # Stratified 70/15/15 split (like the training notebook) so both
    # classes keep the same balance in all three sets
    X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=0.50, random_state=42, stratify=y_rest)
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_classification_drift_data(N=200):
    # Same relationship as training, but the input distribution shifts upward
    np.random.seed(42)
    X = 40 + np.random.rand(N, 2).reshape(-1, 2) * 100  # between 40 and 140
    logit = 0.08 * (X[:, 0] - 50)
    p = 1 / (1 + np.exp(-logit))
    y = (np.random.rand(N) < p).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def get_classification_concept_data(N=200):
    # Same input distribution as training, but the relationship is reversed
    np.random.seed(42)
    X = np.random.rand(N, 2).reshape(-1, 2) * 100  # between 0 and 100
    logit = -0.08 * (X[:, 0] - 50)  # flipped sign
    p = 1 / (1 + np.exp(-logit))
    y = (np.random.rand(N) < p).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error


def get_data(N=100):
    np.random.seed(42)
    X = np.random.rand(N, 2).reshape(-1, 2) * 100  # between 0 and 100
    y = X[0, :] + np.random.normal(0, 5, size=(N, 1))  # add noise
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Load data
X_train, X_test, y_train, y_test = get_data()

# Define the model hyperparameters
params = {
    "alpha": 0.5,
    "max_iter": 1000,
    "random_state": 8,
}

# Train the model
lr = Lasso(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

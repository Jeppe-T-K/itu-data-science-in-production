from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Load a binary classification dataset that ships with scikit-learn
X, y = load_breast_cancer(return_X_y=True)

# Split into train (70%), validation (15%) and test (15%) sets, keeping the
# class balance the same in all three: first hold out the test set, then
# carve the validation set out of the remaining training data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=15 / 85, random_state=42, stratify=y_train
)

# Define the model hyperparameters
params = {
    "C": 0.5,
    "max_iter": 5000,
    "random_state": 8,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1}")

# Load SKLearn libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import joblib
import os
import shutil


# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a logistic regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Print the accuracy of the model
accuracy_score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy_score}')

# Save model in artifacts
shutil.rmtree("./artifacts", ignore_errors=True)
os.makedirs("artifacts", exist_ok=True)
joblib.dump(clf, "artifacts/model.pkl")

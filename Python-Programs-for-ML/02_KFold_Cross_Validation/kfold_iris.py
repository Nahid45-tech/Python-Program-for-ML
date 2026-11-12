# K-Fold Cross-Validation using Logistic Regression on Iris Dataset
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Dataset loaded successfully:")
print(f"Original dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class names: {iris.target_names}")

# 2. Use only two classes (Setosa and Versicolor)
# Exclude Virginica (class 2)
X_binary = X[y != 2]
y_binary = y[y != 2]

print("\nAfter filtering for binary classification:")
print(f"Binary dataset shape: {X_binary.shape}")
print(f"Class distribution in binary dataset: {np.bincount(y_binary)}")
print(f"Classes used: {iris.target_names[0]} and {iris.target_names[1]}")

# 3. Implement Logistic Regression
model = LogisticRegression(random_state=42)

# 4. Perform 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X_binary, y_binary, cv=kf, scoring="accuracy")

# 5. Display accuracy for each fold and average accuracy
print("\n10-FOLD CROSS-VALIDATION RESULTS:")
print("=" * 40)
for i, acc in enumerate(scores, 1):
    print(f"Fold {i}: Accuracy = {acc:.4f}")

print("-" * 40)
print(f"Average Accuracy = {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

# Additional: Display model performance summary
print("\nPERFORMANCE SUMMARY:")
print("=" * 40)
print("Model: Logistic Regression")
print("Dataset: Iris (Binary: Setosa vs Versicolor)")
print("Cross-Validation: 10-fold")
print(f"Best Average Accuracy: {np.mean(scores):.4f}")

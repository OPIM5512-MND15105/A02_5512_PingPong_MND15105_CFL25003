import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

california_housing = fetch_california_housing()

X = california_housing.data
y = california_housing.target

print("California Housing dataset loaded successfully.")
print(f"Shape of features (X): {X.shape}")
print(f"Shape of target (y): {y.shape}")




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Dataset split into training and testing sets successfully.")
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# Instantiate MLPRegressor with early_stopping=True and custom hidden_layer_sizes
mlp_regressor = MLPRegressor(
     hidden_layer_sizes=(10,5),
                   max_iter=500,
                   batch_size=1000,
                   activation="relu",
                   validation_fraction=0.2,
                   early_stopping=True
)

# Fit the model to the training data
mlp_regressor.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred = mlp_regressor.predict(X_train)

# Make predictions on the testing set
y_test_pred = mlp_regressor.predict(X_test)

print("MLPRegressor model trained successfully and predictions made for training and testing sets.")

from pathlib import Path

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Training plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_train, y=y_train_pred, alpha=0.3)
plt.plot([y_train.min(), y_train.max()],
         [y_train.min(), y_train.max()],
         'r--', lw=2)
plt.xlabel('Actual Values (Training Set)')
plt.ylabel('Predicted Values (Training Set)')
plt.title('Actual vs. Predicted Values - Training Set')
plt.grid(True)
plt.savefig(FIGURES_DIR / "actual_vs_predicted_train.png", dpi=300, bbox_inches="tight")
plt.close()

# Testing plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2)
plt.xlabel('Actual Values (Testing Set)')
plt.ylabel('Predicted Values (Testing Set)')
plt.title('Actual vs. Predicted Values - Testing Set')
plt.grid(True)
plt.savefig(FIGURES_DIR / "actual_vs_predicted_test.png", dpi=300, bbox_inches="tight")
plt.close()
print("Plots saved successfully in the 'figures' directory.")
print("First 5 actual target values:", y[:5])   
print("First 5 feature rows:\n", X[:5]) 
print("First 5 predicted target values for training set:", y_train_pred[:5])    

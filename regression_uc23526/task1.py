import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data
script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.join(script_dir, "regression_insurance.csv")
data = pd.read_csv(file_name)

# Split into training and testing sets (80% / 20%)
X = data.drop(columns=['charges'])
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Separate categorical and numeric features
categorical_features = ['sex', 'smoker', 'region']
numeric_features = ['age', 'bmi', 'children']

X_train_cat = X_train[categorical_features]
X_train_num = X_train[numeric_features]
X_test_cat = X_test[categorical_features]
X_test_num = X_test[numeric_features]

# Preprocess categorical features using one-hot encoding
# Fit encoder on training data only
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_train_cat_encoded = encoder.fit_transform(X_train_cat)
X_test_cat_encoded = encoder.transform(X_test_cat)

# Preprocess numeric features using standardization
# Fit scaler on training data only
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

# Concatenate numeric and categorical features
X_train_processed = np.concatenate([X_train_num_scaled, X_train_cat_encoded], axis=1)
X_test_processed = np.concatenate([X_test_num_scaled, X_test_cat_encoded], axis=1)

# Get feature names for later use
cat_feature_names = encoder.get_feature_names_out(categorical_features)
all_feature_names = list(numeric_features) + list(cat_feature_names)

# Train linear regression model
model = LinearRegression()
model.fit(X_train_processed, y_train)

# Make predictions
y_train_pred = model.predict(X_train_processed)
y_test_pred = model.predict(X_test_processed)

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Print learned coefficients
print("Learned Coefficients:")
print("-" * 40)
for i, feature_name in enumerate(all_feature_names):
    coefficient = model.coef_[i]
    print(f"{feature_name}: {coefficient:.3f}")
print(f"\nIntercept: {model.intercept_:.3f}")

# Print RMSE
print("\n" + "=" * 40)
print(f"Training RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print("=" * 40)

# Scatter plot: predicted vs actual charges on test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Linear Regression: Predicted vs Actual Charges (Test Set)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

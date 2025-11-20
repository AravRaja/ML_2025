import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.join(script_dir, "regression_insurance.csv")
data = pd.read_csv(file_name)

X = data.drop(columns=['charges'])
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
categorical_features = ['sex', 'smoker', 'region']
numeric_features = ['age', 'bmi', 'children']
X_train_cat = X_train[categorical_features]
X_train_num = X_train[numeric_features]
X_test_cat = X_test[categorical_features]
X_test_num = X_test[numeric_features]
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_train_cat_encoded = encoder.fit_transform(X_train_cat)
X_test_cat_encoded = encoder.transform(X_test_cat)
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)
X_train_processed = np.concatenate([X_train_num_scaled, X_train_cat_encoded], axis=1)
X_test_processed = np.concatenate([X_test_num_scaled, X_test_cat_encoded], axis=1)
cat_feature_names = encoder.get_feature_names_out(categorical_features)
all_feature_names = list(numeric_features) + list(cat_feature_names)

lin_reg = LinearRegression()
lin_reg.fit(X_train_processed, y_train)

y_train_pred = lin_reg.predict(X_train_processed)
y_test_pred = lin_reg.predict(X_test_processed)

RMSE_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
RMSE_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("Learned Coefficients:")
for i, feature_name in enumerate(all_feature_names):
    coefficient = lin_reg.coef_[i]
    print(f"{feature_name}: {coefficient:.3f}")
print(f"\nIntercept: {lin_reg.intercept_:.3f}")

print('\nTrain RMSE: ', round(RMSE_train, 3))
print('Test RMSE: ', round(RMSE_test, 3))

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_test_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Linear Regression: Predicted vs Actual Charges in the Test Set')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

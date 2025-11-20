import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import pymc as pm
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

# Prepare data for PyMC
n_features = X_train_processed.shape[1]

# Build Bayesian linear regression model (following lab conventions)
num_samples = 2000
model = pm.Model()

with model:
    # Prepare data containers inside the model context
    X_data = pm.Data("X_data", X_train_processed)
    y_data = pm.Data("y_data", y_train.values)
    
    # Priors for regression coefficients (simple priors as in lab)
    beta = pm.Normal("beta", mu=0, sigma=1000, shape=n_features)
    
    # Prior for intercept
    alpha = pm.Normal("alpha", mu=0, sigma=5000)
    
    # Prior for noise (standard deviation) - using Uniform as in lab example
    sigma = pm.Uniform("sigma", lower=0, upper=10000)
    
    # Linear model
    mu = alpha + pm.math.dot(X_data, beta)
    
    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)
    
    # MCMC sampling using NUTS sampler (as in lab)
    sampler = pm.NUTS()
    idata = pm.sample(num_samples, tune=1000, step=sampler, return_inferencedata=True, random_seed=42, progressbar=True)

# Print posterior means for coefficients
print("Posterior Means for Coefficients:")
print("-" * 40)
posterior_means = idata.posterior.mean(dim=["chain", "draw"])
for i, feature_name in enumerate(all_feature_names):
    coefficient_mean = posterior_means["beta"].values[i]
    print(f"{feature_name}: {coefficient_mean:.3f}")
print(f"\nIntercept (alpha): {posterior_means['alpha'].values:.3f}")

# Print posterior mean for noise
print(f"\nNoise (sigma): {posterior_means['sigma'].values:.3f}")
print("=" * 40)

# Make predictions using posterior means (as in lab)
beta_mean = posterior_means["beta"].values
alpha_mean = float(posterior_means["alpha"].values)

# Ensure beta_mean is 1D array
if beta_mean.ndim > 1:
    beta_mean = beta_mean.flatten()

# Predictions on training set
y_train_pred = alpha_mean + np.dot(X_train_processed, beta_mean)

# Predictions on test set
y_test_pred = alpha_mean + np.dot(X_test_processed, beta_mean)

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train.values, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test.values, y_test_pred))

# Print RMSE (as required by task)
print('\nTrain RMSE: ', round(train_rmse, 2))
print('Test RMSE: ', round(test_rmse, 2))

# Scatter plot: predicted vs actual charges on test set
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test.values, y_test_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Charges')
ax.set_ylabel('Predicted Charges')
ax.set_title('Bayesian Linear Regression: Predicted vs Actual Charges (Test Set)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

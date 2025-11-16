import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pymc as pm
import arviz as az

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

# Build Bayesian linear regression model
with pm.Model() as model:
    # Prepare data containers inside the model context
    X_data = pm.Data("X_data", X_train_processed)
    y_data = pm.Data("y_data", y_train.values)
    
    # Priors for regression coefficients
    beta = pm.Normal("beta", mu=0, sigma=10, shape=n_features)
    
    # Prior for intercept
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    
    # Prior for noise (standard deviation)
    sigma = pm.HalfNormal("sigma", sigma=10)
    
    # Linear model
    mu = alpha + pm.math.dot(X_data, beta)
    
    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)
    
    # MCMC sampling
    trace = pm.sample(2000, tune=1000, return_inferencedata=True, random_seed=42)

# Print posterior means for coefficients
print("Posterior Means for Coefficients:")
print("-" * 40)
posterior_means = trace.posterior.mean(dim=["chain", "draw"])
for i, feature_name in enumerate(all_feature_names):
    coefficient_mean = posterior_means["beta"].values[i]
    print(f"{feature_name}: {coefficient_mean:.3f}")
print(f"\nIntercept (alpha): {posterior_means['alpha'].values:.3f}")

# Print posterior mean for noise
print(f"\nNoise (sigma): {posterior_means['sigma'].values:.3f}")
print("=" * 40)

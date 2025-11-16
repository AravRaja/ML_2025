import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
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

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_processed)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_processed)
y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)

# Define neural network architecture
input_size = X_train_processed.shape[1]
hidden_size1 = 64
hidden_size2 = 32
output_size = 1

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate on training set
model.eval()
with torch.no_grad():
    y_train_pred_tensor = model(X_train_tensor)
    train_rmse = np.sqrt(mean_squared_error(y_train.values, y_train_pred_tensor.numpy()))

# Evaluate on test set
with torch.no_grad():
    y_test_pred_tensor = model(X_test_tensor)
    test_rmse = np.sqrt(mean_squared_error(y_test.values, y_test_pred_tensor.numpy()))

# Print RMSE
print("\n" + "=" * 40)
print(f"Training RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print("=" * 40)

# Convert predictions to numpy for plotting
y_test_pred = y_test_pred_tensor.numpy().flatten()

# Scatter plot: predicted vs actual charges on test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test.values, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Neural Network: Predicted vs Actual Charges (Test Set)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

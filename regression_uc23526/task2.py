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

# Further split training set into train and validation (80% train, 20% validation of original train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Separate categorical and numeric features
categorical_features = ['sex', 'smoker', 'region']
numeric_features = ['age', 'bmi', 'children']

X_train_cat = X_train[categorical_features]
X_train_num = X_train[numeric_features]
X_val_cat = X_val[categorical_features]
X_val_num = X_val[numeric_features]
X_test_cat = X_test[categorical_features]
X_test_num = X_test[numeric_features]

# Preprocess categorical features using one-hot encoding
# Fit encoder on training data only
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_train_cat_encoded = encoder.fit_transform(X_train_cat)
X_val_cat_encoded = encoder.transform(X_val_cat)
X_test_cat_encoded = encoder.transform(X_test_cat)

# Preprocess numeric features using standardization
# Fit scaler on training data only
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_val_num_scaled = scaler.transform(X_val_num)
X_test_num_scaled = scaler.transform(X_test_num)

# Concatenate numeric and categorical features
X_train_processed = np.concatenate([X_train_num_scaled, X_train_cat_encoded], axis=1)
X_val_processed = np.concatenate([X_val_num_scaled, X_val_cat_encoded], axis=1)
X_test_processed = np.concatenate([X_test_num_scaled, X_test_cat_encoded], axis=1)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_processed)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
X_val_tensor = torch.FloatTensor(X_val_processed)
y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_processed)
y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)

# Define neural network architecture
input_size = X_train_processed.shape[1]
hidden_size1 = 128
hidden_size2 = 64
hidden_size3 = 32
output_size = 1

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize model, loss function, and optimizer
model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training loop with early stopping
num_epochs = 2000
patience = 50
best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training mode
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    train_loss = criterion(outputs, y_train_tensor)
    train_loss.backward()
    optimizer.step()
    
    # Validation mode
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
    
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    
    # Early stopping
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        patience_counter = 0
        # Save best model
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
    
    if (epoch + 1) % 200 == 0:
        train_rmse = np.sqrt(train_loss.item())
        val_rmse = np.sqrt(val_loss.item())
        print(f'Epoch [{epoch+1}/{num_epochs}], Train RMSE: {train_rmse:.2f}, Val RMSE: {val_rmse:.2f}')
    
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

# Load best model
model.load_state_dict(best_model_state)

# Evaluate on training set
model.eval()
with torch.no_grad():
    y_train_pred_tensor = model(X_train_tensor)
    train_rmse = np.sqrt(mean_squared_error(y_train.values, y_train_pred_tensor.numpy()))

# Evaluate on test set
with torch.no_grad():
    y_test_pred_tensor = model(X_test_tensor)
    test_rmse = np.sqrt(mean_squared_error(y_test.values, y_test_pred_tensor.numpy()))

# Print RMSE (as required by task)
print('\nTrain RMSE: ', round(train_rmse, 2))
print('Test RMSE: ', round(test_rmse, 2))

# Plot loss curves
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(train_losses, label='Training Loss')
ax.plot(val_losses, label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.set_title('Neural Network Training: Loss Curves')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Convert predictions to numpy for plotting
y_test_pred = y_test_pred_tensor.numpy().flatten()

# Scatter plot: predicted vs actual charges on test set
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test.values, y_test_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Charges')
ax.set_ylabel('Predicted Charges')
ax.set_title('Neural Network: Predicted vs Actual Charges (Test Set)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

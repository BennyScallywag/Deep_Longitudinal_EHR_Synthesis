import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Generate a simpler nonlinear time series data
np.random.seed(42)
n = 200
time = np.arange(n)
nonlinear_trend = np.sin(0.2 * time) + np.random.normal(scale=0.1, size=n)

# Define in-sample and out-of-sample ranges
in_sample_end = 100
n_test = 100
lag = 15  # Lookback window

# In-sample data
train_series = nonlinear_trend[:in_sample_end]

# Out-of-sample true data
y_test_new = nonlinear_trend[in_sample_end:in_sample_end + n_test - lag]

# Linear AR model using statsmodels
from statsmodels.tsa.ar_model import AutoReg
model_ar_new = AutoReg(train_series, lags=lag).fit()
preds_ar_new = model_ar_new.predict(start=lag, end=in_sample_end-1)
preds_ar_test_new = model_ar_new.predict(start=in_sample_end, end=in_sample_end + n_test - lag - 1)

# Normalize the data
scaler = StandardScaler()
train_series_scaled = scaler.fit_transform(train_series.reshape(-1, 1)).flatten()

# Prepare training data for LSTM
X_train_new = np.array([train_series_scaled[i-lag:i] for i in range(lag, in_sample_end)])
y_train_new = train_series_scaled[lag:]
X_test_new = np.array([train_series_scaled[-lag:]])

# Define the LSTM-based nonlinear autoregressive model using PyTorch
class LSTMNonlinearARModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):  # Increased layers
        super(LSTMNonlinearARModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)  # Increased dropout
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last time step output
        return out

# Set up the model parameters
input_size = 1  # We will feed one lagged value at a time
hidden_size = 128  # Increased hidden units
output_size = 1
num_layers = 3
model = LSTMNonlinearARModel(input_size, hidden_size, output_size, num_layers)

# Define the loss function and optimizer with weight decay for regularization
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay

# Prepare the training data for LSTM
X_train_lstm = torch.tensor(X_train_new.reshape(-1, lag, 1), dtype=torch.float32)
y_train_lstm = torch.tensor(y_train_new, dtype=torch.float32).view(-1, 1)

# Train the LSTM model
epochs = 4000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_lstm)
    loss = criterion(outputs, y_train_lstm)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{epochs} - Loss: {loss.item()}')

# Generate predictions using the trained LSTM model for in-sample data
model.eval()
with torch.no_grad():
    preds_nonlinear_in_sample = model(X_train_lstm).flatten().numpy()

# Prepare out-of-sample data and predict using LSTM
X_test_lstm = torch.tensor(X_test_new.reshape(1, lag, 1), dtype=torch.float32)
preds_nonlinear_out_sample = []
current_input = X_test_lstm
for _ in range(n_test - lag):
    with torch.no_grad():
        pred = model(current_input).item()
    preds_nonlinear_out_sample.append(pred)
    current_input = torch.roll(current_input, shifts=-1, dims=1)
    current_input[0, -1, 0] = pred

# Inverse transform the predictions to the original scale
preds_nonlinear_in_sample = scaler.inverse_transform(preds_nonlinear_in_sample.reshape(-1, 1)).flatten()
preds_nonlinear_out_sample = scaler.inverse_transform(np.array(preds_nonlinear_out_sample).reshape(-1, 1)).flatten()

# Plot the results
plt.figure(figsize=(14, 7))

# Shaded in-sample area
plt.axvspan(lag, in_sample_end, color='lightgrey', alpha=0.5)

# True time series (in-sample and out-of-sample combined)
plt.plot(np.arange(lag, in_sample_end + n_test - lag), np.concatenate([train_series[lag:], y_test_new]), label='True Time Series', color='black', linestyle='-')

# Linear AR model predictions (combined)
plt.plot(np.arange(lag, in_sample_end + n_test - lag), np.concatenate([preds_ar_new, preds_ar_test_new]), label='Linear AR Model', color='indigo', linestyle='-.')

# LSTM model predictions (combined)
plt.plot(np.arange(lag, in_sample_end + n_test - lag), np.concatenate([preds_nonlinear_in_sample, preds_nonlinear_out_sample]), label='Nonlinear Model (LSTM)', color='green', linestyle='--')

# Boundary between in-sample and out-of-sample
plt.axvline(x=in_sample_end, color='gray', linestyle='--')

# Adding text annotations to indicate in-sample and out-of-sample portions
plt.text(lag + 10, np.max(train_series), 'In-Sample', fontsize=12, verticalalignment='center', color='gray')
plt.text(in_sample_end + 10, np.max(train_series), 'Out-of-Sample', fontsize=12, verticalalignment='center', color='gray')

plt.title('Model Performance: Linear AR Model vs LSTM')
plt.legend(loc='upper left')
plt.show()

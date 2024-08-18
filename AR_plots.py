import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.ar_model import AutoReg

# # Generate a simpler nonlinear time series data
# np.random.seed(42)
# n = 200
# time = np.arange(n)
# #nonlinear_trend = np.sin(0.2 * time) + np.random.normal(scale=0.1, size=n)
# nonlinear_trend = 0.2 * time + np.random.normal(scale=0.5, size=n)

# # Define in-sample and out-of-sample ranges
# in_sample_end = 50
# n_test = 100
# lag = 3  # Lookback window

# # In-sample data
# train_series = nonlinear_trend[:in_sample_end]

# # Out-of-sample true data
# y_test_new = nonlinear_trend[in_sample_end:in_sample_end + n_test - lag]

# # Linear AR model using statsmodels
# model_ar_new = AutoReg(train_series, lags=lag).fit()
# preds_ar_new = model_ar_new.predict(start=lag, end=in_sample_end-1)
# preds_ar_test_new = model_ar_new.predict(start=in_sample_end, end=in_sample_end + n_test - lag - 1)

# # Normalize the data using MinMaxScaler
# scaler = MinMaxScaler()
# train_series_scaled = scaler.fit_transform(train_series.reshape(-1, 1)).flatten()

# # Prepare training data for LSTM
# X_train_new = np.array([train_series_scaled[i-lag:i] for i in range(lag, in_sample_end)])
# y_train_new = train_series_scaled[lag:]
# X_test_new = np.array([train_series_scaled[-lag:]])

# # Define a simpler LSTM-based autoregressive model using PyTorch
# class GRUARModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=1):  # Reduced layers
#         super(GRUARModel, self).__init__()
#         self.lstm = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
    
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])  # Use the last time step output
#         return out

# # Set up the model parameters
# input_size = 1  # We will feed one lagged value at a time
# hidden_size = 5  # Reduced hidden units
# output_size = 1
# num_layers = 1  # Reduced layers
# model = GRUARModel(input_size, hidden_size, output_size, num_layers)

# # Define the loss function and optimizer without weight decay
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.00001)

# # Prepare the training data for LSTM
# X_train_lstm = torch.tensor(X_train_new.reshape(-1, lag, 1), dtype=torch.float32)
# y_train_lstm = torch.tensor(y_train_new, dtype=torch.float32).view(-1, 1)

# # Train the LSTM model
# epochs = 16000
# for epoch in range(epochs):
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(X_train_lstm)
#     loss = criterion(outputs, y_train_lstm)
#     loss.backward()
#     optimizer.step()

#     if epoch % 100 == 0:
#         print(f'Epoch {epoch}/{epochs} - Loss: {loss.item()}')

# # Generate predictions using the trained LSTM model for in-sample data
# model.eval()
# with torch.no_grad():
#     preds_nonlinear_in_sample = model(X_train_lstm).flatten().numpy()

# # Prepare out-of-sample data and predict using LSTM
# X_test_lstm = torch.tensor(X_test_new.reshape(1, lag, 1), dtype=torch.float32)
# preds_nonlinear_out_sample = []
# current_input = X_test_lstm
# for _ in range(n_test - lag):
#     with torch.no_grad():
#         pred = model(current_input).item()
#     preds_nonlinear_out_sample.append(pred)
#     current_input = torch.roll(current_input, shifts=-1, dims=1)
#     current_input[0, -1, 0] = pred

# # Inverse transform the predictions to the original scale
# preds_nonlinear_in_sample = scaler.inverse_transform(preds_nonlinear_in_sample.reshape(-1, 1)).flatten()
# preds_nonlinear_out_sample = scaler.inverse_transform(np.array(preds_nonlinear_out_sample).reshape(-1, 1)).flatten()
# # Plot the results
# plt.figure(figsize=(14, 7))

# # Shaded in-sample area
# plt.axvspan(lag, in_sample_end, color='lightgrey', alpha=0.5)

# # True time series (in-sample and out-of-sample combined)
# plt.plot(np.arange(lag, in_sample_end + n_test - lag), np.concatenate([train_series[lag:], y_test_new]), label='True Time Series', color='black', linestyle='-')

# # Linear AR model predictions (combined)
# plt.plot(np.arange(lag, in_sample_end + n_test - lag), np.concatenate([preds_ar_new, preds_ar_test_new]), label='Linear AR Model', color='indigo', linestyle='-.')

# # LSTM model predictions (combined)
# plt.plot(np.arange(lag, in_sample_end + n_test - lag), np.concatenate([preds_nonlinear_in_sample, preds_nonlinear_out_sample]), label='Nonlinear Model (LSTM)', color='green', linestyle='--')

# # Boundary between in-sample and out-of-sample
# plt.axvline(x=in_sample_end, color='gray', linestyle='--')

# # Adding text annotations to indicate in-sample and out-of-sample portions
# plt.text(lag + 10, np.max(train_series), 'In-Sample', fontsize=12, verticalalignment='center', color='gray')
# plt.text(in_sample_end + 10, np.max(train_series), 'Out-of-Sample', fontsize=12, verticalalignment='center', color='gray')

# plt.title('Model Performance: Linear AR Model vs LSTM')
# plt.legend(loc='upper left')
# plt.show()


# Generate a linear function with some noise
np.random.seed(42)
n = 100
time = np.arange(n)
linear_trend = 0.5 * time + np.random.normal(scale=0.5, size=n)

# Split the data into training (first half) and testing (second half)
train_size = n // 2
test_size = n - train_size

train_series = linear_trend[:train_size]
test_series = linear_trend[train_size:]

# Normalize the data
scaler = MinMaxScaler()
train_series_scaled = scaler.fit_transform(train_series.reshape(-1, 1)).flatten()

# Prepare training data for LSTM with longer sequence length (lag)
lag = 15  # Increased lag for longer sequence
X_train = np.array([train_series_scaled[i:i + lag] for i in range(train_size - lag)])
y_train = train_series_scaled[lag:train_size]

# Define the LSTM model with simpler architecture
class LSTMARModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMARModel, self).__init__()
        self.linear_in = nn.Linear(lag, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.sigmoid(self.linear_in(x))  # Pass through initial linear layer
        x = x.unsqueeze(2)  # Add a dimension for LSTM input, making it (batch_size, hidden_size, 1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last time step output
        return out

# Set up the model parameters
input_size = 1
hidden_size = 100  # Simplified hidden units
output_size = 1
num_layers = 1  # Simplified layers
model = LSTMARModel(input_size, hidden_size, output_size, num_layers)

# Define the loss function and optimizer with reduced learning rate
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Reduced learning rate

# Prepare the training data for LSTM
X_train_lstm = torch.tensor(X_train, dtype=torch.float32)
y_train_lstm = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Train the LSTM model with early stopping
epochs = 4000
patience = 50
best_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_lstm)
    loss = criterion(outputs, y_train_lstm)
    loss.backward()
    optimizer.step()
    
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{epochs} - Loss: {loss.item()}')

# Forecasting the second half (testing)
model.eval()
X_test_lstm = X_train_lstm[-1].unsqueeze(0)  # Start with the last window from training data
X_test_lstm = model.linear_in(X_test_lstm)  # Apply the linear layer as done in forward
X_test_lstm = X_test_lstm.unsqueeze(2)  # Add the third dimension for LSTM

preds = []

with torch.no_grad():
    for _ in range(test_size):
        pred = model.fc(model.lstm(X_test_lstm)[0][:, -1, :]).item()  # Directly predict next value
        preds.append(pred)
        X_test_lstm = torch.roll(X_test_lstm, shifts=-1, dims=1)
        X_test_lstm[:, -1, :] = torch.tensor([pred]).reshape(1, 1, 1)  # Update the last input correctly

# Inverse transform the predictions to the original scale
preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(time[:train_size], train_series, label='Training Data (In-Sample)', color='blue')
plt.plot(time[train_size:], test_series, label='True Testing Data (Out-of-Sample)', color='green')
plt.plot(time[train_size:], preds, label='LSTM Forecast', color='red', linestyle='dashed')
plt.axvline(x=train_size, color='grey', linestyle='--')
plt.title('LSTM Forecasting on Linear Function')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

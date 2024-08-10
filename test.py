import opacus.layers
from Plotting_and_Visualization import plot_original_vs_generated, plot_4pane
from torch_dataloading import sine_data_generation
#from sdmetrics.single_table.multi_column.statistical import KSTest
import sdmetrics
from scipy.stats import ks_2samp
import numpy as np
import torch.nn as nn
import torch
from opacus import PrivacyEngine
import opacus
from opacus.layers import DPLSTM, DPGRU
from torch.utils.data import DataLoader, TensorDataset


# # Create a simple Discriminator neural network
# class Discriminator(nn.Module):
#     def __init__(self, input_dim, output_dim, numlayers):
#         super(Discriminator, self).__init__()
        
#         # Initialize an empty list to store layers
#         layers = []

#         # Input layer
#         layers.append(nn.Linear(input_dim, 128))
#         layers.append(nn.ReLU())

#         # Hidden layers based on numlayers
#         for _ in range(numlayers - 2):
#             layers.append(nn.Linear(128, 128))
#             layers.append(nn.ReLU())

#         # Final hidden layer
#         layers.append(nn.Linear(128, 64))
#         layers.append(nn.ReLU())

#         # Output layer
#         layers.append(nn.Linear(64, output_dim))
#         layers.append(nn.Sigmoid())

#         # Combine layers into a sequential model
#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x)

# # Generate some dummy data (features and labels)
# def generate_dummy_data(num_samples=1000):
#     X = torch.randn(num_samples, 5)
#     y = torch.round(torch.sigmoid(torch.sum(X, dim=1))).unsqueeze(1)
#     return X, y

# # Create a DataLoader
# def create_dataloader(X, y, batch_size=32):
#     dataset = TensorDataset(X, y)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     return dataloader

# # Train the Discriminator with Differential Privacy
# def train_discriminator_with_dp(epochs=5, batch_size=32, lr=0.001):
#     # Generate data
#     X, y = generate_dummy_data()
#     dataloader = create_dataloader(X, y, batch_size)

#     # Initialize the Discriminator model
#     input, output = 5, 1
#     numlayers = 3
#     model = Discriminator(input, output, numlayers=3)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.BCELoss()

#     # Attach the PrivacyEngine
#     privacy_engine = PrivacyEngine()
#     a,b,c = privacy_engine.make_private(
#         module=model,
#         optimizer=optimizer,
#         data_loader= dataloader,
#         noise_multiplier=1.0,
#         max_grad_norm=1.0,
#     )

#     # Training loop
#     a.train()
#     for epoch in range(epochs):
#         epoch_loss = 0.0
#         for X_batch, y_batch in c:
#             b.zero_grad()
#             y_pred = a(X_batch)
#             loss = criterion(y_pred, y_batch)
#             loss.backward()
#             b.step()

#             epoch_loss += loss.item()

#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss/len(c)}")

#     # Print final epsilon value
#     epsilon = privacy_engine.get_epsilon(delta=1e-5)
#     print(f"(ε = {epsilon:.2f}, δ = {1e-5})")

# train_discriminator_with_dp()

'''class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Discriminator, self).__init__()

        # Initialize the LSTM layer
        self.lstm = DPGRU(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Final fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_dim)
        
        # Pass through the fully connected layer
        out = self.fc(out)
        
        # Apply sigmoid activation
        out = self.sigmoid(out)
        
        return out

# Generate some dummy sequence data (features and labels)
def generate_dummy_data(num_samples=100, seq_len=16, input_dim=5):
    X = torch.tensor(sine_data_generation(num_samples, seq_len, input_dim), dtype=torch.float32)  # Random sequence data
    y = torch.round(torch.sigmoid(torch.sum(X, dim=(1, 2)))).unsqueeze(1)  # Labels based on sequence sum
    return X, y

# Create a DataLoader
def create_dataloader(X, y, batch_size=32):
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Train the Discriminator with Differential Privacy
def train_discriminator_with_dp(epochs=10, batch_size=32, lr=0.001):
    # Generate data
    X, y = generate_dummy_data(num_samples=1000, seq_len=10, input_dim=5)
    dataloader = create_dataloader(X, y, batch_size)

    # Initialize the Discriminator model
    input_dim = 5
    hidden_dim = 128
    output_dim = 1
    num_layers = 3
    model = Discriminator(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Attach the PrivacyEngine
    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
    )

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss/len(dataloader)}")

    # Print final epsilon value
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    print(f"(ε = {epsilon:.2f}, δ = {1e-5})")

train_discriminator_with_dp()'''

# Define the Generator class
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out

# Define the Discriminator class
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Discriminator, self).__init__()
        self.lstm = DPLSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

# Generate some dummy sequence data (features and labels)
def generate_dummy_data(num_samples=100, seq_len=16, input_dim=5):
    X = torch.randn(num_samples, seq_len, input_dim)
    y = torch.round(torch.sigmoid(torch.sum(X, dim=(1, 2)))).unsqueeze(1)
    return X, y

# Create a DataLoader
def create_dataloader(X, y, batch_size=32):
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Train the Discriminator and Generator with Differential Privacy
def train_gan_with_dp(epochs=10, batch_size=32, lr=0.001):
    X, y = generate_dummy_data(num_samples=1000, seq_len=10, input_dim=5)
    dataloader = create_dataloader(X, y, batch_size)

    input_dim = 5
    hidden_dim = 128
    output_dim = 5  # The generator should produce sequences with the same feature dimension as the input
    num_layers = 3

    discriminator = Discriminator(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers)
    generator = Generator(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    criterion = nn.BCELoss()

    privacy_engine = PrivacyEngine()
    discriminator, d_optimizer, dataloader = privacy_engine.make_private(
        module=discriminator,
        optimizer=d_optimizer,
        data_loader=dataloader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
    )

    discriminator.train()
    generator.train()
    for epoch in range(epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0

        for X_batch, _ in dataloader:
            X_batch = X_batch.float()
            batch_size = X_batch.size(0)

            # 1. Train Discriminator
            d_optimizer.zero_grad()

            Z = torch.randn(batch_size, X_batch.size(1), input_dim).to(X_batch.device)
            fake_data = generator(Z)

            real_labels = torch.ones(batch_size, 1).to(X_batch.device)
            fake_labels = torch.zeros(batch_size, 1).to(X_batch.device)

            real_output = discriminator(X_batch)
            d_real_loss = criterion(real_output, real_labels)

            fake_output = discriminator(fake_data.detach())
            d_fake_loss = criterion(fake_output, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            d_optimizer.zero_grad()

            epoch_d_loss += d_loss.item()

            # 2. Train Generator
            g_optimizer.zero_grad()

            fake_output = discriminator(fake_data)
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward(retain_graph=True)
            g_optimizer.step()  # Move this step immediately after the backward pass

            epoch_g_loss += g_loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, D Loss: {epoch_d_loss/len(dataloader)}, G Loss: {epoch_g_loss/len(dataloader)}")

    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    print(f"(ε = {epsilon:.2f}, δ = {1e-5})")

train_gan_with_dp()
# ori1 = sine_data_generation(50, 24, 5)
# ori2 = sine_data_generation(50, 24, 5)

# data = torch.utils.data.TensorDataset(torch.Tensor(ori1))
# dataloader = torch.utils.data.DataLoader(data, batch_size=3, shuffle=True)

# for data in dataloader:
#     print(data)
#plot_4pane(ori1, ori2, filename='test')

# Load the demo data, which includes:
# - A dict containing the real tables as pandas.DataFrames.
# - A dict containing the synthetic clones of the real data.
# - A dict containing metadata about the tables.
#real_data, synthetic_data, metadata = sdmetrics.load_demo()

# Obtain the list of multi table metrics, which is returned as a dict
# containing the metric names and the corresponding metric classes.
#metrics = sdmetrics.multi_table.MultiTableMetric.get_subclasses()

# Run all the compatible metrics and get a report
#print(sdmetrics.compute_metrics(metrics, real_data, synthetic_data, metadata=metadata))


#print(ori1[:][0][0])
#print(ks_2samp(ori1[:][0][0], ori2[:][0][0]))
#print(sdmetrics.compute_metrics([sdmetrics.single_table.KSComplement], real_data, synthetic_data, metadata=metadata))

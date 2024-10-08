import torch
import torch.nn as nn

import numpy as np
import sys
import os
# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch_utils import get_device, extract_time
from torch_dataloading import real_data_loading, sine_data_generation


'''I might be wrong here, but I think it works like this:
the oneclass NN is just one big embedding network which converts the input data to a latent space. 
This is helpful for time series because metrics that were not super relevant in the original temporaldistribution space 
become more relevant if we look at the latent space as a static space - in this sence, the transformation to latent space
allows us to investigate temporal data distributions as if they were static. Then, we can perform the usual metrics.'''
# class OneClassNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(OneClassNN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#         self.center = nn.Parameter(torch.zeros(output_dim))
    
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         return self.fc2(x)

class OneClassNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(OneClassNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim * 10, output_dim)
        self.center = nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = x.view(-1, 10 * hidden_dim)
        x = self.fc3(x)
        return x

def train_oneclassnn(model, real_data, epochs=1000, lr=0.001):
    device = get_device()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print("Begin Training OneClass Net for New Metrics")
    for epoch in range(epochs):
        optimizer.zero_grad()
        embedded_real = model(real_data)
        loss = criterion(embedded_real, model.center)
        loss.backward()
        optimizer.step()

        if epoch % 100 ==0:
            print(f"Loss: {loss.item()}")

    print("End Training OneClass Model for New Metrics")
    return model

def compute_alpha_precision(embedded_real, embedded_synthetic, alpha=0.9):
    distances_real = torch.norm(embedded_real - model.center, dim=1)
    alpha_quantile = torch.quantile(distances_real, alpha)
    distances_synthetic = torch.norm(embedded_synthetic - model.center, dim=1)
    alpha_precision = torch.mean((distances_synthetic <= alpha_quantile).float())
    return alpha_precision

def compute_beta_recall(embedded_real, embedded_synthetic, beta=0.9):
    distances_synthetic = torch.norm(embedded_synthetic - model.center, dim=1)
    beta_quantile = torch.quantile(distances_synthetic, beta)
    distances_real = torch.norm(embedded_real - model.center, dim=1)
    beta_recall = torch.mean((distances_real <= beta_quantile).float())
    return beta_recall

# # Example usage
input_dim = 5
hidden_dim = 64
output_dim = 16
seq_len = 10

model = OneClassNN(input_dim, hidden_dim, output_dim)
real_data = torch.randn(1000, seq_len, input_dim)  # Example real data
#real_data = torch.tensor(sine_data_generation(100, seq_len, input_dim), dtype=torch.float32)  # Example real data
synthetic_data = torch.randn(500, seq_len, input_dim)  # Example synthetic data

# Train the OneClassNN model
model = train_oneclassnn(model, real_data)

# Embed real and synthetic data
embedded_real = model(real_data)
embedded_synthetic = model(synthetic_data)

# Compute α-Precision
alpha_precision = compute_alpha_precision(embedded_real, embedded_synthetic, alpha=0.9)
print(f'α-Precision: {alpha_precision:.4f}')

# Compute β-Recall
beta_recall = compute_beta_recall(embedded_real, embedded_synthetic, beta=0.9)
print(f'β-Recall: {beta_recall:.4f}')

# def compute_authenticity(embedded_real, embedded_synthetic, epsilon=0.1):
#     distance_matrix = torch.cdist(embedded_synthetic, embedded_real, p=2)
#     min_distances = torch.min(distance_matrix, dim=1)[0]
#     authenticity = torch.mean((min_distances > epsilon).float())
#     return authenticity

# Compute Authenticity
# authenticity = compute_authenticity(embedded_real, embedded_synthetic, epsilon=0.1)
# print(f'Authenticity: {authenticity:.4f}')

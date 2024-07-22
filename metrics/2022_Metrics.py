import torch
import torch.nn as nn
from torch_utils import get_device, extract_time
import numpy as np

class OneClassNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(OneClassNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.center = nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def train_oneclassnn(real_data, generated_data, epochs=1000, lr=0.001):
    device = get_device()

    # Basic Parameters
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)
    no, seq_len, dim = ori_data.shape    
    
    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
    
    # Build a post-hoc RNN discriminator network 
    hidden_dim = int(dim / 2)
    iterations = 2000
    batch_size = 128

    model = OneClassNN()

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

def compute_authenticity(embedded_real, embedded_synthetic, epsilon=0.1):
    distance_matrix = torch.cdist(embedded_synthetic, embedded_real, p=2)
    min_distances = torch.min(distance_matrix, dim=1)[0]
    authenticity = torch.mean((min_distances > epsilon).float())
    return authenticity

# Example usage
input_dim = 32
hidden_dim = 64
output_dim = 16

model = OneClassNN(input_dim, hidden_dim, output_dim)
real_data = torch.randn(100, input_dim)  # Example real data
synthetic_data = torch.randn(50, input_dim)  # Example synthetic data

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

# Compute Authenticity
authenticity = compute_authenticity(embedded_real, embedded_synthetic, epsilon=0.1)
print(f'Authenticity: {authenticity:.4f}')

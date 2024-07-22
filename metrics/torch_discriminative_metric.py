'''# Necessary Packages
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torch_utils import train_test_divide, extract_time, batch_generator

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        _, hidden = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        hidden = hidden[-1]
        y_hat_logit = self.fc(hidden)
        y_hat = torch.sigmoid(y_hat_logit)
        return y_hat_logit, y_hat

def discriminative_score_metrics(ori_data, generated_data):
    """Use post-hoc RNN to classify original data and synthetic data
    
    Args:
        ori_data: original data
        generated_data: generated synthetic data
        
    Returns:
        discriminative_score: np.abs(classification accuracy - 0.5)
    """
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape    
    
    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(ori_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
    
    # Network parameters
    hidden_dim = int(dim / 2)
    iterations = 2000
    batch_size = 128
    
    # Initialize the discriminator model
    discriminator = Discriminator(dim, hidden_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(discriminator.parameters())
    
    # Train/test division for both original and generated data
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)
    
    train_dataset = TensorDataset(torch.tensor(train_x, dtype=torch.float32),
                                  torch.tensor(train_x_hat, dtype=torch.float32),
                                  torch.tensor(train_t, dtype=torch.int64),
                                  torch.tensor(train_t_hat, dtype=torch.int64))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training step
    for itt in range(iterations):
        for X_mb, X_hat_mb, T_mb, T_hat_mb in train_loader:
            X_mb, X_hat_mb = X_mb.to(device), X_hat_mb.to(device)
            T_mb, T_hat_mb = T_mb.to(device), T_hat_mb.to(device)
            
            # Train discriminator
            optimizer.zero_grad()
            y_logit_real, y_pred_real = discriminator(X_mb, T_mb)
            y_logit_fake, y_pred_fake = discriminator(X_hat_mb, T_hat_mb)
            
            d_loss_real = criterion(y_pred_real, torch.ones_like(y_pred_real))
            d_loss_fake = criterion(y_pred_fake, torch.zeros_like(y_pred_fake))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer.step()
    
    # Test the performance on the testing set
    discriminator.eval()
    with torch.no_grad():
        test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
        test_x_hat = torch.tensor(test_x_hat, dtype=torch.float32).to(device)
        test_t = torch.tensor(test_t, dtype=torch.int64).to(device)
        test_t_hat = torch.tensor(test_t_hat, dtype=torch.int64).to(device)
        
        y_pred_real_curr = discriminator(test_x, test_t)[1].cpu().numpy()
        y_pred_fake_curr = discriminator(test_x_hat, test_t_hat)[1].cpu().numpy()
    
    y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis=0))
    y_label_final = np.concatenate((np.ones(len(y_pred_real_curr)), np.zeros(len(y_pred_fake_curr))), axis=0)
    
    # Compute the accuracy
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    discriminative_score = np.abs(0.5 - acc)
    
    return discriminative_score'''


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch_utils import train_test_divide, extract_time, batch_generator, get_device
import subprocess

# def extract_time(data):
#     """Extract time information from the data."""
#     time = [len(seq) for seq in data]
#     max_seq_len = max(time)
#     return time, max_seq_len

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, t):
        t_cpu = torch.tensor(t, dtype=torch.long).cpu()  # Ensure lengths are on CPU
        packed_input = nn.utils.rnn.pack_padded_sequence(x, t_cpu, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        y_hat_logit = self.fc(output[:, -1, :])  # Get the last output state
        y_hat = self.sigmoid(y_hat_logit)
        return y_hat_logit, y_hat

# def train_test_divide(ori_data, generated_data, ori_time, generated_time):
#     """Divide the original and generated data into train and test sets."""
#     # For simplicity, assume 70% train and 30% test split
#     train_size = int(0.7 * len(ori_data))
#     train_x, train_x_hat = ori_data[:train_size], generated_data[:train_size]
#     test_x, test_x_hat = ori_data[train_size:], generated_data[train_size:]
#     train_t, train_t_hat = ori_time[:train_size], generated_time[:train_size]
#     test_t, test_t_hat = ori_time[train_size:], generated_time[train_size:]
    
#     return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat

# def batch_generator(data, time, batch_size):
#     """Generate batches of data."""
#     idx = np.random.permutation(len(data))
#     idx = idx[:batch_size]
#     batch_data = [torch.tensor(data[i], dtype=torch.float32) for i in idx]
#     batch_time = [time[i] for i in idx]
#     batch_data = nn.utils.rnn.pad_sequence(batch_data, batch_first=True)
#     return batch_data, batch_time

# def get_device():
#     if torch.cuda.is_available():
#         try:
#             # Get the list of GPUs
#             result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
#             output = result.stdout.decode('utf-8')
            
#             # Parse the output
#             gpus = output.strip().split('\n')
#             free_memory = []
#             for gpu in gpus:
#                 index, memory_free = gpu.split(',')
#                 free_memory.append((int(index), int(memory_free)))
            
#             # Get the GPU with the most free memory
#             gpu_index = max(free_memory, key=lambda x: x[1])[0]
#             device = torch.device(f"cuda:{gpu_index}")
#             print(f"Using GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")
#         except Exception as e:
#             print(f"Error querying GPUs: {e}")
#             device = torch.device("cuda:0")
#             print(f"Defaulting to GPU 0: {torch.cuda.get_device_name(0)}")
#     else:
#         device = torch.device("cpu")
#         print("Using CPU")
#     return device

def discriminative_score_metrics(ori_data, generated_data):
    """Use post-hoc RNN to classify original data and synthetic data
    
    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      
    Returns:
      - discriminative_score: np.abs(classification accuracy - 0.5)
    """
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
    
    # Initialize the model, loss function, and optimizer
    model = Discriminator(input_dim=dim, hidden_dim=hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Train/test division for both original and generated data
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)
    
    # Training step
    model.train()
    for itt in range(iterations):
        # Batch setting
        X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)
        
        X_mb, X_hat_mb = X_mb.to(device), X_hat_mb.to(device)
        
        # Train discriminator
        optimizer.zero_grad()
        
        y_logit_real, y_pred_real = model(X_mb, T_mb)
        y_logit_fake, y_pred_fake = model(X_hat_mb, T_hat_mb)
        
        d_loss_real = criterion(y_logit_real, torch.ones_like(y_logit_real))
        d_loss_fake = criterion(y_logit_fake, torch.zeros_like(y_logit_fake))
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer.step()

        if itt % 100 == 0:
            print(f'iteration: {itt}/{iterations}, Loss: {d_loss.item()}')
    
    # Test the performance on the testing set
    model.eval()
    test_x, test_x_hat = torch.tensor(test_x, dtype=torch.float32).to(device), torch.tensor(test_x_hat, dtype=torch.float32).to(device)
    test_t, test_t_hat = test_t, test_t_hat  # Keep lengths on CPU
    
    with torch.no_grad():
        y_logit_real_curr, y_pred_real_curr = model(test_x, test_t)
        y_logit_fake_curr, y_pred_fake_curr = model(test_x_hat, test_t_hat)
    
    y_pred_final = torch.cat((y_pred_real_curr, y_pred_fake_curr), axis=0).cpu().numpy()
    y_label_final = np.concatenate((np.ones(len(y_pred_real_curr)), np.zeros(len(y_pred_fake_curr))), axis=0)
    
    # Compute the accuracy
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    discriminative_score = np.abs(0.5 - acc)
    
    return discriminative_score
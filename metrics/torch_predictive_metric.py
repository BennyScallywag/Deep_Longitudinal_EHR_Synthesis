import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from torch_utils import get_device, batch_generator, train_test_divide, extract_time
import subprocess

# def extract_time(data):
#     """Extract time information from the data."""
#     time = [len(seq) for seq in data]
#     max_seq_len = max(time)
#     return time, max_seq_len

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

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Predictor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, t):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, t, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        y_hat_logit = self.fc(output)
        y_hat = self.sigmoid(y_hat_logit)
        return y_hat

def predictive_score_metrics(ori_data, generated_data):
    """Report the performance of Post-hoc RNN one-step ahead prediction.
    
    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      
    Returns:
      - predictive_score: MAE of the predictions on the original data
    """
    
    # Basic Parameters
    device = get_device()
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)
    no, seq_len, dim = ori_data.shape
    
    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
    
    # Build a post-hoc RNN predictive network 
    hidden_dim = int(dim / 2)
    iterations = 5000
    batch_size = 128
    
    # Initialize the model, loss function, and optimizer
    model = Predictor(input_dim=dim-1, hidden_dim=hidden_dim).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters())
    
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = train_test_divide(ori_data, generated_data, ori_time, generated_time, train_rate=0.8)

    # Training using Synthetic dataset
    model.train()
    for itt in range(iterations):
        idx = np.random.permutation(len(generated_data))
        train_idx = idx[:batch_size]     
        
        X_mb = [torch.tensor(generated_data[i][:-1, :(dim-1)], dtype=torch.float32).to(device) for i in train_idx]
        T_mb = [generated_time[i] - 1 for i in train_idx]
        Y_mb = [torch.tensor(generated_data[i][1:, (dim-1)].reshape(-1, 1), dtype=torch.float32).to(device) for i in train_idx]
        
        X_mb = nn.utils.rnn.pad_sequence(X_mb, batch_first=True)
        Y_mb = nn.utils.rnn.pad_sequence(Y_mb, batch_first=True)
        
        optimizer.zero_grad()
        y_pred = model(X_mb, T_mb)
        loss = criterion(y_pred, Y_mb)
        loss.backward()
        optimizer.step()

        if itt % 100 == 0:
            print(f'iteration: {itt}/{iterations}, Loss: {loss.item()}')
    
    # Test the trained model on the original data
    model.eval()
    idx = np.random.permutation(len(ori_data))
    train_idx = idx[:no]
    
    X_mb = [torch.tensor(ori_data[i][:-1, :(dim-1)], dtype=torch.float32).to(device) for i in train_idx]
    T_mb = [ori_time[i] - 1 for i in train_idx]
    Y_mb = [torch.tensor(ori_data[i][1:, (dim-1)].reshape(-1, 1), dtype=torch.float32).to(device) for i in train_idx]
    
    X_mb = nn.utils.rnn.pad_sequence(X_mb, batch_first=True)
    Y_mb = nn.utils.rnn.pad_sequence(Y_mb, batch_first=True)

    with torch.no_grad():
        pred_Y_curr = model(X_mb, T_mb)
    
    predictive_score = torch.mean(torch.abs(Y_mb - pred_Y_curr))
    
    #predictive_score = MAE_temp / no
    
    return predictive_score

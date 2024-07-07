import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error

def extract_time(data):
    """Extract time information from the data."""
    time = [len(seq) for seq in data]
    max_seq_len = max(time)
    return time, max_seq_len

def get_device(gpu_index=0):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_index}")
        print(f"Using GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

device = get_device(gpu_index=2)

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
    
    # Compute the performance in terms of MAE
    MAE_temp = 0
    for i in range(no):
        MAE_temp += mean_absolute_error(Y_mb[i].cpu().numpy(), pred_Y_curr[i].cpu().numpy())
    
    predictive_score = MAE_temp / no
    
    return predictive_score

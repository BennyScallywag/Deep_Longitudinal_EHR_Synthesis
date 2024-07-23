## Necessary Packages
import numpy as np
import torch
import torch.nn as nn
import subprocess

#can replace the following four functions, instead using the ones in the utils.py file
def MinMaxScaler(data):
    """Min-Max Normalizer.
    Args:
      - data: raw data
    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """    
    min_val = np.min(np.min(data, axis = 0), axis = 0)
    data = data - min_val
      
    max_val = np.max(np.max(data, axis = 0), axis = 0)
    norm_data = data / (max_val + 1e-7)
      
    return norm_data, min_val, max_val


def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate = 0.8):
  """Divide train and test data for both original and synthetic data.
  
  Args:
    - data_x: original data
    - data_x_hat: generated data
    - data_t: original time
    - data_t_hat: generated time
    - train_rate: ratio of training data from the original data
  """
  # Divide train/test index (original data)
  no = len(data_x)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
    
  train_x = np.array([data_x[i] for i in train_idx])
  test_x = np.array([data_x[i] for i in test_idx])
  train_t = np.array([data_t[i] for i in train_idx])
  test_t = np.array([data_t[i] for i in test_idx])      
    
  # Divide train/test index (synthetic data)
  no = len(data_x_hat)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
  
  train_x_hat = np.array([data_x_hat[i] for i in train_idx])
  test_x_hat = np.array([data_x_hat[i] for i in test_idx])
  train_t_hat = np.array([data_t_hat[i] for i in train_idx])
  test_t_hat = np.array([data_t_hat[i] for i in test_idx])
  
  return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time (data):
  """Returns Maximum sequence length and each sequence length.
  
  Args:
    - data: original data
    
  Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
  """
  time = list()
  max_seq_len = 0
  for i in range(len(data)):
    max_seq_len = max(max_seq_len, len(data[i][:,0]))
    time.append(len(data[i][:,0]))
    
  return time, max_seq_len

def rnn_cell(module_name, hidden_dim):
    """Basic RNN Cell.

    Args:
        - module_name: 'gru', 'lstm', or 'lstmLN'
        - hidden_dim: dimension of the hidden state
    
    Returns:
        - rnn_cell: RNN Cell
    """
    assert module_name in ['gru', 'lstm', 'lstmLN'], "module_name must be 'gru', 'lstm', or 'lstmLN'"
    
    if module_name == 'gru':
        rnn_cell = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
    elif module_name == 'lstm':
        rnn_cell = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
    elif module_name == 'lstmLN':
        rnn_cell = nn.Module()
        rnn_cell.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        rnn_cell.layer_norm = nn.LayerNorm(hidden_dim)
        
        def forward(self, x):
            output, (hn, cn) = self.rnn(x)
            output = self.layer_norm(output)
            return output, (hn, cn)
        
        rnn_cell.forward = forward.__get__(rnn_cell)
    
    return rnn_cell



def random_generator (batch_size, z_dim, T_mb, max_seq_len):
  """Random vector generation.
  
  Args:
    - batch_size: size of the random vector
    - z_dim: dimension of random vector
    - T_mb: time information for the random vector
    - max_seq_len: maximum sequence length
    
  Returns:
    - Z_mb: generated random vector
  """
  #print('batch size:',batch_size)
  #print('Len(t_mb):',len(T_mb))
  #print('zdim', z_dim)
  #print('max_seq_len', max_seq_len)

  Z_mb = list()
  for i in range(batch_size):
    #this portion is new code because the last batch is a little smaller, which led to index error
    if i >= len(T_mb):
      break  # Stop the loop if i exceeds the length of T_mb
    temp = np.zeros([max_seq_len, z_dim])
    temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
    temp[:T_mb[i],:] = temp_Z
    Z_mb.append(temp_Z)
  return np.array(Z_mb)

def batch_generator(data, time, batch_size):
    """Generate batches of data."""
    idx = np.random.permutation(len(data))
    idx = idx[:batch_size]
    batch_data = [torch.tensor(data[i], dtype=torch.float32) for i in idx]
    batch_time = [time[i] for i in idx]
    batch_data = nn.utils.rnn.pad_sequence(batch_data, batch_first=True)
    return batch_data, batch_time

def get_device():
    if torch.cuda.is_available():
        try:
            # Get the list of GPUs
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
            output = result.stdout.decode('utf-8')
            
            # Parse the output
            gpus = output.strip().split('\n')
            free_memory = []
            for gpu in gpus:
                index, memory_free = gpu.split(',')
                free_memory.append((int(index), int(memory_free)))
            
            # Get the GPU with the most free memory
            gpu_index = max(free_memory, key=lambda x: x[1])[0]
            device = torch.device(f"cuda:{gpu_index}")
            print(f"Using GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")
        except Exception as e:
            print(f"Error querying GPUs: {e}")
            device = torch.device("cuda:0")
            print(f"Defaulting to GPU 0: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device
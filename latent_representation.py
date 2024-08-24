import torch
import numpy as np
import torch.nn as nn
from torch_dataloading import sine_data_generation
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch_utils as tu
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import mahalanobis

class Embed(nn.Module):
    def __init__(self, num_features, hidden_dim):         #Feature size, hidden_dim, n_layers
        super(Embed, self).__init__()
        self.gru = nn.GRU(num_features, hidden_dim, 1, batch_first=True)

    def forward(self, x):
        all_hiddens, last_hidden  = self.gru(x)         #expects input of shape (seq_len, batch, input_size) or (seq_len, num_features) or (batch, seq_len, num_features)
        representation = last_hidden[-1, :, :]          #shape (batch_size, hidden_dim)
        #all_hiddens output shape is (batch_size, seq_len, hidden_dim) for batch_first=True
        #last_hidden output shape is (num_layers, batch_size, hidden_dim) - need to use this one since recovery will need to copy over both layers for innitial hidden states
        return last_hidden, representation
    
#Note: THE RECOVERY NETWORK LEARNS TO RECOVER THE ORIGINAL DATA 
#WITH THE SEQUENCE DIMENSION REVERSED (must flip afterward if using)
class Recover(nn.Module):
    def __init__(self, hidden_dim, num_features, seq_len):              
        super(Recover, self).__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, 1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_features)                       #FC layer to return to original feature space
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

    def forward(self, h_0):                                              #input shape should be (batch_size, seq_len, hidden_dim) i.e. (N,L,H)
        batch_size = h_0.size(1)
        x = torch.zeros(batch_size, self.seq_len, self.hidden_dim)       #these are only to give the correct shape and sequence length, have no effect on the output
        all_hiddens, _  = self.gru(x, h_0)                               #copy over final hidden from embedder as innitial hidden here
        recovered_seq = self.fc(all_hiddens)
        return recovered_seq


def reinitialize_parameters(model):
    """
    Reinitialize all parameters of the given model.
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)


def train_representation(original_data, hidden_dim, epochs=5000):
    '''Train the embedder and recovery networks to generate and revese
    a d-dimensional latent vector representation of time series data
    -------------Inputs-------------
    Original data (tensor): shape = (batch_size, seq_len, num_features)'
    hidden_dim (int): dimension of the latent space
    epochs (int): number of training epochs
    --------Returns: Trained models--------
    '''
    device = tu.get_device()
    seq_len, num_features = original_data.size(1), original_data.size(2)
    real_dataloader = DataLoader(original_data, batch_size=128, shuffle=False)        #Batch is first!
    
    embedder = Embed(num_features, hidden_dim).to(device)
    recovery = Recover(hidden_dim, num_features, seq_len).to(device)
    
    embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=0.001)
    recovery_optimizer = torch.optim.Adam(recovery.parameters(), lr=0.001)
    
    criterion = nn.MSELoss()
    
    #Train the nets together
    for epoch in range(epochs):
        for data in real_dataloader:
            data = data.float().to(device)
            embedder_optimizer.zero_grad()
            recovery_optimizer.zero_grad()
            
            last_hidden, representation = embedder(data)
            recovered = recovery(last_hidden)
            
            loss = criterion(recovered, torch.flip(data, dims=[1]))     #flipping the data marginally improves loss convergence rates
            loss.backward()
            embedder_optimizer.step()
            recovery_optimizer.step()
            
        if epoch % 50 == 0:
            print(f'Epoch {epoch}/{epochs}: Recovery Loss = {loss.item()}')

    return embedder, recovery


def compute_mmd(real_data, synthetic_data, gamma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between real and synthetic data using an RBF kernel.
    ----------Inputs-----------
        real_data (np.array): Real data with shape (batch_size, hidden_dimension).
        synthetic_data (np.array): Synthetic data with shape (batch_size, hidden_dimension).
        gamma (float): Parameter for the RBF kernel. Default is 1.0.
    ---------Returns-----------
        float: The MMD value between the two distributions.
    """
    # Compute the RBF kernel matrices
    K_real = rbf_kernel(real_data, real_data, gamma=gamma)
    K_synthetic = rbf_kernel(synthetic_data, synthetic_data, gamma=gamma)
    K_cross = rbf_kernel(real_data, synthetic_data, gamma=gamma)

    # Calculate the MMD using the kernel matrices
    mmd_value = np.mean(K_real) + np.mean(K_synthetic) - 2 * np.mean(K_cross)
    
    return mmd_value


def compute_mahalanobis(real_data, synthetic_data):
    """Compute Mahalanobis distance between the means of real and synthetic data.
    ----------Inputs-----------
        real_data (np.array): Real data with shape (batch_size, hidden_dimension).
        synthetic_data (np.array): Synthetic data with shape (batch_size, hidden_dimension).
    ---------Returns-----------
        float: The Mahalanobis distance between the means of the two distributions.
    """
    # Compute the mean vectors
    mean_real = np.mean(real_data, axis=0)
    mean_synthetic = np.mean(synthetic_data, axis=0)

    # Compute the covariance matrix of the real data
    cov_real = np.cov(real_data, rowvar=False)
    
    # Compute the Mahalanobis distance
    inv_cov_real = np.linalg.inv(cov_real)
    mahalanobis_dist = mahalanobis(mean_real, mean_synthetic, inv_cov_real)
    
    return mahalanobis_dist

def evaluate_latent_metrics(original_data, synthetic_data, hidden_dim, training_epochs):
    '''Evaluate the latent space representation of the original data
    using the embedder and recovery networks
    -------------Inputs-------------
    Original data (tensor): shape = (batch_size, seq_len, num_features)'
    Synthetic data (tensor): shape = (batch_size, seq_len, num_features)'
    hidden_dim (int): dimension of the latent space
    training_epochs (int): number of training epochs
    ------------Returns-------------
    dict: statistical metrics based on the latent representation
    '''
    embedder, recovery = train_representation(original_data, hidden_dim, training_epochs)

    #get latent representations of original and synthetic data
    _, H = embedder(original_data)
    _, H_hat = embedder(synthetic_data)

    mmd = compute_mmd(H.detach().numpy(), H_hat.detach().numpy())
    mahalanobis_dist = compute_mahalanobis(H.detach().numpy(), H_hat.detach().numpy())
    latent_metric_results = {'MMD': mmd, 'Mahalanobis Distance': mahalanobis_dist}
    print(latent_metric_results)

    return latent_metric_results

#Testing 
#real_data = torch.tensor(sine_data_generation(100, 12, 5),dtype=torch.float32)
#synthetic_data = torch.tensor(sine_data_generation(100, 12, 5),dtype=torch.float32)
#evaluate_latent_metrics(real_data, synthetic_data, 10, 1000)

#test_vect = torch.tensor([[[1,2,3],[3,4,5]],[[1,2,3],[3,4,5]]],dtype=torch.float32)
#e,r=train_representation(test_vect, 2, 4000)
#print(r(e(test_vect)[0]))





import torch
import numpy as np
import torch.nn as nn
from torch_dataloading import sine_data_generation
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def reinitialize_parameters(model):
    """
    Reinitialize all parameters of the given model.
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)

#starting with unbatched data
class Embed(nn.Module):
    def __init__(self, num_features, hidden_dim):         #Feature size, hidden_dim, n_layers
        super(Embed, self).__init__()
        self.gru = nn.GRU(num_features, hidden_dim, 1, batch_first=True)

    def forward(self, x):
        all_hiddens, last_hidden  = self.gru(x)         #expects input of shape (seq_len, batch, input_size) or (seq_len, num_features)
        #all_hiddens output shape is (batch_size, seq_len, hidden_dim) for batch_first=True
        #last_hidden output shape is (num_layers, batch_size, hidden_dim) - need to use this one since recovery will need to copy over both layers for innitial hidden states
        return last_hidden
    
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
    

epochs = 1000
no, seq_len, num_features = 200, 12, 1
hidden_dim = 5
    
real_data = np.array(sine_data_generation(no, seq_len, num_features))
real_data = torch.tensor(real_data, dtype=torch.float32)
real_dataloader = DataLoader(real_data, batch_size=32, shuffle=True)        #Batch is first!

embedder = Embed(num_features, hidden_dim)
recovery = Recover(hidden_dim, num_features, seq_len)

embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=0.001)
recovery_optimizer = torch.optim.Adam(recovery.parameters(), lr=0.001)

criterion = nn.MSELoss()

#Train the nets together
for epoch in range(epochs):
    for data in real_dataloader:
        embedder_optimizer.zero_grad()
        recovery_optimizer.zero_grad()
        
        embedded = embedder(data)
        recovered = recovery(embedded)
        
        loss = criterion(recovered, torch.flip(data, dims=[1]))     #flipping the data marginally improves loss convergence rates
        loss.backward()
        
        embedder_optimizer.step()
        recovery_optimizer.step()
        
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: {loss.item()}')
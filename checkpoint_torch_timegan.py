import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
#rnn_cell uses tensorflow builtins, the other three do not
from torch_utils import extract_time, rnn_cell, random_generator, batch_generator

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

def load_checkpoint(filename='checkpoint.pth'):
    # Define the relative path to the Checkpoints directory within the script's directory
    script_dir = os.path.dirname(__file__)
    checkpoint_path = os.path.join(script_dir, 'Checkpoints', filename)

    # Load the checkpoint if it exists
    if os.path.exists(checkpoint_path):
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint

    # if os.path.exists(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path)
    #     print(f"Checkpoint loaded from {checkpoint_path}")
    #     return checkpoint
    # else:
    #     print(f"No checkpoint found at {checkpoint_path}")
    #     return None

# def save_checkpoint(epoch, model_dict, optimizer_dict, losses, filename='checkpoint.pth'):
#     checkpoint = {
#         'epoch': epoch,
#         'model_state_dict': {k: v.state_dict() for k, v in model_dict.items()},
#         'optimizer_state_dict': {k: v.state_dict() for k, v in optimizer_dict.items()},
#         'losses': losses
#     }
#     torch.save(checkpoint, filename)
#     print(f"Checkpoint saved at epoch {epoch}")

def save_checkpoint(epoch, model_dict, optimizer_dict, losses, filename='checkpoint.pth'):
    # Create the Checkpoints directory if it does not exist
    #checkpoint_dir = os.path.join(os.getcwd(), 'Checkpoints')
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'Checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Create the full checkpoint file path
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': {k: v.state_dict() for k, v in model_dict.items()},
        'optimizer_state_dict': {k: v.state_dict() for k, v in optimizer_dict.items()},
        'losses': losses
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

# def save_checkpoint(epoch, model_dict, optimizer_dict, losses, filename='checkpoint.pth'):
#     # Define the relative path to the Checkpoints directory within the MMSC_Thesis folder
#     checkpoint_dir = os.path.join(os.path.dirname(__file__), 'MMSC_Thesis', 'Checkpoints')
#     if not os.path.exists(checkpoint_dir):
#         os.makedirs(checkpoint_dir)
    
#     # Create the full checkpoint file path
#     checkpoint_path = os.path.join(checkpoint_dir, filename)
    
#     checkpoint = {
#         'epoch': epoch,
#         'model_state_dict': {k: v.state_dict() for k, v in model_dict.items()},
#         'optimizer_state_dict': {k: v.state_dict() for k, v in optimizer_dict.items()},
#         'losses': losses
#     }
#     torch.save(checkpoint, checkpoint_path)
#     print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

def get_device(gpu_index=0):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_index}")
        print(f"Using GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

device = get_device(gpu_index=2)

class Embedder(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers):
        super(Embedder, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        h, _ = self.rnn(x)
        h = torch.sigmoid(self.fc(h))
        #h = self.fc(h)
        return h

class Recovery(nn.Module):
    def __init__(self, hidden_dim, output_size, num_layers):
        super(Recovery, self).__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, h):
        x_tilde, _ = self.rnn(h)
        x_tilde = torch.sigmoid(self.fc(x_tilde))
        #x_tilde = self.fc(x_tilde)
        return x_tilde

class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_layers):
        super(Generator, self).__init__()
        self.rnn = nn.GRU(z_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z):
        e, _ = self.rnn(z)
        e = torch.sigmoid(self.fc(e))
        #e = self.fc(e)
        return e

class Supervisor(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(Supervisor, self).__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h):
        s, _ = self.rnn(h)
        s = torch.sigmoid(self.fc(s))
        #s = self.fc(s)
        return s

#Dont need to use padded sequence since inputs are always in latent space (same dimension)
class Discriminator(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        y_hat, _ = self.rnn(h)
        y_hat = self.fc(y_hat)
        return y_hat

def timegan(ori_data, parameters, checkpoint_file='checkpoint.pth'):
    """TimeGAN function.

    Args:
       - ori_data: original time-series data
       - parameters: A dataframe or dict with 4 values corresponding to headings: 
         ['hidden_dim', 'num_layer', 'iterations','batch_size', 'z_dim', 'gamma']
         where iterations = num epochs, z_dim = ..., gamma = ...
       - checkpoint_file: Filename of the checkpoint to load. If None, starts from scratch.
    """
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape       #works with seq_len=1
    
    # Maximum sequence length and each sequence length
    ori_time, max_seq_len = extract_time(ori_data)      #works with seq_len=1
    
    # Normalization
    normed_data, min_val, max_val = MinMaxScaler(ori_data)      #works now, originally had nan problem
    
    # Network Parameters
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    z_dim = dim
    gamma = 1
    
    # Model Initialization
    embedder = Embedder(dim, hidden_dim, num_layers).to(device)
    recovery = Recovery(hidden_dim, dim, num_layers).to(device)
    generator = Generator(z_dim, hidden_dim, num_layers).to(device)
    supervisor = Supervisor(hidden_dim, num_layers).to(device)
    discriminator = Discriminator(hidden_dim, num_layers).to(device)
    
    # Combined optimizer for both embedder and recovery networks
    er_combined_params = list(embedder.parameters()) + list(recovery.parameters())
    er_optimizer = optim.Adam(er_combined_params)

    generator_optimizer = optim.Adam(generator.parameters())
    supervisor_optimizer = optim.Adam(supervisor.parameters())
    discriminator_optimizer = optim.Adam(discriminator.parameters())
    
    #COMBINED GENERATOR AND SUPERVISOR OPTIMIZER - BACKPROP THROUGH BOTH NETS In Phase 2
    gs_combined_params = list(generator.parameters()) + list(supervisor.parameters())
    gs_optimizer = optim.Adam(gs_combined_params)
    
    # DataLoader
    normed_data = torch.tensor(normed_data, dtype=torch.float32).to(device)
    ori_time = torch.tensor(ori_time, dtype=torch.int32).to(device)
    dataset = TensorDataset(normed_data, ori_time)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Load checkpoint if one exists
    start_epoch = {'embedding': 0, 'supervisor': 0, 'joint': 0}
    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint:
        #if checkpoint_file and os.path.exists(checkpoint_file):
        start_epoch = checkpoint['epoch']
        embedder.load_state_dict(checkpoint['model_state_dict']['embedder'])
        recovery.load_state_dict(checkpoint['model_state_dict']['recovery'])
        generator.load_state_dict(checkpoint['model_state_dict']['generator'])
        supervisor.load_state_dict(checkpoint['model_state_dict']['supervisor'])
        discriminator.load_state_dict(checkpoint['model_state_dict']['discriminator'])
        er_optimizer.load_state_dict(checkpoint['optimizer_state_dict']['er_optimizer'])
        generator_optimizer.load_state_dict(checkpoint['optimizer_state_dict']['generator_optimizer'])
        supervisor_optimizer.load_state_dict(checkpoint['optimizer_state_dict']['supervisor_optimizer'])
        discriminator_optimizer.load_state_dict(checkpoint['optimizer_state_dict']['discriminator_optimizer'])
        print(f"Resumed training from epoch {start_epoch}")

    # Training Loop
    print('Start Embedding Network Training')
    for itt in range(start_epoch['embedding'], iterations):
        for X_mb, T_mb in dataloader:
            X_mb, T_mb = X_mb.to(device), T_mb.to(device)
            er_optimizer.zero_grad()
            H = embedder(X_mb)
            X_hat = recovery(H)
            E_loss_T0 = nn.functional.mse_loss(X_hat, X_mb)
            E_loss0 = 10 * torch.sqrt(E_loss_T0)
            E_loss0.backward()
            er_optimizer.step()
        
        if itt % 10 == 0:
            print(f'step: {itt}/{iterations}, e_loss: {E_loss0.item()}')
        
        # Save checkpoint every 100 epochs
        if (itt + 1) % 100 == 0:
            save_checkpoint(
                {'embedding': itt + 1, 'supervisor': 0, 'joint': 0},
                {
                    'embedder': embedder,
                    'recovery': recovery,
                    'generator': generator,
                    'supervisor': supervisor,
                    'discriminator': discriminator
                },
                {
                    'er_optimizer': er_optimizer,
                    'generator_optimizer': generator_optimizer,
                    'supervisor_optimizer': supervisor_optimizer,
                    'gs_optimizer': gs_optimizer,
                    'discriminator_optimizer': discriminator_optimizer
                },
                {'E_loss0': E_loss0.item()},
                filename = checkpoint_file
            )
    
    print('Finish Embedding Network Training')
    
    print('Start Training with Supervised Loss Only')
    for itt in range(start_epoch['supervisor'], iterations):
        for X_mb, T_mb in dataloader:
            X_mb, T_mb = X_mb.to(device), T_mb.to(device)
            #generator_optimizer.zero_grad()
            #supervisor_optimizer.zero_grad()
            gs_optimizer.zero_grad()
            H = embedder(X_mb)
            H_supervise = supervisor(H)
            G_loss_S = nn.functional.mse_loss(H[:, 1:, :], H_supervise[:, :-1, :])
            G_loss_S.backward()
            #generator_optimizer.step()
            #supervisor_optimizer.step()
            gs_optimizer.step()
        
        if itt % 10 == 0:
            print(f'step: {itt}/{iterations}, s_loss: {G_loss_S.item()}')
        
        # Save checkpoint every 100 epochs
        if (itt + 1) % 100 == 0:
            save_checkpoint(
                {'embedding': iterations, 'supervisor': itt + 1, 'joint': 0},
                {
                    'embedder': embedder,
                    'recovery': recovery,
                    'generator': generator,
                    'supervisor': supervisor,
                    'discriminator': discriminator
                },
                {
                    'er_optimizer': er_optimizer,
                    'generator_optimizer': generator_optimizer,
                    'supervisor_optimizer': supervisor_optimizer,
                    'gs_optimizer': gs_optimizer,
                    'discriminator_optimizer': discriminator_optimizer
                },
                {'G_loss_S': G_loss_S.item()},
                filename = checkpoint_file
            )
    
    print('Finish Training with Supervised Loss Only')

    print('Start Joint Training')
    for itt in range(start_epoch['joint'], iterations):
        for kk in range(2): 
            for X_mb, T_mb in dataloader:       #training gen
                X_mb, T_mb = X_mb.to(device), T_mb.to(device)
                generator_optimizer.zero_grad()
                #er_optimizer.zero_grad()
                Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
                Z_mb = torch.tensor(Z_mb, dtype=torch.float32).to(device)
                H_hat = generator(Z_mb)
                H_hat_supervise = supervisor(H_hat)     #generate latent
                X_hat = recovery(H_hat_supervise)     #recover to feature space

                # Calculate mean and variance for original and generated data
                mean_X = torch.mean(X_mb, dim=0)
                var_X = torch.var(X_mb, dim=0, unbiased=False)
                mean_X_hat = torch.mean(X_hat, dim=0)
                var_X_hat = torch.var(X_hat, dim=0, unbiased=False)

                # Calculate G_loss_V1 (difference in standard deviation)
                G_loss_V1 = torch.mean(torch.abs(torch.sqrt(var_X_hat + 1e-6) - torch.sqrt(var_X + 1e-6)))

                # Calculate G_loss_V2 (difference in mean)
                G_loss_V2 = torch.mean(torch.abs(mean_X_hat - mean_X))
                G_loss_V = G_loss_V1 + G_loss_V2        #sum V1 and V2

                #One Y_fake for raw output from Generator, Another for Supervised output
                Y_fake_gen = discriminator(H_hat)       #raw from gen
                Y_fake = discriminator(H_hat_supervise)     #supervised
                G_loss_U = nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.ones_like(Y_fake))
                G_loss_U_gen = nn.functional.binary_cross_entropy_with_logits(Y_fake_gen, torch.ones_like(Y_fake_gen))

                G_loss_S = nn.functional.mse_loss(embedder(X_mb)[:, 1:, :], H_hat_supervise[:, :-1, :])
                
                #Step Generator
                G_loss = G_loss_U + gamma * G_loss_U_gen + 100*(torch.sqrt(G_loss_S) + G_loss_V)
                G_loss.backward()
                generator_optimizer.step()

            
                # Embedder training loop
            for X_mb, T_mb in dataloader:
                X_mb, T_mb = X_mb.to(device), T_mb.to(device)
                er_optimizer.zero_grad()
                H = embedder(X_mb)
                X_tilde = recovery(H)
                #Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)       #new
                #Z_mb = torch.tensor(Z_mb, dtype=torch.float32).to(device)       #new
                #H_hat = generator(Z_mb)     #new
                #H_hat_supervise = supervisor(H_hat) #new
                #G_loss_S = nn.functional.mse_loss(H[:, 1:, :], H_hat_supervise[:, :-1, :]) #new

                E_loss_T0 = nn.functional.mse_loss(X_tilde, X_mb)
                E_loss0 = 10 * torch.sqrt(E_loss_T0)        #new
                #E_loss = E_loss0 + 0.1*G_loss_S         #gets backprop gradient info from gen/sup net? new
                E_loss0.backward()
                #E_loss.backward()
                er_optimizer.step()

        
        for X_mb, T_mb in dataloader:
            X_mb, T_mb = X_mb.to(device), T_mb.to(device)
            discriminator_optimizer.zero_grad()
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            Z_mb = torch.tensor(Z_mb, dtype=torch.float32).to(device)
            H_hat = generator(Z_mb)
            H_hat_supervise = supervisor(H_hat)

            #Insert Noise into discrim
            H_real = embedder(X_mb)
            noise_real = torch.normal(mean=0, std=0.2, size=H_real.size()).to(device)
            noise_fake = torch.normal(mean=0, std=0.2, size=Y_fake.size()).to(device)
            noisyY_fake = discriminator(H_hat_supervise + noise_fake)
            noisyY_fake_gen = discriminator(H_hat + noise_fake)
            noisyY_real = discriminator(embedder(X_mb) + noise_real)

            #add noise for discrim loss inputs (testing)
            # noise_real = torch.normal(mean=0, std=0.2, size=Y_real.size()).to(device)
            # noise_fake = torch.normal(mean=0, std=0.2, size=Y_fake.size()).to(device)
            # noisyY_real = Y_real + noise_real
            # noisyY_fake = Y_fake + noise_fake
            # noisyY_fake_gen = Y_fake_gen + noise_fake
            
            #Again. one loss for raw outputs from gen, another for supervised outputs
            #D_loss_real = nn.functional.binary_cross_entropy_with_logits(Y_real, torch.ones_like(Y_real))
            #D_loss_fake_gen = nn.functional.binary_cross_entropy_with_logits(Y_fake_gen, torch.zeros_like(Y_fake_gen))
            #D_loss_fake = nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.zeros_like(Y_fake))
            
            #testing with noise in loss for all discrim inputs
            D_loss_real = nn.functional.binary_cross_entropy_with_logits(noisyY_real, torch.ones_like(noisyY_real))
            D_loss_fake_gen = nn.functional.binary_cross_entropy_with_logits(noisyY_fake_gen, torch.zeros_like(Y_fake_gen))
            D_loss_fake = nn.functional.binary_cross_entropy_with_logits(noisyY_fake, torch.zeros_like(Y_fake))
            D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_gen

            if D_loss.item() > 0.15:
                discriminator_optimizer.zero_grad()
                D_loss.backward()
                discriminator_optimizer.step()    

        
        if itt % 10 == 0:
            print(f'step: {itt}/{iterations}, d_loss: {D_loss.item()}, g_loss_u: {G_loss_U.item()}, g_loss_s: {G_loss_S.item()}, g_loss_v: {G_loss_V.item()}')
        
        # Save checkpoint every 100 epochs
        if (itt + 1) % 100 == 0:
            save_checkpoint(
                {'embedding': iterations, 'supervisor': iterations, 'joint': itt + 1},
                {
                    'embedder': embedder,
                    'recovery': recovery,
                    'generator': generator,
                    'supervisor': supervisor,
                    'discriminator': discriminator
                },
                {
                    'er_optimizer': er_optimizer,
                    'generator_optimizer': generator_optimizer,
                    'supervisor_optimizer': supervisor_optimizer,
                    'gs_optimizer': gs_optimizer,
                    'discriminator_optimizer': discriminator_optimizer
                },
                {
                    'D_loss': D_loss.item(),
                    'G_loss_U': G_loss_U.item(),
                    'G_loss_S': G_loss_S.item(),
                    'G_loss_V': G_loss_V.item()
                },
                filename = checkpoint_file
            )
    
    print('Finish Joint Training')
    
    # Generate random latent variables
    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
    #Z_mb = np.array(Z_mb)
    Z_mb = torch.tensor(Z_mb, dtype=torch.float32).to(device)
    
    # Generate synthetic data
    with torch.no_grad():
        H_hat = generator(Z_mb)
        H_hat_supervise = supervisor(H_hat)
        generated_data_curr = recovery(H_hat_supervise).cpu().numpy()
    
    # Post-process generated data
    generated_data = []
    for i in range(no):
        temp = generated_data_curr[i, :ori_time[i], :]
        generated_data.append(temp)
    
    # Renormalization
    generated_data = np.array([(data * max_val) + min_val for data in generated_data])
    
    return generated_data, embedder, recovery, supervisor, discriminator, generator

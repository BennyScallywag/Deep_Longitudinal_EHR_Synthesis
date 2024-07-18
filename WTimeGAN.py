import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from torch_utils import extract_time, rnn_cell, random_generator, batch_generator

def MinMaxScaler(data):
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val
    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)
    return norm_data, min_val, max_val

def load_checkpoint(filename='checkpoint.pth'):
    script_dir = os.path.dirname(__file__)
    checkpoint_path = os.path.join(script_dir, 'Checkpoints', filename)
    if os.path.exists(checkpoint_path):
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint
    return None

def save_checkpoint(epoch, model_dict, optimizer_dict, losses, filename='checkpoint.pth'):
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'Checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': {k: v.state_dict() for k, v in model_dict.items()},
        'optimizer_state_dict': {k: v.state_dict() for k, v in optimizer_dict.items()},
        'losses': losses
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

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
        h = self.fc(h)
        return h

class Recovery(nn.Module):
    def __init__(self, hidden_dim, output_size, num_layers):
        super(Recovery, self).__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, h):
        x_tilde, _ = self.rnn(h)
        x_tilde = self.fc(x_tilde)
        return x_tilde

class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_layers):
        super(Generator, self).__init__()
        self.rnn = nn.GRU(z_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z):
        e, _ = self.rnn(z)
        e = self.fc(e)
        return e

class Supervisor(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(Supervisor, self).__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h):
        s, _ = self.rnn(h)
        s = self.fc(s)
        return s

class Critic(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(Critic, self).__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        y_hat, _ = self.rnn(h)
        y_hat = self.fc(y_hat)
        return y_hat

def gradient_penalty(critic, real_data, fake_data, device):
    alpha = torch.rand(real_data.size(0), 1, 1).to(device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.requires_grad_(True).to(device)
    critic_interpolates = critic(interpolates)
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)

def w_timegan(ori_data, parameters, checkpoint_file='checkpoint.pth'):
    no, seq_len, dim = np.asarray(ori_data).shape
    ori_time, max_seq_len = extract_time(ori_data)
    normed_data, min_val, max_val = MinMaxScaler(ori_data)
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    z_dim = dim
    gamma = 1
    
    embedder = Embedder(dim, hidden_dim, num_layers).to(device)
    recovery = Recovery(hidden_dim, dim, num_layers).to(device)
    generator = Generator(z_dim, hidden_dim, num_layers).to(device)
    supervisor = Supervisor(hidden_dim, num_layers).to(device)
    critic = Critic(hidden_dim, num_layers).to(device)
    
    er_combined_params = list(embedder.parameters()) + list(recovery.parameters())
    er_optimizer = optim.RMSprop(er_combined_params, lr=1e-5)
    generator_optimizer = optim.RMSprop(generator.parameters(), lr=1e-5)
    supervisor_optimizer = optim.RMSprop(supervisor.parameters(), lr=1e-5)
    critic_optimizer = optim.RMSprop(critic.parameters(), lr=1e-5)
    
    gs_combined_params = list(generator.parameters()) + list(supervisor.parameters())
    gs_optimizer = optim.RMSprop(gs_combined_params, lr=1e-4)
    
    normed_data = torch.tensor(normed_data, dtype=torch.float32).to(device)
    ori_time = torch.tensor(ori_time, dtype=torch.int32).to(device)
    dataset = TensorDataset(normed_data, ori_time)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    start_epoch = {'embedding': 0, 'supervisor': 0, 'joint': 0}
    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint:
        start_epoch = checkpoint['epoch']
        embedder.load_state_dict(checkpoint['model_state_dict']['embedder'])
        recovery.load_state_dict(checkpoint['model_state_dict']['recovery'])
        generator.load_state_dict(checkpoint['model_state_dict']['generator'])
        supervisor.load_state_dict(checkpoint['model_state_dict']['supervisor'])
        critic.load_state_dict(checkpoint['model_state_dict']['critic'])
        er_optimizer.load_state_dict(checkpoint['optimizer_state_dict']['er_optimizer'])
        generator_optimizer.load_state_dict(checkpoint['optimizer_state_dict']['generator_optimizer'])
        supervisor_optimizer.load_state_dict(checkpoint['optimizer_state_dict']['supervisor_optimizer'])
        critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict']['critic_optimizer'])
        print(f"Resumed training from epoch {start_epoch}")

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
        
        if (itt + 1) % 100 == 0:
            save_checkpoint(
                {'embedding': itt + 1, 'supervisor': 0, 'joint': 0},
                {
                    'embedder': embedder,
                    'recovery': recovery,
                    'generator': generator,
                    'supervisor': supervisor,
                    'critic': critic
                },
                {
                    'er_optimizer': er_optimizer,
                    'generator_optimizer': generator_optimizer,
                    'supervisor_optimizer': supervisor_optimizer,
                    'gs_optimizer': gs_optimizer,
                    'critic_optimizer': critic_optimizer
                },
                {'E_loss0': E_loss0.item()},
                filename=checkpoint_file
            )
    
    print('Finish Embedding Network Training')
    
    print('Start Training with Supervised Loss Only')
    for itt in range(start_epoch['supervisor'], iterations):
        for X_mb, T_mb in dataloader:
            X_mb, T_mb = X_mb.to(device), T_mb.to(device)
            gs_optimizer.zero_grad()
            H = embedder(X_mb)
            H_supervise = supervisor(H)
            G_loss_S = nn.functional.mse_loss(H[:, 1:, :], H_supervise[:, :-1, :])
            G_loss_S.backward()
            gs_optimizer.step()
        
        if itt % 10 == 0:
            print(f'step: {itt}/{iterations}, s_loss: {G_loss_S.item()}')
        
        if (itt + 1) % 100 == 0:
            save_checkpoint(
                {'embedding': iterations, 'supervisor': itt + 1, 'joint': 0},
                {
                    'embedder': embedder,
                    'recovery': recovery,
                    'generator': generator,
                    'supervisor': supervisor,
                    'critic': critic
                },
                {
                    'er_optimizer': er_optimizer,
                    'generator_optimizer': generator_optimizer,
                    'supervisor_optimizer': supervisor_optimizer,
                    'gs_optimizer': gs_optimizer,
                    'critic_optimizer': critic_optimizer
                },
                {'G_loss_S': G_loss_S.item()},
                filename=checkpoint_file
            )
    
    print('Finish Training with Supervised Loss Only')

    print('Start Joint Training')
    for itt in range(start_epoch['joint'], iterations):
        for kk in range(5):
            for X_mb, T_mb in dataloader:
                X_mb, T_mb = X_mb.to(device), T_mb.to(device)
                generator_optimizer.zero_grad()
                Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
                Z_mb = torch.tensor(Z_mb, dtype=torch.float32).to(device)
                H_hat = generator(Z_mb)
                H_hat_supervise = supervisor(H_hat)
                X_hat = recovery(H_hat_supervise)

                mean_X = torch.mean(X_mb, dim=0)
                var_X = torch.var(X_mb, dim=0, unbiased=False)
                mean_X_hat = torch.mean(X_hat, dim=0)
                var_X_hat = torch.var(X_hat, dim=0, unbiased=False)

                G_loss_V1 = torch.mean(torch.abs(torch.sqrt(var_X_hat + 1e-6) - torch.sqrt(var_X + 1e-6)))
                G_loss_V2 = torch.mean(torch.abs(mean_X_hat - mean_X))
                G_loss_V = G_loss_V1 + G_loss_V2

                Y_fake_gen = critic(H_hat)
                Y_fake = critic(H_hat_supervise)
                G_loss_U = -torch.mean(Y_fake)
                G_loss_U_gen = -torch.mean(Y_fake_gen)

                G_loss_S = nn.functional.mse_loss(embedder(X_mb)[:, 1:, :], H_hat_supervise[:, :-1, :])

                G_loss = G_loss_U + gamma * G_loss_U_gen + 100 * (torch.sqrt(G_loss_S) + G_loss_V)
                G_loss.backward()
                generator_optimizer.step()

            for X_mb, T_mb in dataloader:
                X_mb, T_mb = X_mb.to(device), T_mb.to(device)
                er_optimizer.zero_grad()
                H = embedder(X_mb)
                X_tilde = recovery(H)

                E_loss_T0 = nn.functional.mse_loss(X_tilde, X_mb)
                E_loss0 = 10 * torch.sqrt(E_loss_T0)
                E_loss0.backward()
                er_optimizer.step()

        for X_mb, T_mb in dataloader:
            X_mb, T_mb = X_mb.to(device), T_mb.to(device)
            critic_optimizer.zero_grad()
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            Z_mb = torch.tensor(Z_mb, dtype=torch.float32).to(device)
            H_hat = generator(Z_mb)
            H_hat_supervise = supervisor(H_hat)

            H_real = embedder(X_mb)
            noise_real = torch.normal(mean=0, std=0.2, size=H_real.size()).to(device)
            noise_fake = torch.normal(mean=0, std=0.2, size=H_hat_supervise.size()).to(device)
            Y_fake = critic(H_hat_supervise + noise_fake)
            Y_fake_gen = critic(H_hat + noise_fake)
            Y_real = critic(H_real + noise_real)

            D_loss_real = -torch.mean(Y_real)
            D_loss_fake_gen = torch.mean(Y_fake_gen)
            D_loss_fake = torch.mean(Y_fake)

            gp = gradient_penalty(critic, embedder(X_mb).detach(), H_hat_supervise.detach(), device)
            D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_gen + 10 * gp

            D_loss.backward()
            critic_optimizer.step()
        
        if itt % 10 == 0:
            print(f'step: {itt}/{iterations}, c_loss: {D_loss.item()}, g_loss_u: {G_loss_U.item()}, g_loss_s: {G_loss_S.item()}, g_loss_v: {G_loss_V.item()}')
        
        if (itt + 1) % 100 == 0:
            save_checkpoint(
                {'embedding': iterations, 'supervisor': iterations, 'joint': itt + 1},
                {
                    'embedder': embedder,
                    'recovery': recovery,
                    'generator': generator,
                    'supervisor': supervisor,
                    'critic': critic
                },
                {
                    'er_optimizer': er_optimizer,
                    'generator_optimizer': generator_optimizer,
                    'supervisor_optimizer': supervisor_optimizer,
                    'gs_optimizer': gs_optimizer,
                    'critic_optimizer': critic_optimizer
                },
                {
                    'D_loss': D_loss.item(),
                    'G_loss_U': G_loss_U.item(),
                    'G_loss_S': G_loss_S.item(),
                    'G_loss_V': G_loss_V.item()
                },
                filename=checkpoint_file
            )
    
    print('Finish Joint Training')
    
    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
    Z_mb = torch.tensor(Z_mb, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        H_hat = generator(Z_mb)
        H_hat_supervise = supervisor(H_hat)
        generated_data_curr = recovery(H_hat_supervise).cpu().numpy()
    
    generated_data = []
    for i in range(no):
        temp = generated_data_curr[i, :ori_time[i], :]
        generated_data.append(temp)
    
    generated_data = np.array([(data * max_val) + min_val for data in generated_data])
    
    return generated_data, embedder, recovery, supervisor, critic, generator

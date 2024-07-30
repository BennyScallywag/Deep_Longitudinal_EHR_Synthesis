import torch
import torch.nn as nn

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
        #also try rnn(hidden --> output) then linear(output-->output)

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
#Note: IN THIS IMPLEMENTATION THE DISCRIM PRODUCES A SCORE FOR EACH TIME STEP IN THE SEQUENCE
#i.e. if i pass in 10 sequences each of length 24, the output of the discrim will have shape (10, 24)
class Discriminator(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        #self.fc = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, hidden_dim)        #temporarily changed back to normal version
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        y_hat, _ = self.rnn(h)
        y_hat = self.fc(y_hat)
        y_hat = self.sigmoid(y_hat)
        return y_hat
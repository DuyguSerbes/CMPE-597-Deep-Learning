import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
kernel_size = 4
image_channels = 1 # MNIST images are grayscale
latent_dim = 2 # latent dimension for sampling
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 1
num_classes = 2


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_log_var = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        # One layer LSTM
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        out, _ = self.lstm(x, (h0, c0))
        hidden = out[:, -1, :]  # (batch, seq, num_classes)

        # mu and log_var from 2 different fully connected layer
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc2 = nn.Linear(latent_dim, 64 * 7 * 7)
        # decoder
        # 4 layer transpose CNN
        self.dec1 = nn.ConvTranspose2d(in_channels=64, out_channels=56, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(in_channels=56, out_channels=28, kernel_size=4, stride=1, padding=1)
        self.dec3 = nn.ConvTranspose2d(in_channels=28, out_channels=14, kernel_size=4, stride=2, padding=2)
        self.dec4 = nn.ConvTranspose2d(in_channels=14, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        x = self.fc2(x)
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))

        return reconstruction

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Consists of encoder and decoder
        self.encoder=Encoder()
        self.decoder=Decoder()

    def reparameterize(self, mu, log_var):
        # mu is mean from encoder latent space
        # log_var means log variance from encoder latent space
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # epsilon
        sample = mu + (eps * std)  # sampling

        return sample

    def forward(self, x):
        mu, log_var= self.encoder(x)
        x = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(x)
        return reconstruction, mu, log_var

def final_loss(bce_loss, mu, logvar):
    # Binary cross entropy and KL divergence
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD , BCE + KLD
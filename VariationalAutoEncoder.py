import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalEncoder(nn.Module):
    def __init__(self):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(270, 270)
        self.linear2 = nn.Linear(270, 54)
        self.linear3 = nn.Linear(270, 54)
        
        self.kl = 0
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).mean()
        
        return z
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(54, 270)
        self.linear2 = nn.Linear(270, 270)
        
    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
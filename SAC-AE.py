#Soft Actor-Critic with Auto-Encoder
'''
Combines SAC with unsupervised latent representation learning using autoencoders, significantly enhancing sample efficiency and stability by learning compressed latent states.
The autoencoder has two parts:
- Encoder z = f_phi(s): Maps high-dimensional input s to a latent representation z.
- Decoder s' = g_psi(z): Reconstructs input s from latent representation z.

Training AE is done by minimizing reconstruction loss:
    L_AE = (phi, psi) = ||s-g_psi(f_phi(s))||^2

SAC-AE optimizizes the following modified objective:
- Original SAC entropy-regularized obj:
    J(pi) = E[sum_t(r(s_t, a_t) + alpha * H(pi(.|s_t)))]
- Modified SAC-AE obj:
    z = f_phi(s_t), J_AE(pi, f_phi) = E[sum_t(gamma^t * (r(s_t, a_t) + alpha * H(pi(.|s_t))]

The Actor and critic now operate on latent representation z_t.

'''

#Implementation
import torch
import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class Autoencoder(nn.Module):
    def __init__(self, latent_dim = 50):
        super().__init__()

        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3,32,4,stride=2),
            nn.ReLU(),
            nn.Conv2d(32,64,4,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,128,4,stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*10*10, latent_dim)
        )

        #Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128*10*10),
            nn.ReLU(),
            nn.Unflatten(1, (128,10,10)),
            nn.ConvTranspose2d(128,64,5,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,5,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32,3,6,stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z
    
class Actor(nn.Module):
    def __init__(self, latent_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, z):
        x = self.net(z)
        mu, log_std = self.mu(x), self.log_std(x)
        std = torch.exp(log_std)
        return mu, std
    
    def sample(self, z):
        mu, std = self(z)
        dist = torch.distributions.Normal(mu, std)
        action = torch.tanh(dist.rsample()) * self.max_action
        return action, dist.log_prob(action).sum(-1)
    


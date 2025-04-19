# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 13:54:43 2025

@author: noa
"""

import torch
import torch.nn as nn
from gmm_prior import GMM_Prior

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # [28x28] -> [14x14]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # [14x14] -> [7x7]
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # [7x7] -> [14x14]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),  # [14x14] -> [28x28]
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 64, 7, 7)
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self, latent_dim, gmm_components=10):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.gmm_prior = GMM_Prior(latent_dim, gmm_components)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z
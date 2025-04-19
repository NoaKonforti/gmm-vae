# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 17:34:56 2025

@author: noa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GMM_Prior(nn.Module):
    def __init__(self, latent_dim, n_components):
        super().__init__()
        self.K = n_components
        self.latent_dim = latent_dim

        # Learnable GMM params
        self.mu_k = nn.Parameter(torch.randn(self.K, latent_dim))              # means
        self.log_var_k = nn.Parameter(torch.zeros(self.K, latent_dim))         # log variances
        self.pi_k_logits = nn.Parameter(torch.zeros(self.K))                   # logits for softmax

    def get_pi(self):
        return F.softmax(self.pi_k_logits, dim=0)  # normalized weights

    def log_prob(self, z):
        """
        Compute log p(z) where p(z) = sum_k pi_k N(z | mu_k, var_k)
        """
        z = z.unsqueeze(1)  # [B, 1, D]
        mu = self.mu_k.unsqueeze(0)  # [1, K, D]
        var = torch.exp(self.log_var_k).unsqueeze(0)  # [1, K, D]
        pi = self.get_pi().unsqueeze(0)  # [1, K]

        # Log probability per component
        log_p_k = -0.5 * (((z - mu)**2) / var + self.log_var_k + torch.log(torch.tensor(2 * torch.pi))).sum(-1)  # [B, K]

        log_p = torch.logsumexp(log_p_k + torch.log(pi), dim=1)  # [B]
        return log_p

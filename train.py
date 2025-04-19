# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 13:54:46 2025

@author: noa
"""

import torch
import torch.nn.functional as F
import numpy as np
from evaluate import gmm_component_accuracy
import matplotlib.pyplot as plt


def reconstruction_loss(x, recon):
    return F.binary_cross_entropy(recon, x, reduction='sum')

def separation_loss(mu_k, log_var_k, min_ratio=3.0):
    """
    Encourages separation between GMM components relative to their variances.
    Penalizes components whose distance is smaller than min_ratio * average sigma.
    """
    dists = torch.cdist(mu_k, mu_k, p=2)  # Pairwise Euclidean distances [K, K]
    sigmas = torch.exp(0.5 * log_var_k)   # Standard deviation [K, D]

    sigma_mean = sigmas.mean(dim=1, keepdim=True)  # [K, 1]
    avg_sigma = (sigma_mean + sigma_mean.T) / 2    # [K, K]

    ratio = dists / (avg_sigma + 1e-8)  # Ratio between distance and sigma

    mask = (ratio < min_ratio).float()  # Identify where the ratio is too small
    penalty = (min_ratio - ratio) * mask  # Penalize those pairs

    return penalty.sum() / (mu_k.shape[0] ** 2)


def compute_kl_gmm(mu, logvar, gmm, n_samples=1):
    """
    KL divergence between q(z|x) ~ N(mu, sigma^2) and p(z) ~ GMM
    Uses Monte Carlo estimation with reparameterization trick.
    """
    B, D = mu.shape
    std = torch.exp(0.5 * logvar)
    
    eps = torch.randn_like(std)
    z_sampled = mu + eps * std  # sampled z ~ q(z|x)
        
    #Compute log_q(z|x)
    log_qzx = -0.5 * ((eps ** 2) + logvar + torch.log(torch.tensor(2 * torch.pi, device=logvar.device))).sum(dim=1)
        
    #Compute log_p(z) from trainable GMM prior
    log_pz = gmm.log_prob(z_sampled) 
        
    #KL divergence (Monte Carlo approx)
    kl = (log_qzx - log_pz).sum()  # scalar
    return kl


def train_gmm_vae(model, train_loader, device, epochs=10, beta=5e-1, gamma = 10.0):
    """
    Trains the GMM-VAE model on MNIST.
    
    Parameters
    ----------
    model : nn.Module
        GMM-VAE model instance.
    train_loader : DataLoader
        PyTorch DataLoader for training data.
    device : torch.device
        Training device (CPU or GPU).
    epochs : int, optional
        Number of training epochs. Default is 10.
    beta : float, optional
        Weight for KL divergence term. Default is 0.5.
    gamma : float, optional
        Regularization strength of the seperation loss. Default is 10.0.
    
    Returns
    -------
    None
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    accuracy_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()

            recon, mu, logvar, z = model(x)

            #Reconstruction Loss
            rec_loss = reconstruction_loss(x, recon)

            #KL divergence (Monte Carlo approx)
            kl = compute_kl_gmm(mu, logvar, model.gmm_prior, n_samples=1)
          
            #Separation regularization
            sep_loss = separation_loss(model.gmm_prior.mu_k, model.gmm_prior.log_var_k, min_ratio=3.0)
            
            #Total loss
            loss = rec_loss + beta * kl + gamma * sep_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch + 1}] Total Loss: {total_loss:.2f}")
        evaluate_training(model, train_loader, device, accuracy_history)

def evaluate_training(model, train_loader, device, accuracy_history):
    model.eval()
    zs = []
    labels = []

    with torch.no_grad():
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            _, _, _, z = model(x_batch)
            zs.append(z.cpu().numpy())
            labels.extend(y_batch.numpy())

    zs = np.concatenate(zs)
    labels = np.array(labels)
    z_tensor = torch.tensor(zs, dtype=torch.float32).to(device)

    # GMM responsibilities
    z_exp = z_tensor.unsqueeze(1)
    mu_k = model.gmm_prior.mu_k.unsqueeze(0)
    var_k = torch.exp(model.gmm_prior.log_var_k).unsqueeze(0)
    pi_k = model.gmm_prior.get_pi().unsqueeze(0)

    log_p_k = -0.5 * (((z_exp - mu_k) ** 2) / var_k + model.gmm_prior.log_var_k + np.log(2 * np.pi))
    log_p_k = log_p_k.sum(-1) + torch.log(pi_k)
    responsibilities = torch.softmax(log_p_k, dim=1)
    assignments = responsibilities.argmax(dim=1).cpu().numpy()

    # Accuracy
    acc = gmm_component_accuracy(labels, assignments)
    accuracy_history.append(acc)

    # Plot curve
    plot_accuracy_curve(accuracy_history)


def plot_accuracy_curve(accuracy_history):
    plt.figure(figsize=(8, 5))
    plt.plot(accuracy_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Unsupervised Accuracy")
    plt.title("GMM-VAE Accuracy vs. Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_vs_epoch.png")
    plt.show()
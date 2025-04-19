# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 14:08:02 2025

@author: noa
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import os
current_directory = os.getcwd()
sys.path.append(current_directory)
from GMM_VAE_model import VAE
from train import train_gmm_vae
from evaluate import evaluate_latents

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 8
N_COMPONENTS = 10
BATCH_SIZE = 128
EPOCHS = 2#150

def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_data = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root="./data", train=False, transform=transform)
    return train_data, test_data

def main():
    train_dataset, test_dataset = load_mnist()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = VAE(latent_dim=LATENT_DIM, gmm_components=N_COMPONENTS).to(device)

    print("Training GMM-VAE...")
    train_gmm_vae(model=model, train_loader=train_loader, device=device, epochs=EPOCHS)

    print("Evaluating latent space clustering...")
    evaluate_latents(model, test_loader, None, device)  # GMM is inside model


if __name__ == "__main__":
    main()

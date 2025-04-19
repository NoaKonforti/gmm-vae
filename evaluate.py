# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 14:10:16 2025

@author: noa
"""

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score


def visualize_tsne(zs, labels):
    print("Running t-SNE on latent space...")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    z_2d = tsne.fit_transform(zs)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="Set3", s=5)
    handles = [mpatches.Patch(color=scatter.cmap(scatter.norm(i)), label=str(i)) for i in range(10)]
    plt.legend(handles=handles, title="Digit Label", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title("t-SNE of Latent Space (colored by true labels)")
    plt.savefig("tsne_latent_space.png")
    plt.show()


def plot_gmm_component_vs_labels(model, data_loader, device):
    model.eval()
    zs = []
    true_labels = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            _, _, _, z = model(x)
            zs.append(z.cpu().numpy())
            true_labels.extend(y.cpu().numpy())

    zs = np.concatenate(zs)  # [N, D]
    true_labels = np.array(true_labels)
    z_tensor = torch.tensor(zs, dtype=torch.float32).to(device)

    # Compute responsibilities
    z_exp = z_tensor.unsqueeze(1)
    mu_k = model.gmm_prior.mu_k.unsqueeze(0)
    var_k = torch.exp(model.gmm_prior.log_var_k).unsqueeze(0)
    pi_k = model.gmm_prior.get_pi().unsqueeze(0)

    log_p_k = -0.5 * (((z_exp - mu_k) ** 2) / var_k + model.gmm_prior.log_var_k + np.log(2 * np.pi))
    log_p_k = log_p_k.sum(-1) + torch.log(pi_k)
    responsibilities = torch.softmax(log_p_k, dim=1)

    assigned = torch.argmax(responsibilities, dim=1).cpu().numpy()  # [N]

    # Build contingency table
    df = pd.DataFrame({'Component': assigned, 'Label': true_labels})
    confusion = pd.crosstab(df['Component'], df['Label'])

    # Plot
    plt.figure(figsize=(10, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap="YlGnBu")
    plt.title("True Labels per GMM Component")
    plt.xlabel("True Digit")
    plt.ylabel("GMM Component")
    plt.tight_layout()
    plt.savefig("gmm_vs_true_labels.png")
    plt.show()    


def gmm_component_accuracy(y_true, gmm_assignments):
    """
    Evaluate unsupervised accuracy by assigning the majority class label to each component.
    """
    from scipy.stats import mode
    import numpy as np

    # Get number of components
    components = np.unique(gmm_assignments)
    
    # Assign most frequent true label to each component
    mapping = {}
    for comp in components:
        idx = gmm_assignments == comp
        if np.sum(idx) > 0:
            most_common_label = mode(y_true[idx], keepdims=True)[0][0]
            mapping[comp] = most_common_label
    
    # Map predicted clusters to labels
    pred_labels = np.vectorize(mapping.get)(gmm_assignments)
    
    return accuracy_score(y_true, pred_labels)

def plot_varying_w(gmm_prior, decoder, device, latent_dim, component_idx=0, grid_size=10):
    mu_k = gmm_prior.mu_k.detach().cpu().numpy()
    var_k = torch.exp(gmm_prior.log_var_k).detach().cpu().numpy()

    mu = mu_k[component_idx]
    std = np.sqrt(var_k[component_idx])

    # Create a 2D grid of variations around the component mean
    z_samples = []
    for i in np.linspace(-2, 2, grid_size):
        for j in np.linspace(-2, 2, grid_size):
            z = mu.copy()
            if latent_dim >= 2:
                z[0] += i * std[0]
                z[1] += j * std[1]
            z_samples.append(z)

    z_tensor = torch.tensor(z_samples, dtype=torch.float32).to(device)
    with torch.no_grad():
        imgs = decoder(z_tensor).cpu().numpy()

    # Plot grid
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            axs[i, j].imshow(imgs[i * grid_size + j].squeeze(), cmap="gray")
            axs[i, j].axis("off")
    plt.suptitle("Generated Samples: Varying w (Style within a component)")
    plt.tight_layout()
    plt.savefig("varying_w.png")
    plt.show()


def plot_varying_z(gmm_prior, decoder, device, latent_dim, n_per_component=10):
    pi = gmm_prior.get_pi().detach().cpu().numpy()
    mu_k = gmm_prior.mu_k.detach().cpu().numpy()
    var_k = torch.exp(gmm_prior.log_var_k).detach().cpu().numpy()
    K = len(pi)

    all_images = []

    for k in range(K):
        images = []
        for _ in range(n_per_component):
            z = np.random.normal(loc=mu_k[k], scale=np.sqrt(var_k[k]))
            z_tensor = torch.from_numpy(z).float().unsqueeze(0).to(device)
            with torch.no_grad():
                img = decoder(z_tensor).cpu().numpy().squeeze()
            images.append(img)
        all_images.append(images)

    # Plot grid
    fig, axs = plt.subplots(K, n_per_component, figsize=(n_per_component, K))
    for i in range(K):
        for j in range(n_per_component):
            axs[i, j].imshow(all_images[i][j], cmap="gray")
            axs[i, j].axis("off")
    plt.suptitle("Generated Samples: Varying z (GMM Component)")
    plt.tight_layout()
    plt.savefig("varying_z.png")
    plt.show()
    
def evaluate_latents(model, data_loader, gmm, device):
    model.eval()
    zs = []
    labels = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            _, _, _, z = model(x)
            zs.append(z.cpu().numpy())
            labels.extend(y.numpy())

    zs = np.concatenate(zs)
    labels = np.array(labels)


    
    with torch.no_grad():
        z_tensor = torch.tensor(zs, dtype=torch.float32).to(device)
        z_exp = z_tensor.unsqueeze(1)  # [N, 1, D]
        mu_k = model.gmm_prior.mu_k.unsqueeze(0)
        var_k = torch.exp(model.gmm_prior.log_var_k).unsqueeze(0)
        pi_k = model.gmm_prior.get_pi().unsqueeze(0)
    
        log_p_k = -0.5 * (((z_exp - mu_k) ** 2) / var_k + model.gmm_prior.log_var_k + np.log(2 * np.pi))
        log_p_k = log_p_k.sum(-1) + torch.log(pi_k)
        responsibilities = torch.softmax(log_p_k, dim=1)
        gmm_pred = responsibilities.argmax(dim=1).cpu().numpy()
    
    ari = adjusted_rand_score(labels, gmm_pred)
    nmi = normalized_mutual_info_score(labels, gmm_pred)

    print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}")

    # t-SNE Visualization
    visualize_tsne(zs, labels)
    
    plot_gmm_component_vs_labels(model, data_loader, device)
        
    acc = gmm_component_accuracy(labels, gmm_pred)
    print(f"Accuracy (via majority vote): {acc:.4f}")
    
    plot_varying_z(model.gmm_prior, model.decoder, device, 8)
    
    plot_varying_w(model.gmm_prior, model.decoder, device, 8, component_idx=0)

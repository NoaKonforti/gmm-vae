# gmm-vae: Variational Autoencoder with Gaussian Mixture Prior
This project implements a **Gaussian Mixture Variational Autoencoder (GMM-VAE)** for unsupervised clustering and image generation on the MNIST dataset. The model replaces the standard normal prior in a VAE with a **learnable Gaussian Mixture Model (GMM)** prior, encouraging structured latent representations aligned with semantic classes (e.g., digit identities).

> This implementation is inspired by [Dilokthanakul et al., 2017](https://arxiv.org/abs/1611.02648).

---

## Features

- Convolutional encoder/decoder architecture
- Trainable GMM prior with K components
- Monte Carlo approximation of KL divergence
- Separation loss to improve cluster distinctness
- Evaluation: ARI, NMI, majority-vote accuracy
- Visualizations: t-SNE, heatmaps, and sample generations

---

## Project Structure
 ``` 
. ├── main.py              # Entry point: training + evaluation
├── train.py               # Training loop and loss functions
├── GMM_VAE_model.py       # VAE architecture with GMM prior
├── gmm_prior.py           # GMM prior module
├── evaluate.py            # Evaluation metrics and plots
├── README.md              # You're here :)
 ``` 

---
## Author

**Noa Konforti**  
MSc student in Electrical Engineering 
[Connect with me on LinkedIn](www.linkedin.com/in/noa-konforti)

import matplotlib.pyplot as plt
import numpy as np

# Example epoch data for synthetic losses and accuracy values
epochs = range(1, 21)

# Total Loss data for each model (replace with actual training loss data)
vae_loss = [0.85, 0.8, 0.75, 0.73, 0.7, 0.68, 0.66, 0.65, 0.63, 0.6, 0.59, 0.58, 0.56, 0.55, 0.53, 0.52, 0.5, 0.49, 0.48, 0.47]
gan_loss = [0.9, 0.85, 0.83, 0.8, 0.78, 0.76, 0.74, 0.72, 0.71, 0.7, 0.68, 0.67, 0.65, 0.64, 0.62, 0.6, 0.58, 0.56, 0.55, 0.54]
hierarchical_vae_gan_loss = [0.75, 0.7, 0.67, 0.65, 0.63, 0.61, 0.6, 0.58, 0.57, 0.56, 0.55, 0.53, 0.52, 0.51, 0.5, 0.48, 0.47, 0.46, 0.45, 0.44]

# Reconstruction Loss data
vae_reconstruction_loss = [0.65, 0.6, 0.57, 0.55, 0.53, 0.51, 0.5, 0.49, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.4, 0.39, 0.38, 0.37, 0.36]
gan_generator_loss = [0.7, 0.68, 0.66, 0.64, 0.63, 0.61, 0.6, 0.59, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52, 0.51, 0.5, 0.49, 0.48, 0.47, 0.46]
hierarchical_vae_gan_reconstruction_loss = [0.6, 0.57, 0.55, 0.53, 0.52, 0.5, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.4, 0.39, 0.38, 0.37, 0.36]

# KL Divergence Loss (for VAE models)
vae_kl_loss = [0.2, 0.18, 0.17, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09, 0.085, 0.08, 0.075, 0.07, 0.065, 0.06, 0.058, 0.055, 0.053, 0.05]
hierarchical_vae_gan_kl_loss = [0.18, 0.16, 0.15, 0.13, 0.12, 0.11, 0.1, 0.095, 0.09, 0.085, 0.08, 0.075, 0.07, 0.065, 0.06, 0.058, 0.055, 0.053, 0.051, 0.05]

# Accuracy data for each model
vae_accuracy = [0.5, 0.55, 0.57, 0.6, 0.62, 0.63, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78]
gan_accuracy = [0.45, 0.5, 0.52, 0.55, 0.57, 0.6, 0.61, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75]
hierarchical_vae_gan_accuracy = [0.52, 0.57, 0.6, 0.63, 0.65, 0.67, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82]

# Plot Overall Model Loss Comparison
plt.figure(figsize=(10, 6))
plt.plot(epochs, vae_loss, label='VAE Total Loss', marker='o', linestyle='-')
plt.plot(epochs, gan_loss, label='GAN Total Loss', marker='s', linestyle='--')
plt.plot(epochs, hierarchical_vae_gan_loss, label='Hierarchical VAE-GAN Total Loss', marker='^', linestyle='-.')
plt.xlabel('Epochs')
plt.ylabel('Overall Model Loss')
plt.title('Overall Model Loss Comparison for VAE, GAN, and Hierarchical VAE-GAN')
plt.legend()
plt.grid(True)
plt.show()

# Plot Reconstruction Loss Comparison
plt.figure(figsize=(10, 6))
plt.plot(epochs, vae_reconstruction_loss, label='VAE Reconstruction Loss', marker='o', linestyle='-')
plt.plot(epochs, gan_generator_loss, label='GAN Generator Loss', marker='s', linestyle='--')
plt.plot(epochs, hierarchical_vae_gan_reconstruction_loss, label='Hierarchical VAE-GAN Reconstruction Loss', marker='^', linestyle='-.')
plt.xlabel('Epochs')
plt.ylabel('Reconstruction Loss')
plt.title('Reconstruction Loss Comparison for VAE, GAN, and Hierarchical VAE-GAN')
plt.legend()
plt.grid(True)
plt.show()

# Plot KL Divergence Loss for VAE and Hierarchical VAE-GAN
plt.figure(figsize=(10, 6))
plt.plot(epochs, vae_kl_loss, label='VAE KL Divergence Loss', marker='o', linestyle='-')
plt.plot(epochs, hierarchical_vae_gan_kl_loss, label='Hierarchical VAE-GAN KL Divergence Loss', marker='^', linestyle='-.')
plt.xlabel('Epochs')
plt.ylabel('KL Divergence Loss')
plt.title('KL Divergence Loss Comparison for VAE and Hierarchical VAE-GAN')
plt.legend()
plt.grid(True)
plt.show()

# Plot Accuracy Comparison for VAE, GAN, and Hierarchical VAE-GAN
plt.figure(figsize=(10, 6))
plt.plot(epochs, vae_accuracy, label='VAE Accuracy', marker='o', linestyle='-')
plt.plot(epochs, gan_accuracy, label='GAN Accuracy', marker='s', linestyle='--')
plt.plot(epochs, hierarchical_vae_gan_accuracy, label='Hierarchical VAE-GAN Accuracy', marker='^', linestyle='-.')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for VAE, GAN, and Hierarchical VAE-GAN')
plt.legend()
plt.grid(True)
plt.show()

# Synthetic Velocity Model Generation using Wasserstein Autoencoder (WAE)

## Overview

This project implements a deep learning-based approach for generating and learning synthetic velocity models using a Wasserstein Autoencoder (WAE).

The project generates 2D synthetic velocity models of size 128 × 128 pixels. Each model contains a background velocity field with randomly generated defects such as circular defects, elliptical defects, and cracks.

The generated velocity models are used to train a WAE that learns a compact latent representation of the velocity structures and can reconstruct and generate synthetic velocity models.

---

## Objectives

The main objectives of this project are:

- Generate synthetic 2D velocity models automatically.
- Introduce different types of structural defects into the velocity models.
- Train a Wasserstein Autoencoder on the generated models.
- Learn a low-dimensional latent representation of velocity models.
- Reconstruct input velocity models using the trained WAE.
- Generate new synthetic velocity models from random latent vectors.
- Visualize training performance and generated velocity structures.

---

## Synthetic Velocity Model Generation

The synthetic models are generated using a uniform background velocity field.

### Model size

128 × 128 pixels
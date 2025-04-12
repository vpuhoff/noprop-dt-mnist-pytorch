# NoProp-PyTorch: An Experimental Implementation of Training Without Backpropagation

## Description

This repository contains an experimental PyTorch implementation exploring the **NoProp** algorithm, presented in the paper "NOPROP: TRAINING NEURAL NETWORKS WITHOUT BACK-PROPAGATION OR FORWARD-PROPAGATION".

The goal of NoProp is to train neural networks without relying on traditional end-to-end backpropagation. Instead, it trains network layers (blocks) independently using a local denoising objective inspired by diffusion models.

**Disclaimer:** This is an educational implementation attempt based on the paper. Achieving stable training and high accuracy required **significant deviations from the original method** described in the paper, particularly regarding the denoising target and classification mechanism. This implementation does not guarantee exact replication of the original work but demonstrates the feasibility of a modified approach.

## Source Paper

* **Title:** NOPROP: TRAINING NEURAL NETWORKS WITHOUT BACK-PROPAGATION OR FORWARD-PROPAGATION
* **Authors:** Qinyu Li, Yee Whye Teh, Razvan Pascanu
* **arXiv:** [https://arxiv.org/abs/2503.24322v1](https://arxiv.org/abs/2503.24322v1) 

## Core Concept of NoProp (and Implementation Differences)

* **Original NoProp Idea:**
    * Network blocks are trained independently.
    * Each block `t` learns to predict a "clean" target label embedding `u_y` from the input `x` and a noised version `z_{t-1}`.
    * Inference uses Equation 3 from the paper.

* **Problems & Modifications in this Implementation:**
    * Training blocks to predict the clean `u_y` proved unstable and ineffective.
    * **Key Change 1:** The objective of the blocks was changed to predict the **added noise $\epsilon$**, similar to standard diffusion models (DDPM, VDM, CARD). Denoising loss became MSE between predicted and actual noise.
    * **Key Change 2:** The classifier was trained on the **predicted clean embedding $\hat{u}_T$** (reconstructed from $z_{T-1}$ and predicted $\hat{\epsilon}_T$), instead of the noisy sample $z_T$.
    * **Key Change 3:** Inference was modified to a **DDPM-like reverse process** using the predicted noise $\epsilon$.

## Implementation Details (Final Working Configuration)

* **Framework:** PyTorch
* **Method:** Modified NoProp-DT (predicting noise $\epsilon$, classifying from $\hat{u}_T$)
* **Dataset:** MNIST
* **Architecture:** Based on Figure 7 (left) from the paper's appendix (CNN for image, MLP for embedding, combined layers). Block output is predicted noise $\epsilon$ (`embed_dim`).
* **Embeddings:** Learned (`dim=20`), orthogonal initialization, embedding norm clipping (`max_norm_embed=50.0`), very low weight decay (`embed_wd=1e-7`).
* **Noise Schedule:** Cosine schedule (`alphas_bar`), T=10 steps.
* **Denoising Loss Weight:** Uniform weight scaled by `ETA_LOSS_WEIGHT=0.5`.
* **Inference:** Implemented DDPM-like reverse steps using predicted $\epsilon$.
* **Optimizer:** AdamW with initial `LR=0.01` for all parameters, `Weight Decay=1e-3` for blocks/classifier.
* **Scheduler:** `CosineAnnealingLR` (`T_max=100`, `eta_min=1e-6`) applied to all optimizers.
* **Stabilization:** Gradient clipping (`max_norm=1.0`).
* **Epochs:** Trained up to 100 epochs with early stopping (`patience=15`).

## Current Status & Experimental Results *(Updated Apr 12, 2025)*

After extensive debugging and significant modifications to the original algorithm (predicting noise $\epsilon$, classifying from $\hat{u}_T$), hyperparameter optimization identified an effective configuration (`LR=0.01`, `ETA=0.5`, `EMBED_WD=1e-7`).

A full training run (up to 100 epochs) was performed using these parameters with a `CosineAnnealingLR` scheduler and early stopping.

* **Training Observations:** The training process was stable. `Classify Loss` decreased significantly, `AvgDenoiseLoss` (MSE on $\epsilon$) remained active and stable, and the embedding norm was controlled.
* **Final Result:** The model achieved a **best test accuracy of 99.01%** on MNIST, reached around epoch 80 before early stopping triggered at epoch 95.

## Conclusions from Experiments

* Reproducing the high accuracy reported in the paper using the method *as described* (predicting $u_y$) was unsuccessful in this implementation.
* Significant **modifications** (changing the prediction target to noise $\epsilon$, altering the classifier input to $\hat{u}_T$) were **essential** to achieve effective and stable learning.
* The modified approach, combined with careful hyperparameter tuning (using HPO), appropriate LR scheduling, and stabilization techniques (clipping), successfully trained a model achieving **high accuracy (99.01%)** on MNIST.
* This result demonstrates the potential of backpropagation-free, diffusion-inspired training methods, although it required deviating significantly from the specific formulation presented in the NoProp paper but aligning more closely with established diffusion model practices (like DDPM, VDM, CARD).
* The small remaining gap to the paper's reported result (>99.4%) might be due to further subtle implementation differences, architectural details, or hyperparameter settings.

## Setup

*(Setup instructions remain the same)*

**Option 1: Using `venv` (Recommended)**
1.  Ensure you have Python 3 installed.
2.  Clone the repository: `git clone <your-repository-url>`
3.  Navigate to the project directory: `cd <your-repository-name>`
4.  Ensure `requirements.txt` exists with the necessary packages.
5.  Run the setup script (creates/updates `noprop_env`):
    ```bash
    chmod +x setup_env.sh
    ./setup_env.sh
    ```
6.  Activate the environment: `source noprop_env/bin/activate`
7.  Run the script (e.g., for a full run): `python noprop_example.py` (make sure `RUN_HPO = False` inside the script).

**Option 2: Manual Installation**
1.  Ensure you have Python 3 installed.
2.  Clone the repository.
3.  Navigate to the project directory.
4.  Create/activate a virtual environment.
5.  Install dependencies: `pip install -r requirements.txt`
6.  Run the script: `python noprop_example.py`
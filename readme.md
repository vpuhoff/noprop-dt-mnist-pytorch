# NoProp-PyTorch: An Experimental Implementation of Training Without Backpropagation

## Description

This repository contains an experimental PyTorch implementation exploring the **NoProp** algorithm, presented in the paper "NOPROP: TRAINING NEURAL NETWORKS WITHOUT BACK-PROPAGATION OR FORWARD-PROPAGATION".

The goal of NoProp is to train neural networks without relying on traditional end-to-end backpropagation. Instead, it trains network layers (blocks) independently using a local denoising objective inspired by diffusion models.

**Disclaimer:** This is an educational implementation attempt based on the paper. It contains **significant deviations from the original method** required to achieve stable training during experimentation. It does not guarantee replication of the results reported in the original work. Reproducing research paper results often requires significant effort in tuning and accounting for precise implementation details not covered in the publication.

## Source Paper

* **Title:** NOPROP: TRAINING NEURAL NETWORKS WITHOUT BACK-PROPAGATION OR FORWARD-PROPAGATION
* **Authors:** Qinyu Li, Yee Whye Teh, Razvan Pascanu
* **arXiv:** [https://arxiv.org/abs/2403.13502](https://arxiv.org/abs/2403.13502) (Link based on search, may differ from user's original source ID)

## Core Concept of NoProp (and Implementation Differences)

* **Original NoProp Idea:**
    * Network blocks are trained independently.
    * Each block `t` learns to predict a "clean" target label embedding `u_y` from the input `x` and a noised version of the target embedding `z_{t-1}` (sampled according to `q(z_{t-1}|y)`).
    * Training avoids end-to-end forward and backward passes.
    * Inference involves iteratively applying the learned blocks starting from noise `z_0`, using a specific update rule (Equation 3 in the paper).

* **Problems & Modifications in this Implementation:**
    * Training blocks to predict the clean `u_y` proved **unstable and ineffective**, leading to collapsing denoising losses and failure to learn the classification task.
    * **Key Change:** The objective of the blocks was changed to predict the **added noise $\epsilon$** (similar to DDPM/VDM/CARD). The denoising loss function became the MSE between the predicted and actual noise.
    * The classifier was trained on the **predicted clean embedding $\hat{u}_T$**, which was reconstructed from $z_{T-1}$ and the predicted noise $\hat{\epsilon}_T$ by the last block (instead of training on the noisy $z_T$).
    * **Inference** was modified to a DDPM-like reverse process using the predicted noise $\epsilon$ for the $z_t \rightarrow z_{t-1}$ steps.

## Implementation Details (Final Working Version)

* **Framework:** PyTorch
* **Method:** Modified NoProp-DT (predicting noise $\epsilon$, classifying from $\hat{u}_T$)
* **Dataset:** MNIST
* **Architecture:** Based on Figure 7 (left) from the paper's appendix (CNN for image path, MLP for embedding path, combined layers). Block output is the predicted noise $\epsilon$ with `embed_dim`.
* **Embeddings:** Learned (`dim=20`) with orthogonal initialization. Used **embedding norm clipping** (`max_norm_embed=50.0`) and small **weight decay** (`embed_wd=1e-5`).
* **Noise Schedule:** Cosine schedule (`alphas_bar`).
* **Denoising Loss Weight:** Used weight $1 / (1 - \bar{\alpha}_{t-1})$ (inverse noise variance of the input $z_{t-1}$), scaled by `ETA_LOSS_WEIGHT=1.0`.
* **Inference:** Implemented DDPM-like reverse steps using predicted $\epsilon$.
* **Optimizer:** AdamW with `LR=1e-4`, `Weight Decay=1e-3` for blocks/classifier.
* **Stabilization:** Used **gradient clipping** (`max_norm=1.0`).
* **Epochs:** Trained for 100 epochs with per-epoch test evaluation and early stopping potential.

## Current Status & Experimental Results *(Updated Apr 10, 2025)*

After numerous debugging iterations and **significant modifications to the original NoProp algorithm** (including changing the prediction target to noise $\epsilon$ and altering the classifier input to $\hat{u}_T$), a **stable training process was achieved**.

* **Training Observations (Final Configuration):**
    * Training `Classify Loss` decreased steadily and significantly (reaching values < 0.1).
    * Training `AvgDenoiseLoss` (MSE on noise $\epsilon$) remained non-zero throughout training.
    * Embedding norm was successfully controlled via clipping.
* **Final Test Accuracy (MNIST):** The model achieved a test accuracy of approximately **XX.XX%** *(Replace XX.XX% with the best test accuracy achieved during the full training run)*.

*Intermediate Debugging Result:* At epoch 10, test accuracy reached **~39%**, demonstrating the viability of the modified approach.

## Conclusions from Experiments

* **Original Replication Challenges:** Reproducing the high accuracy (>99%) reported in the paper for NoProp-DT on MNIST using the method *as described* (predicting $u_y$) was **unsuccessful** in this implementation. Attempts consistently suffered from collapsing losses and failure to learn classification.
* **Necessity of Modifications:** Significant **deviations** from the original paper's description (predicting noise $\epsilon$, modifying the classifier target, specific loss weighting, stabilization techniques) were **required** to achieve *any* meaningful learning beyond random chance.
* **Performance of Modified Approach:** The modified algorithm (predicting $\epsilon$, classifying from $\hat{u}_T$) **demonstrated the capacity to learn**, achieving a test accuracy of **~XX.XX%**. While this is still considerably lower than state-of-the-art for MNIST, it shows that a backpropagation-free, diffusion-inspired approach *can* function with the right objective and stabilization.
* **Training/Inference Gap & Potential Issues:** The difficulties suggest potential issues with the original NoProp formulation's loss weighting, training target definition, sensitivity to hyperparameters, or missing implementation details in the paper that might be crucial for bridging the gap between the local training objectives and the global inference performance.
* **Overall:** This implementation serves as a demonstration of the mechanics of a *modified* NoProp-DT and highlights the significant challenges in reproducing and stabilizing such methods based solely on initial research descriptions.

## Setup

*(Setup instructions remain the same)*

**Option 1: Using `venv` (Recommended)**
1.  Ensure you have Python 3 installed.
2.  Clone the repository: `git clone <your-repository-url>`
3.  Navigate to the project directory: `cd <your-repository-name>`
4.  Run the setup script (this will create a virtual environment named `noprop_env` and install dependencies):
    ```bash
    chmod +x setup_env.sh
    ./setup_env.sh
    ```
5.  Activate the environment: `source noprop_env/bin/activate`
6.  Run the script: `python noprop_example.py`

**Option 2: Manual Installation**
1.  Ensure you have Python 3 installed.
2.  Clone the repository.
3.  Navigate to the project directory.
4.  Create a virtual environment (optional but recommended): `python -m venv noprop_env` and activate it (`source noprop_env/bin/activate` or `noprop_env\Scripts\activate` on Windows).
5.  Install dependencies: `pip install -r requirements.txt`
6.  Run the script: `python noprop_example.py`
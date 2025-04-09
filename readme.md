# NoProp-PyTorch: An Experimental Implementation of Training Without Backpropagation

## Description

This repository contains an experimental PyTorch implementation exploring the **NoProp** algorithm, presented in the paper "NOPROP: TRAINING NEURAL NETWORKS WITHOUT BACK-PROPAGATION OR FORWARD-PROPAGATION".

The goal of NoProp is to train neural networks without relying on traditional end-to-end backpropagation. Instead, it trains network layers independently using a local denoising objective inspired by diffusion models.

**Disclaimer:** This is an educational implementation attempt based on the paper. It may contain simplifications or deviations and does not guarantee replication of the results reported in the original work. Reproducing research paper results often requires significant effort in tuning and precise implementation details.

## Source Paper

* **Title:** NOPROP: TRAINING NEURAL NETWORKS WITHOUT BACK-PROPAGATION OR FORWARD-PROPAGATION
* **Authors:** Qinyu Li, Yee Whye Teh, Razvan Pascanu
* **arXiv:** [https://arxiv.org/abs/2503.24322](https://arxiv.org/abs/2503.24322) (Note: This is a hypothetical link based on the ID from the filename)
* **Version:** v1 [cs.LG] 31 Mar 2025

## Core Concept of NoProp

* Network layers (or blocks) are trained independently.
* Each block `t` learns to predict a "clean" target label embedding `u_y` from the input `x` and a noised version of the target embedding `z_{t-1}` (sampled according to the noise schedule `q(z_{t-1}|y)`).
* Training avoids end-to-end forward and backward passes through the entire network.
* **Inference** involves iteratively applying the learned blocks starting from noise `z_0`, using a specific update rule (Equation 3 in the paper) involving the block's prediction, the previous state, and added noise.

## Implementation Details

* **Framework:** PyTorch
* **Method:** NoProp-DT (Discrete-Time variant)
* **Dataset:** MNIST
* **Architecture:** Based on Figure 6 (left) from the paper's appendix (CNN for image path, MLP for label embedding path, combined layers).
* **Embeddings:** Experiments were run with:
    * Learned embeddings (`dim=20`) with orthogonal initialization.
    * Fixed one-hot embeddings (`dim=10`).
* **Noise Schedule:** Cosine schedule.
* **Inference:** Implemented according to Equation 3 from the paper, including coefficient calculation (with heuristics for handling the `t=1` boundary condition).
* **Optimizer:** AdamW with Weight Decay and Cosine Annealing LR Scheduler.
* **Epochs:** Trained for 100 epochs.

## Current Status & Experimental Results *(Updated Apr 9, 2025)*

Experiments were conducted to replicate the NoProp-DT results on MNIST, aligning hyperparameters (`T=10`, `LR`, `WD`, `BS`, `eta`, `Epochs=100`) and architecture choices more closely with the paper.

Two main configurations were tested for 100 epochs:

1.  **Learned Embeddings (dim=20, Orthogonal Init):**
    * Training `Classify Loss` decreased steadily, reaching **~0.98** by epoch 100.
    * Training `AvgDenoiseLoss` remained non-zero, increasing to ~0.32 towards the end.
    * **Final Test Accuracy (via Inference): ~11.35%**

2.  **Fixed One-Hot Embeddings (dim=10):**
    * Training `Classify Loss` **did not decrease**, remaining at the random guess level (~2.30) for all 100 epochs.
    * Training `AvgDenoiseLoss` successfully reached near zero, as expected with a fixed target.
    * **Final Test Accuracy (via Inference): ~10.27%** (effectively random chance).

**Outcome:** Neither configuration achieved satisfactory classification performance or came close to replicating the >99% accuracy reported in the source paper for MNIST with NoProp-DT.

## Conclusions from Experiments

* **Replication Challenges:** Reproducing the high accuracy reported in the paper proved unsuccessful with this implementation, despite aligning hyperparameters and architecture based on the paper's description.
* **Local vs. Global Learning:** The experiments clearly show that successfully training the blocks for their local denoising task (`AvgDenoiseLoss` -> 0, especially with fixed targets) **does not guarantee** effective performance on the global classification task when the blocks are composed during the inference process (Eq. 3).
* **Inference Discrepancy:** The significant gap between the final training classification loss (reaching ~0.98 in the learned embedding run) and the very low test accuracy suggests a major discrepancy. The representation `z_T` generated via the iterative inference process is likely very different from the distribution `q(z_T|y)` the classifier was trained on, or the inference process itself fails to produce class-discriminative features in this setup.
* **Potential Bottlenecks:** The most likely reasons for failure include:
    * Incorrect or suboptimal handling of the **boundary conditions at t=1** (for SNR loss weight and inference coefficients `a,b,c`) due to theoretical ambiguity and practical workarounds used (`clamp`, `alpha_0` heuristic).
    * Other subtle differences in implementation details or hyperparameters compared to the authors' original (unpublished) code.

* **Overall:** This implementation serves as a demonstration of the NoProp-DT mechanics but highlights the significant challenges in achieving high performance with this method without potentially more detailed guidance or resolving implementation ambiguities.

## Setup

*(Setup instructions remain the same as before - using venv/manual)*

**Option 1: Using `venv` (Recommended)**
1. Ensure you have Python 3 installed.
2. Clone the repository: `git clone <your-repository-url>`
3. Navigate to the project directory: `cd <your-repository-name>`
4. Run the setup script (this will create a virtual environment named `noprop_env` and install dependencies):
   ```bash
   chmod +x setup_env.sh
   ./setup_env.sh
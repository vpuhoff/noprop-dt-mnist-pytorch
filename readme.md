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
* **Embeddings:** Uses learned embeddings (`dim=20`) with orthogonal initialization (one of the strategies explored in the paper; easily modifiable to fixed one-hot).
* **Noise Schedule:** Cosine schedule for `alpha_bar`.
* **Inference:** Implemented according to Equation 3 from the paper, including calculation of coefficients `a_t, b_t, c_t` (with heuristics for handling the `t=1` boundary condition).

## Current Status (as of April 9, 2025)

The implementation runs and demonstrates the mechanics of independent layer training and the specific NoProp inference process. However, convergence on the MNIST classification task is currently observed to be **very slow**.

* After ~44 epochs (out of 100 planned), the classification cross-entropy loss decreased to ~1.96. This is significantly better than random guessing (~2.3) but still indicates low classification accuracy.
* The denoising loss (`AvgDenoiseLoss`) settled at a non-zero value (e.g., ~0.05-0.1), potentially due to the adaptation required for learnable embeddings acting as a moving target.

Further tuning, potentially more epochs, or closer adherence to all details from the original paper (e.g., alternative embedding strategies) might be required to achieve higher performance.

## Implementation Specifics and Differences from Paper

This implementation attempts to follow the NoProp-DT method but includes several simplifications, specific choices, and potential deviations from the authors' original (unpublished) code:

1.  **Boundary Condition Handling (t=1):**
    * **SNR Weight in Loss:** The weight term `SNR(t) - SNR(t-1)` is theoretically problematic at `t=1` (as `SNR(0)` is infinite). This code uses an **approximation**, setting the weight for `t=1` (index `t_idx=0`) equal to `SNR(1)`. The paper does not detail its exact handling of this boundary in the final objective (Eq. 8).
    * **Inference Coefficients (a₁, b₁, c₁):** The formulas for inference coefficients also lead to division by zero (`1 - alpha_bar_0`) at `t=1`. This code uses **`torch.clamp(..., min=1e-6)`** on the denominator and a **heuristic (`alpha_0 = alpha_1`)** for the numerator terms involving `alpha_0`. This allows the code to run but its theoretical correctness and alignment with the authors' approach at the first step are uncertain.

2.  **Class Embedding Strategy (`W_Embed`):**
    * This code implements **one** specific strategy: learned embeddings (`dim=20`) with orthogonal initialization.
    * The paper also explored and reported good results with **fixed one-hot embeddings** and **learned "prototypes"**. These alternatives are not implemented as easily switchable options here (though one-hot can be achieved with code modifications). The choice of embedding strategy can significantly impact results.

3.  **Objective Function:**
    * The KL divergence term `D_KL(q(z_0|y)||p(z_0))` from the full ELBO objective (Eq. 8) is **ignored** during loss calculation and weight updates in this implementation. While often constant w.r.t. model parameters, it's formally part of the objective described.

4.  **Classifier Parameterization (`p_theta_out`):**
    * The standard approach is used: a linear layer followed by `CrossEntropyLoss` (equivalent to Softmax + NLLLoss, corresponding to Eq. 14 in the paper).
    * An alternative method based on radial distance to embeddings (Eq. 15, 16), also proposed and tested in the paper, is **not implemented**.

5.  **Hyperparameter Exploration:**
    * The code uses a single set of hyperparameters (mostly aligned with Table 3 for MNIST/NoProp-DT). The original work likely involved exploration and tuning to find optimal values, and results for different configurations (like embedding types in Table 1) were presented.

6.  **Optimizations and Code Details:**
    * This implementation is conceptual and may differ from the authors' code in minor details (e.g., specific weight initializations for convolutions, BatchNorm parameters, speed/memory optimizations).

These differences and simplifications are important context for understanding the current implementation's behavior and performance relative to the results claimed in the original paper.

## Setup

**Option 1: Using `venv` (Recommended)**

1.  Ensure you have Python 3 installed.
2.  Clone the repository:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
3.  Run the setup script (creates `noprop_env` virtual environment and installs dependencies):
    ```bash
    chmod +x setup_env.sh
    ./setup_env.sh
    ```
4.  **Activate the environment:**
    * Linux/macOS:
        ```bash
        source noprop_env/bin/activate
        ```
    * Windows:
        ```bash
        noprop_env\Scripts\activate
        ```

**Option 2: Manual Setup (using `venv`)**

1.  Clone the repository and navigate into it.
2.  Create a virtual environment: `python3 -m venv noprop_env`
3.  Activate it (see commands above).
4.  Upgrade pip: `pip install --upgrade pip`
5.  Install dependencies:
    ```bash
    pip install torch torchvision numpy
    ```
    *(Note: Ensure you install the correct PyTorch version for your system/CUDA. See [https://pytorch.org/](https://pytorch.org/))*

## Usage

1.  Activate the virtual environment (if not already active).
2.  Run the main script:
    ```bash
    python noprop_example.py
    ```
The script will perform training for the configured number of epochs, print loss values periodically, and finally run evaluation on the MNIST test set using the NoProp inference procedure, reporting the accuracy.

## Files

* `noprop_example.py`: The main Python script containing the NoProp implementation, training loop, and inference function.
* `setup_env.sh`: Shell script to automate environment setup using `venv`.
* `README.md`: This file.

## Dependencies

* Python 3.x
* PyTorch (e.g., 2.x)
* Torchvision
* NumPy
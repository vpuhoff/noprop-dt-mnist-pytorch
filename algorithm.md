# Modified NoProp Algorithm: Mathematical Formulas (Working Implementation)

**Note:** The formulas below describe the *modified* NoProp-DT implementation that achieved successful results (e.g., 99.01% on MNIST) in our experiments. This version **deviates significantly** from the original paper ("NOPROP: TRAINING...", arXiv:2403.13502v1), particularly in the prediction target of the blocks, the inference mechanism, and the input to the classifier during training. These modifications were necessary to achieve stable and effective learning in our setup.

**Notation:**
* `t`: Time step index, typically from 1 to T (for coefficients like alpha_t) or 0 to T (for alpha_bar_t). Code often uses 0-based index `t_idx = t-1`.
* `x`: Input data (e.g., image).
* `y`: True class label.
* `u_y`: Embedding of the class label `y`.
* `z_t`: Noisy representation at step `t` in the *forward* process `q`. $z_0$ is the (theoretical) clean embedding, $z_T$ is near pure noise. The inference process also generates states denoted by `z`, running from $z_T$ down to $z_0$.
* $\alpha_t, \bar{\alpha}_t, \beta_t$: Noise schedule parameters.
* $\epsilon, \epsilon_{target}$: Gaussian noise $\mathcal{N}(0, I)$.
* $\hat{\epsilon}_{\theta_t}(z, x)`: Neural network block (parameterized by `theta_t`), predicting **noise $\epsilon$** (replaces $\hat{u}_{\theta_t}$).
* $\hat{p}_{\theta_{out}}(y|z)`: Final classifier (parameterized by `theta_out`).
* `q(...)`: Forward noising process (known).
* `p(...)`: Reverse denoising/generation process (learned).
* $\mathcal{N}(z | \mu, \sigma^2 I)$: Normal (Gaussian) distribution with mean vector $\mu$ and diagonal covariance matrix $\sigma^2 I$.

## 1. Noise Schedule (Same as Original)

A noise schedule determines how noise is added over T steps. We use `alpha_bar_t` (cumulative product of `alpha_t`) derived from a cosine schedule.

* **$\bar{\alpha}_t$**: `t = 0, 1, ..., T`. Defined by cosine schedule, $\bar{\alpha}_0 = 1$, $\bar{\alpha}_T \approx 0$.
* **$\alpha_t$**: `t = 1, ..., T`. $\alpha_t = \bar{\alpha}_t / \bar{\alpha}_{t-1}$.
* **$\beta_t$**: `t = 1, ..., T`. $\beta_t = 1 - \alpha_t$.

## 2. Forward Process `q` (Noising - For Training Sampling)

Used to generate noisy inputs `z_{t-1}` for training block `t` and the reconstructed $\hat{u}_T$ for training the classifier.

* **Distribution of noisy embedding `z_t` given clean `u_y`:**
    $$ q(z_t | y) = \mathcal{N}(z_t | \sqrt{\bar{\alpha}_t} u_y, (1 - \bar{\alpha}_t) I) $$
* **Sampling:** To get `z_{t-1}` for training block `t` (predicting noise $\epsilon$ added from $t-1 \to t$):
    1. Sample $\epsilon \sim \mathcal{N}(0, I)$.
    2. Compute $z_{t-1} = \sqrt{\bar{\alpha}_{t-1}} u_y + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon$.

## 3. Training Objective (Modified)

The model parameters ($\theta_t$ for blocks, $\theta_{out}$ for classifier, embedding weights) are optimized by minimizing a combined loss function using gradient descent (AdamW). The update follows the Algorithm 1 structure (outer `t` loop, updates within batch loop).

* **Total Loss (per batch, for combined update):**
    $$ L_{total} = L_{classify} + \sum_{t=1}^{T} L_{denoise, t} $$
    *(Note: In the outer loop structure, only one $L_{denoise, t}$ and the $L_{classify}$ contribute to the gradient for a given block's optimizer step, but $L_{classify}$ contributes to the final layer/embedding optimizer step in every batch iteration).*
    *Implementation:* Our working code used a combined `.backward()` call on `weighted_loss_t + classify_loss`.

* **Denoising Loss (for block `t`, target $\epsilon$):**
    $$ L_{denoise, t} = \eta \mathbb{E}_{y, \epsilon} \left[ || \hat{\epsilon}_{\theta_t}(z_{t-1}, x) - \epsilon ||^2 \right] $$
    Where:
        * $\epsilon \sim \mathcal{N}(0, I)$ is the sampled noise (target).
        * $z_{t-1} = \sqrt{\bar{\alpha}_{t-1}} u_y + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon$ is the input to the block.
        * $\hat{\epsilon}_{\theta_t}$ is the noise predicted by block `t` (network indexed `t-1` in code).
        * $\eta$ is the hyperparameter `ETA_LOSS_WEIGHT` (e.g., 0.5 in the best run). Uniform weighting across `t` was used (no complex SNR-based weight).

* **Classification Loss (Target $\hat{u}_T$):**
    $$ L_{classify} = \mathbb{E}_{y, \epsilon_{T-1}} [ -\log \hat{p}_{\theta_{out}}(y | \hat{u}_T.detach()) ] $$
    Where:
        * $\epsilon_{T-1} \sim \mathcal{N}(0, I)$.
        * $z_{T-1} = \sqrt{\bar{\alpha}_{T-1}} u_y + \sqrt{1 - \bar{\alpha}_{T-1}} \epsilon_{T-1}$.
        * $\hat{\epsilon}_T = \hat{\epsilon}_{\theta_T}(z_{T-1}, x)$ is the noise predicted by the *last* block (index `T-1`).
        * $\hat{u}_T = (z_{T-1} - \sqrt{1 - \bar{\alpha}_{T-1}} \hat{\epsilon}_T) / \sqrt{\bar{\alpha}_{T-1}}$ is the reconstructed clean embedding (predicted $z_0$).
        * `.detach()` prevents gradients flowing into block `T-1` from this loss.

## 4. Reverse Process `p` (Inference - Modified DDPM-like Step)

Generates a clean embedding prediction $z_0$ starting from pure noise $z_T$.

* **Initialization:** $z_T \sim \mathcal{N}(0, I)$.
* **Iteration:** For `t` from `T` down to `1`:
    1. Sample noise $\mathbf{z}' \sim \mathcal{N}(0, I)$ (if $t > 1$), else $\mathbf{z}' = 0$.
    2. Predict noise using block `t`: $\hat{\epsilon} = \hat{\epsilon}_{\theta_t}(z_t, x)$ (using `model.blocks[t-1]` in code, which takes $z_t$ as input in the inference function).
    3. Calculate conditional mean:
       $$ \mu_{t-1}(z_t, x) = \frac{1}{\sqrt{\alpha_t}} \left( z_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \hat{\epsilon} \right) $$
    4. Get conditional variance:
       $$ \sigma_t^2 = \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t $$
       (Use $\sigma_t^2 = 0$ if $t=1$).
    5. Sample previous state:
       $$ z_{t-1} = \mu_{t-1}(z_t, x) + \sigma_t \mathbf{z}' $$
* **Final Output:** $z_0$. This $z_0$ is then fed to the classifier.

## 5. Coefficients for Inference (Modified - DDPM-like)

The inference process requires these parameters from the noise schedule:
* $\alpha_t = 1 - \beta_t$
* $\beta_t$
* $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$
* $\bar{\alpha}_{t-1}$
* $\sigma_t^2 = \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$ (Posterior variance, often precomputed)

*(The coefficients $a_t, b_t, c_t$ from the original NoProp paper are **not used** in this modified inference process).*

## 6. Classifier Parameterization (Input Change)

* The classifier $\hat{p}_{\theta_{out}}(y|...)$ remains a standard linear layer followed by Softmax (implicitly included in `nn.CrossEntropyLoss`).
* **Key Change:** During **training**, it takes the *reconstructed clean embedding* $\hat{u}_T$ as input. During **inference**, it takes the final *denoised output* $z_0$ from the reverse process.
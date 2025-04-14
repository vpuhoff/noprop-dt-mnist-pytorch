# Modified NoProp Algorithm: Mathematical Formulas (Final Hybrid Implementation)

**Note:** The formulas below describe the *final, successful hybrid* NoProp-DT implementation that achieved **99.40% accuracy on MNIST** in our experiments. This version deviates significantly from the original paper ("NOPROP: TRAINING...", arXiv:2403.13502v1) and also refines the intermediate "predict noise" version. Key changes include a **hybrid loss function for the blocks** and training the classifier on reconstructed clean embeddings.

**Notation:**
* `t`: Time step index, typically from 1 to T (for coefficients like alpha_t) or 0 to T (for alpha_bar_t). Code often uses 0-based index `t_idx = t-1`.
* `x`: Input data (e.g., image).
* `y`: True class label.
* `u_y`: Embedding of the class label `y`.
* `z_t`: Noisy representation at step `t` in the *forward* process `q`. $z_0$ is the (theoretical) clean embedding, $z_T$ is near pure noise. The inference process also generates states denoted by `z`, running from $z_T$ down to $z_0$.
* $\alpha_t, \bar{\alpha}_t, \beta_t$: Noise schedule parameters.
* $\epsilon, \epsilon_{target}$: Gaussian noise $\mathcal{N}(0, I)$.
* $\hat{\epsilon}_{\theta_t}(z, x)`: Neural network block (parameterized by `theta_t`), predicting **noise $\epsilon$**.
* $\hat{u}_t$: Clean embedding reconstructed from $z_{t-1}$ and $\hat{\epsilon}_{\theta_t}$.
* $\hat{p}_{\theta_{out}}(y|z)`: Final classifier (parameterized by `theta_out`).
* `q(...)`: Forward noising process (known).
* `p(...)`: Reverse denoising/generation process (learned).
* $\mathcal{N}(z | \mu, \sigma^2 I)$: Normal (Gaussian) distribution.
* $\eta_1$: Hyperparameter `ETA_LOSS_WEIGHT` (Weight for noise MSE loss).
* $\eta_2$: Hyperparameter `LAMBDA_GLOBAL` (Weight for target MSE loss).

## 1. Noise Schedule (Same as Previous Working Version)

A noise schedule determines how noise is added over T steps. We use `alpha_bar_t` (cumulative product of `alpha_t`) derived from a cosine schedule.

* **$\bar{\alpha}_t$**: `t = 0, 1, ..., T`. Defined by cosine schedule, $\bar{\alpha}_0 = 1$, $\bar{\alpha}_T \approx 0$.
* **$\alpha_t$**: `t = 1, ..., T`. $\alpha_t = \bar{\alpha}_t / \bar{\alpha}_{t-1}$.
* **$\beta_t$**: `t = 1, ..., T`. $\beta_t = 1 - \alpha_t$.

## 2. Forward Process `q` (Noising - For Training Sampling)

Used to generate noisy inputs `z_{t-1}` for training block `t` and for reconstructing $\hat{u}_T$ when training the classifier.

* **Distribution of noisy embedding `z_t` given clean `u_y`:**
    $$ q(z_t | y) = \mathcal{N}(z_t | \sqrt{\bar{\alpha}_t} u_y, (1 - \bar{\alpha}_t) I) $$
* **Sampling:** To get `z_{t-1}` for training block `t`:
    1.  Sample $\epsilon \sim \mathcal{N}(0, I)$.
    2.  Compute $z_{t-1} = \sqrt{\bar{\alpha}_{t-1}} u_y + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon$.

## 3. Training Objective (Hybrid)

Model parameters ($\theta_t$ for blocks, $\theta_{out}$ for classifier, embedding weights) are optimized using AdamW and a combined loss. Training follows an outer loop over blocks `t` and an inner loop over batches. Updates use gradients from the combined loss after a single `.backward()` call.

* **Total Loss (per batch, for combined update):**
    $$ L_{total} = L_{classify} + L_{block, t} $$
    *(Where $L_{block, t}$ is the hybrid loss for the currently trained block `t`. Gradients from $L_{total}$ update both the block's parameters $\theta_t$ (via its optimizer `optimizer_t`) and the classifier/embedding parameters (via `optimizer_final`)).*

* **Hybrid Block Loss (for block `t`):** This combines a local noise prediction objective and a global target reconstruction objective.
    $$ L_{block, t} = \eta_1 L_{local, t} + \eta_2 L_{global, t} $$
    Where:
    * **Local Noise Prediction Loss:**
        $$ L_{local, t} = \mathbb{E}_{y, \epsilon} \left[ || \hat{\epsilon}_{\theta_t}(z_{t-1}, x) - \epsilon ||^2 \right] $$
        * $\epsilon \sim \mathcal{N}(0, I)$ is the target noise.
        * $z_{t-1} = \sqrt{\bar{\alpha}_{t-1}} u_y + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon$ is the input (using `u_y.detach()`).
        * $\hat{\epsilon}_{\theta_t}$ is the noise predicted by block `t`.
        * $\eta_1$ is `ETA_LOSS_WEIGHT` (best found: **2.0**).
    * **Global Target Reconstruction Loss:**
        $$ L_{global, t} = \mathbb{E}_{y, \epsilon} \left[ || \hat{u}_t - u_y ||^2 \right] $$
        * $\hat{u}_t = (z_{t-1} - \sqrt{1 - \bar{\alpha}_{t-1}} \hat{\epsilon}_{\theta_t}) / \sqrt{\bar{\alpha}_{t-1}}$ is the clean embedding reconstructed using the block's noise prediction $\hat{\epsilon}_{\theta_t}$ (requires gradients w.r.t. $\hat{\epsilon}_{\theta_t}$).
        * $u_y$ is the target clean embedding (use `u_y.detach()` here).
        * $\eta_2$ is `LAMBDA_GLOBAL` (best found: **1.0**).

* **Classification Loss (Same as before, Target $\hat{u}_T$):**
    $$ L_{classify} = \mathbb{E}_{y, \epsilon_{T-1}} [ -\log \hat{p}_{\theta_{out}}(y | \hat{u}_T.detach()) ] $$
    Where $\hat{u}_T$ is the clean embedding reconstructed using the *last* block's noise prediction $\hat{\epsilon}_T$.

## 4. Reverse Process `p` (Inference - DDPM-like Step)

Generates a clean embedding prediction $z_0$ starting from pure noise $z_T$. (This process remains unchanged from the previous working version).

* **Initialization:** $z_T \sim \mathcal{N}(0, I)$.
* **Iteration:** For `t` from `T` down to `1`:
    1.  Sample noise $\mathbf{z}' \sim \mathcal{N}(0, I)$ (if $t > 1$), else $\mathbf{z}' = 0$.
    2.  Predict noise using block `t`: $\hat{\epsilon} = \hat{\epsilon}_{\theta_t}(z_t, x)$.
    3.  Calculate conditional mean:
        $$ \mu_{t-1}(z_t, x) = \frac{1}{\sqrt{\alpha_t}} \left( z_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \hat{\epsilon} \right) $$
    4.  Get conditional variance:
        $$ \sigma_t^2 = \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t $$
        (Use $\sigma_t^2 = 0$ if $t=1$).
    5.  Sample previous state:
        $$ z_{t-1} = \mu_{t-1}(z_t, x) + \sigma_t \mathbf{z}' $$
* **Final Output:** $z_0$. This $z_0$ is then fed to the classifier.

## 5. Coefficients for Inference (DDPM-like)

The inference process requires these parameters from the noise schedule: (Unchanged from previous working version)
* $\alpha_t = 1 - \beta_t$
* $\beta_t$
* $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$
* $\bar{\alpha}_{t-1}$
* $\sigma_t^2 = \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$ (Posterior variance)

*(The coefficients $a_t, b_t, c_t$ from the original NoProp paper are **not used**).*

## 6. Classifier Parameterization (Input Change)

(Unchanged from previous working version)
* The classifier $\hat{p}_{\theta_{out}}(y|...)$ remains a standard linear layer followed by Softmax (implicitly included in `nn.CrossEntropyLoss`).
* **Key Change:** During **training**, it takes the *reconstructed clean embedding* $\hat{u}_T$ as input. During **inference**, it takes the final *denoised output* $z_0$ from the reverse process.
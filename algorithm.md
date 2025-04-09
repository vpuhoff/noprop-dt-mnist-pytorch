# NoProp Algorithm: Mathematical Formulas (NoProp-DT)

This document collects the key mathematical formulas and definitions from the paper "NOPROP: TRAINING NEURAL NETWORKS WITHOUT BACK-PROPAGATION OR FORWARD-PROPAGATION" (arXiv:2503.24322v1), relevant to the Discrete-Time variant (NoProp-DT) and used during the implementation and discussion.

**Notation:**
* `t`: Time step index, typically from 1 to T (in the paper).
* `x`: Input data (e.g., image).
* `y`: True class label.
* `u_y`: Embedding of the class label `y`.
* `z_t`: Noisy representation at step `t`.
* `alpha_t`, `alpha_bar_t`: Noise schedule parameters.
* `u_hat_theta_t(z, x)`: Neural network block (parameterized by `theta_t`), predicting `u_y`.
* `p_hat_theta_out(y|z)`: Final classifier (parameterized by `theta_out`).
* `q(...)`: Forward noising process (known).
* `p(...)`: Reverse denoising/generation process (learned).
* `N(z | mu, sigma^2 * I)`: Normal (Gaussian) distribution with mean vector `mu` and diagonal covariance matrix `sigma^2 * I`.

## 1. Noise Schedule

The implementation used a cosine schedule to define the cumulative product `alpha_bar_t`.

* **`alpha_bar_t`**: Defined for `t = 0, 1, ..., T`. It's set by a cosine schedule such that `alpha_bar_0 = 1`, and `alpha_bar_T` is close to 0.
    ```python
    # Approximate logic for generating alpha_bar (where timesteps = T)
    x = torch.linspace(0, timesteps, timesteps + 1)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0] # Ensures alpha_bar_0 = 1
    ```
* **`alpha_t`**: Derived from `alpha_bar_t` for `t = 1, ..., T`:
    $$ \alpha_t = \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}} $$

## 2. Forward Process `q` (Noising)

This process describes how to obtain a noisy representation `z_t` from the clean label embedding `u_y`. It's used to generate training examples for the blocks `u_hat_theta_t`.

* **Distribution of noisy embedding at step `t` (Eq. 6):**
    $$ q(z_t | y) = \mathcal{N}(z_t | \sqrt{\bar{\alpha}_t} u_y, (1 - \bar{\alpha}_t) I) $$
    *Used for sampling the input `z_{t-1}` for training block `t` (where the time index is shifted relative to the sample) and for sampling `z_T` when training the classifier. Note the paper index `t=1..T`.*

* **One-step reverse transition (Eq. 5):**
    $$ q(z_{t-1} | z_t) = \mathcal{N}(z_{t-1} | \sqrt{\alpha_t} z_t, (1 - \alpha_t) I) $$
    *Mainly used for deriving formulas.*

## 3. Training Objective `L_NoProp` (Eq. 8)

This is the primary loss function minimized during training (or rather, the ELBO is maximized, which is equivalent to minimizing `-ELBO`).

$$
\mathcal{L}_{NoProp} = \underbrace{\mathbb{E}_{q(z_T|y)}[-\log \hat{p}_{\theta_{out}}(y|z_T)]}_{\text{Classifier Loss}} + \underbrace{D_{KL}(q(z_0|y) || p(z_0))}_{\text{KL Term}} + \text{Denoising Loss Term}
$$

Where the Denoising Loss Term is:
$$
\frac{T}{2}\eta \mathbb{E}_{t \sim \mathcal{U}\{1,T\}, q(z_{t-1}|y)} \left[ (SNR(t) - SNR(t-1)) ||\hat{u}_{\theta_t}(z_{t-1}, x) - u_y||^2 \right]
$$

* **`SNR(t)` (Signal-to-Noise Ratio):** Defined as (see after Eq. 8):
    $$ SNR(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t} $$
* **`p(z_0)` (Prior distribution):** Usually a standard normal distribution `p(z_0) = N(z_0 | 0, I)` (Eq. 7).
* **`eta`:** Hyperparameter weighting the denoising term.
* **Note 1:** In the implementation, the `D_KL` term was often ignored during gradient updates, as it might be constant with respect to model parameters `theta_t` and `theta_out`.
* **Note 2:** In the implementation, the weight `SNR(t) - SNR(t-1)` at `t=1` was approximated as `SNR(1)` because `SNR(0)` is undefined.
* **Note 3:** The expectation `E_{q(z_{t-1}|y)}` means that to compute the MSE loss for block `t`, the input `z_{t-1}` is sampled from `q(z_{t-1}|y)`.

## 4. Reverse Process `p` (Inference / Generation)

This process describes how, starting from noise `z_0`, to sequentially generate `z_1, z_2, ..., z_T` using the learned blocks `u_hat_theta_t`.

* **Initial step:** `z_0 ~ N(0, I)`.
* **Iterative update (Eq. 3 and Eq. 7):** For `t` from 1 to `T`:
    $$ z_t = \mu_t(z_{t-1}, \hat{u}_{\theta_t}(z_{t-1}, x)) + \sqrt{c_t} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I) $$
    Where `mu_t(z, u) = a_t u + b_t z`. Substituting `mu_t` gives the explicit formula:
    $$ z_t = a_t \hat{u}_{\theta_t}(z_{t-1}, x) + b_t z_{t-1} + \sqrt{c_t} \epsilon_t $$

## 5. Inference Coefficients (`a_t, b_t, c_t`)

These coefficients link the noise schedule parameters (`alpha_t`, `alpha_bar_t`) to the dynamics of the reverse process. The formulas are given between Eq. 6 and 7 (derived in Appendix A.3). The index `t` runs from 1 to `T`.

$$ a_t = \frac{\sqrt{\bar{\alpha}_t}(1 - \alpha_{t-1})}{1 - \bar{\alpha}_{t-1}} $$
$$ b_t = \frac{\sqrt{\alpha_{t-1}}(1 - \bar{\alpha}_t)}{1 - \bar{\alpha}_{t-1}} $$
$$ c_t = \frac{(1 - \bar{\alpha}_t)(1 - \alpha_{t-1})}{1 - \bar{\alpha}_{t-1}} $$

* **Note:** As discussed, these formulas are theoretically problematic at `t=1` because `alpha_bar_0 = 1`. The implementation used clamping (`clamp(..., min=1e-6)`) for the denominator and a heuristic (`alpha_0 = alpha_1`) to handle this boundary case practically.

## 6. Classifier Parameterization (`p_theta_out`)

* **Standard Approach (Eq. 14):** Uses a linear layer `f_{theta_out}` followed by Softmax.
    $$ \hat{p}_{\theta_{out}}(y|z_T) = \text{softmax}(f_{\theta_{out}}(z_T))_y $$
    *This is the approach implemented in the code using `nn.Linear` and `nn.CrossEntropyLoss`.*

* **Alternative Approach (Eq. 15, 16):** The paper also describes a method using radial distance to class embeddings. *This was not used in the current implementation.*
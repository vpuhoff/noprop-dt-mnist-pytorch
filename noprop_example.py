import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import math

# --- Hyperparameters ---
T = 20  # Number of "layers" or diffusion steps (Reduced for simplicity)
EMBED_DIM = 128 # Dimension for label embeddings
NUM_CLASSES = 10
IMG_SIZE = 28
IMG_CHANNELS = 1
BATCH_SIZE = 128
EPOCHS = 10 # Reduced for quick demo
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ETA_LOSS_WEIGHT = 0.1 # Corresponds to eta in Eq. 8 (hyperparameter)

# --- 1. Noise Schedule (Simplified Cosine Schedule) ---
# Generate alpha_bar schedule (cumulative product of alphas)
# Based on https://arxiv.org/abs/2102.09672 (Improved Denoising Diffusion Probabilistic Models)
def get_cosine_schedule(timesteps, s=0.008):
    """Generates cosine schedule alphas."""
    steps = timesteps + 1
    # ИЗМЕНЕНО: Используем torch.float32 вместо torch.float64
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    # Убедимся, что результат тоже float32
    return torch.clip(alphas, 0.0001, 0.9999).float() # Добавлено .float() для надежности

alphas = get_cosine_schedule(T).to(DEVICE)
alphas_cumprod = torch.cumprod(alphas, dim=0).to(DEVICE)
# Precompute values needed for q(z_t | y) = N(z_t | sqrt(alpha_bar_t)*u_y, (1-alpha_bar_t)I)
# Note: The paper uses t=1...T. Index 0 here corresponds to t=1 in the paper.
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# SNR(t) = alpha_bar_t / (1 - alpha_bar_t)
# Need SNR(t) and SNR(t-1) for loss weighting (Eq. 56, 8)
# Let SNR(-1) = SNR(0) for t=1 case, or handle boundary appropriately.
# Paper uses SNR(t) - SNR(t-1). Add a value for t=0 for indexing ease.
snr = alphas_cumprod / (1.0 - alphas_cumprod)
snr_diff = torch.zeros_like(snr)
snr_diff[0] = snr[0] # Approximation for t=1 (SNR(1)-SNR(0)) - Needs careful definition at boundary
snr_diff[1:] = snr[1:] - snr[:-1]
# Ensure weighting is positive as in Eq. 55 -> 56 derivation
snr_loss_weight = 0.5 * torch.abs(snr_diff) # Simplified weight factor from Eq. 8/56

# --- 2. Label Embeddings (Fixed One-Hot for Simplicity) ---
# In the paper, W_Embed could be identity, learned, or prototypes.
# Using fixed one-hot embeddings projected to EMBED_DIM.
label_embedding = nn.Embedding(NUM_CLASSES, EMBED_DIM).to(DEVICE)
label_embedding.weight.data.normal_(0, 0.1) # Initialize randomly
# For fixed one-hot, you could manually create the matrix or use nn.functional.one_hot

# --- 3. Model Architecture (Simplified) ---
# Each block u_theta_t needs to process image x and noisy label z_{t-1}
# Based loosely on Figure 6 but highly simplified

class DenoisingBlockPaper(nn.Module):
    def __init__(self, embed_dim, img_channels=1, img_size=28):
        super().__init__()
        self.embed_dim = embed_dim

        # --- Путь обработки изображения (x) ---
        self.img_conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # 28x28 -> 14x14 (для MNIST)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # 7x7 -> 3x3 (убедитесь, что размер корректен для ваших данных)
            nn.Flatten()
        )
        # Рассчитываем размер после сверток и пулинга
        # Для MNIST (1, 28, 28) -> (128, 3, 3) -> 128 * 3 * 3 = 1152
        # Для CIFAR (3, 32, 32) -> (128, 4, 4) -> 128 * 4 * 4 = 2048
        # Сделаем расчет динамическим, если возможно, или зададим явно
        with torch.no_grad():
             dummy_input = torch.zeros(1, img_channels, img_size, img_size)
             conv_output_size = self.img_conv(dummy_input).shape[-1]

        self.img_fc = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.BatchNorm1d(256), # BatchNorm ПЕРЕД ReLU как на схеме
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # --- Путь обработки зашумленной метки (z_t_minus_1) ---
        # Диаграмма показывает 2 слоя FC(256) -> BN -> ReLU -> Dropout
        self.label_embed = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # --- Комбинированный путь после конкатенации ---
        # Вход: 256 (от img_fc) + 256 (от label_embed) = 512
        self.combined = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Выходной слой блока предсказывает u_y, поэтому размер embed_dim
            nn.Linear(128, self.embed_dim)
        )

    def forward(self, x, z_t_minus_1):
        img_features_conv = self.img_conv(x)
        img_features = self.img_fc(img_features_conv)

        label_features = self.label_embed(z_t_minus_1)

        combined_features = torch.cat([img_features, label_features], dim=1)
        predicted_u_y = self.combined(combined_features)
        return predicted_u_y


class NoPropNet(nn.Module):
    def __init__(self, num_blocks, embed_dim, num_classes, img_channels=1, img_size=28): # Добавлен img_size
        super().__init__()
        self.num_blocks = num_blocks
        # Создаем T независимых блоков новой архитектуры
        self.blocks = nn.ModuleList([
            # Используем новый класс блока
            DenoisingBlockPaper(embed_dim, img_channels, img_size)
            for _ in range(num_blocks)
        ])
        # Финальный классификатор остается тем же
        self.classifier = nn.Linear(embed_dim, num_classes)

    # Концептуальный forward для inference (НЕ используется в NOPROP обучении)
    def forward(self, x, z_0):
        z_t = z_0
        for t in range(self.num_blocks):
            # Используем блок t для предсказания u_y
            pred_u_y = self.blocks[t](x, z_t)
            # Упрощенное обновление (требует реализации Ур. 3 для корректного inference)
            z_t = pred_u_y
        logits = self.classifier(z_t)
        return logits

# --- 4. Dataset ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# --- 5. Initialization ---
model = NoPropNet(num_blocks=T, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES, img_channels=IMG_CHANNELS, img_size=IMG_SIZE).to(DEVICE)
# Separate optimizer for each block + one for classifier/embeddings
optimizers = [optim.Adam(block.parameters(), lr=LR) for block in model.blocks]
optimizer_final = optim.Adam(list(model.classifier.parameters()) + list(label_embedding.parameters()), lr=LR) # Include label_embedding if learnable

mse_loss = nn.MSELoss()
ce_loss = nn.CrossEntropyLoss()

# --- 6. NOPROP Training Loop ---
print("Starting NoProp Training...")
for epoch in range(EPOCHS):
    running_loss_denoise = 0.0
    running_loss_classify = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        batch_size = inputs.shape[0]

        # Get target label embeddings u_y
        with torch.no_grad(): # Embeddings might be fixed or learnable
             u_y = label_embedding(labels)

        # --- Core NoProp: Train each block independently ---
        total_denoising_loss_batch = 0.0
        for t in range(T): # Iterate through layers/timesteps t=1...T (index t=0..T-1)
            # Sample noise epsilon ~ N(0, I)
            epsilon = torch.randn_like(u_y).to(DEVICE)

            # Calculate z_{t-1} using q(z_{t-1}|y) = N(z_{t-1} | sqrt(a_bar_{t-1})u_y, (1-a_bar_{t-1})I)
            # Need alpha_bar at t-1. Use alpha_bar[t-1] (index maps directly)
            sqrt_a_bar = sqrt_alphas_cumprod[t].view(1, -1).expand(batch_size, -1) # Expand to batch size
            sqrt_1_minus_a_bar = sqrt_one_minus_alphas_cumprod[t].view(1, -1).expand(batch_size, -1)
            
            # Sample z_t (corresponding to z_{t-1} in paper's notation for input to block t)
            z_t_input = sqrt_a_bar * u_y + sqrt_1_minus_a_bar * epsilon

            # Get prediction from the t-th block
            block_to_train = model.blocks[t]
            predicted_u_y = block_to_train(inputs, z_t_input.detach()) # detach z_t? Check paper theory if z_t depends on learnable params

            # Calculate denoising loss for this block (MSE term from Eq. 8 / Eq. 56)
            # Use || \hat{u}_{\theta_t}(z_{t-1}, x) - u_y ||^2
            loss_t = mse_loss(predicted_u_y, u_y)

            # Apply weighting factor T * eta * (SNR(t) - SNR(t-1)) / 2
            # Use precomputed snr_loss_weight[t]
            weighted_loss_t = T * ETA_LOSS_WEIGHT * snr_loss_weight[t] * loss_t
            total_denoising_loss_batch += weighted_loss_t.item()

            # --- Independent Backpropagation for block t ---
            optimizers[t].zero_grad()
            weighted_loss_t.backward() # Computes gradients ONLY for parameters in block t
            optimizers[t].step() # Updates ONLY parameters in block t

        # --- Train Classifier and Embeddings (if learnable) ---
        # This uses the CrossEntropy term from Eq. 8: E_q(z_T|y)[-log p_theta_out(y|z_T)]
        # Sample z_T from q(z_T|y)
        epsilon_final = torch.randn_like(u_y).to(DEVICE)
        sqrt_a_bar_T = sqrt_alphas_cumprod[T-1].view(1,-1).expand(batch_size,-1) # alpha_bar at T (index T-1)
        sqrt_1_minus_a_bar_T = sqrt_one_minus_alphas_cumprod[T-1].view(1,-1).expand(batch_size,-1)
        z_T_sample = sqrt_a_bar_T * u_y + sqrt_1_minus_a_bar_T * epsilon_final

        # Get classification logits
        logits = model.classifier(z_T_sample.detach()) # Detach z_T as it's input here
        classify_loss = ce_loss(logits, labels)
        running_loss_classify += classify_loss.item()

        # Backprop for classifier and embeddings
        optimizer_final.zero_grad()
        classify_loss.backward()
        optimizer_final.step()

        # --- Logging ---
        running_loss_denoise += total_denoising_loss_batch
        if i % 100 == 99: # Print every 100 mini-batches
             print(f'[Epoch {epoch + 1}, Batch {i + 1:5d}] Avg Denoise Loss: {running_loss_denoise / (100 * T):.3f}, Classify Loss: {running_loss_classify / 100:.3f}')
             running_loss_denoise = 0.0
             running_loss_classify = 0.0

print('Finished Training')

# --- 7. Inference (Conceptual Description) ---
# Inference requires implementing Eq. 3:
# z_t = a_t * hat_u_theta_t(z_{t-1}, x) + b_t * z_{t-1} + sqrt(c_t) * epsilon_t
# Where a_t, b_t, c_t depend on the alpha schedule.
# This involves a loop:
# 1. Start with z_0 ~ N(0, I)
# 2. For t = 1 to T:
#    a. Predict u_y using model.blocks[t-1](x, z_{t-1})
#    b. Sample noise epsilon_t ~ N(0, I) (or use deterministic ODE trajectory if applicable)
#    c. Calculate a_t, b_t, c_t based on the noise schedule (alphas). See Appendix A.3 / Eq. 6.
#    d. Compute z_t using Eq. 3.
# 3. Use the final z_T as input to model.classifier(z_T) to get logits.

# Due to the complexity of implementing a_t, b_t, c_t correctly and the focus
# on the training loop, the inference code is omitted here.
# A proper implementation would require deriving these coefficients from the chosen alpha schedule.

print("\nNOTE: Inference requires implementing the iterative update from Eq. 3 using coefficients derived from the noise schedule (a_t, b_t, c_t). This example focused on the training loop.")
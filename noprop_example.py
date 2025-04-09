import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import math
import time  # Для замера времени

# --- Hyperparameters ---
# Number of "layers" or diffusion steps (Можно увеличить до 10, как в статье)
T = 10
# Dimension for label embeddings (В статье были и другие значения)
EMBED_DIM = 20
NUM_CLASSES = 10
IMG_SIZE = 28
IMG_CHANNELS = 1
BATCH_SIZE = 128
EPOCHS = 100  # Увеличено количество эпох
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ETA_LOSS_WEIGHT = 0.1  # Corresponds to eta in Eq. 8 (from Table 3)

print(f"Using device: {DEVICE}")
print(f"Parameters: T={T}, EmbedDim={EMBED_DIM}, Epochs={EPOCHS}, LR={LR}")

# --- 1. Noise Schedule (Обновлено для Inference) ---


def get_alpha_bar_schedule(timesteps, s=0.008):
    """Generates cosine schedule alphas_cumprod (alpha_bar) including alpha_bar_0 = 1."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps,
                       dtype=torch.float32)  # 0, 1, ..., T
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / \
        alphas_cumprod[0]  # Ensure alpha_bar_0 = 1
    return torch.clip(alphas_cumprod, 0.0001, 1.0)


# Size T+1, index 0..T maps to alpha_bar_0..alpha_bar_T
alphas_bar = get_alpha_bar_schedule(T).to(DEVICE)

# Derive alpha_t = alpha_bar_t / alpha_bar_{t-1} for t=1..T
# alphas[t] corresponds to alpha_{t+1} in paper's 1..T indexing
# Size T, index 0..T-1 maps to alpha_1..alpha_T
alphas = alphas_bar[1:] / alphas_bar[:-1]

# Precompute sqrt values needed for training q(z_t | y)
# Index 0..T-1 maps to paper t=1..T
sqrt_alphas_cumprod_train = torch.sqrt(alphas_bar[1:])
sqrt_one_minus_alphas_cumprod_train = torch.sqrt(1.0 - alphas_bar[1:])

# SNR calculations for training loss - based on paper t=1..T (index 0..T-1)
# Avoid division by zero
snr = alphas_bar[1:] / torch.clamp(1.0 - alphas_bar[1:], min=1e-6)
snr_diff = torch.zeros_like(snr)
# Paper uses SNR(t) - SNR(t-1). Need SNR(0) for t=1.
# We approximate SNR(0) using SNR(1) value, or simply use SNR(1) for the first term's weight.
# Let's use clamp to avoid issues with near-zero values if alpha_bar_1 is near 1.
snr_diff[0] = snr[0]  # Weight for t=1 (index 0)
snr_diff[1:] = snr[1:] - snr[:-1]
# Ensure weight is positive, factor 0.5 from Eq 56
# Index 0..T-1 maps to weights for t=1..T
snr_loss_weight = 0.5 * torch.abs(snr_diff)

# --- Precompute coefficients a_t, b_t, sqrt_c_t for INFERENCE ---
# t goes from 1 to T (paper index). Code index t_idx = 0 to T-1.
a_coeffs = []
b_coeffs = []
sqrt_c_coeffs = []

# Handle boundary case t=1 (t_idx=0) using heuristic alpha_0 = alpha_1
# alpha_1 is alphas[0]
alpha_0_heuristic = alphas[0]
alpha_bar_0 = alphas_bar[0]  # Should be 1.0

for t_idx in range(T):  # t_idx = 0..T-1 corresponds to paper t = 1..T
    alpha_bar_t = alphas_bar[t_idx + 1]
    alpha_bar_t_minus_1 = alphas_bar[t_idx]

    # Use heuristic for alpha_{t-1} when t=1 (t_idx=0) -> corresponds to alpha_0
    alpha_t_minus_1 = alpha_0_heuristic if t_idx == 0 else alphas[t_idx - 1]

    # Clamp denominator slightly away from zero for stability
    denom = torch.clamp(1.0 - alpha_bar_t_minus_1, min=1e-6)

    a_t = (torch.sqrt(alpha_bar_t) * (1.0 - alpha_t_minus_1)) / denom
    b_t = (torch.sqrt(torch.clamp(alpha_t_minus_1, min=1e-6)) *
           (1.0 - alpha_bar_t)) / denom  # Clamp alpha_t_minus_1 before sqrt
    c_t = ((1.0 - alpha_bar_t) * (1.0 - alpha_t_minus_1)) / denom
    sqrt_c_t = torch.sqrt(torch.clamp(c_t, min=1e-6))  # Clamp c_t before sqrt

    a_coeffs.append(a_t)
    b_coeffs.append(b_t)
    sqrt_c_coeffs.append(sqrt_c_t)

# Convert lists to tensors and reshape for broadcasting
# Shape [T, 1, 1] for broadcasting over [B, EmbDim]
a_coeffs = torch.stack(a_coeffs).view(T, 1, 1).to(DEVICE)
b_coeffs = torch.stack(b_coeffs).view(T, 1, 1).to(DEVICE)
sqrt_c_coeffs = torch.stack(sqrt_c_coeffs).view(T, 1, 1).to(DEVICE)

print("Noise schedule and Inference coefficients a, b, sqrt(c) precomputed.")


# --- 2. Label Embeddings ---
# Using learnable embeddings
label_embedding = nn.Embedding(NUM_CLASSES, EMBED_DIM).to(DEVICE)

if NUM_CLASSES <= EMBED_DIM:
    try: # Добавим try-except на случай редких проблем с размерностями
        nn.init.orthogonal_(label_embedding.weight)
        print(f"Applied orthogonal initialization to label embeddings (dim={EMBED_DIM}).")
    except ValueError as e:
         print(f"Warning: Orthogonal init failed ({e}). Using default init.")
else:
    # В нашем случае NUM_CLASSES=10, EMBED_DIM=20, условие выполняется
    print(f"Warning: Cannot apply orthogonal init (num_classes {NUM_CLASSES} > embed_dim {EMBED_DIM}). Using default init.")


# --- 3. Model Architecture (ближе к Рисунку 6, слева) ---
class DenoisingBlockPaper(nn.Module):
    def __init__(self, embed_dim, img_channels=1, img_size=28):
        super().__init__()
        self.embed_dim = embed_dim

        # --- Путь обработки изображения (x) ---
        self.img_conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7 -> 3x3
            nn.Flatten()
        )
        # Рассчитываем размер после сверток и пулинга
        with torch.no_grad():
            dummy_input = torch.zeros(1, img_channels, img_size, img_size)
            conv_output_size = self.img_conv(dummy_input).shape[-1]
            # print(f"Calculated conv output size: {conv_output_size}") # Debugging

        self.img_fc = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.BatchNorm1d(256),  # BatchNorm ПЕРЕД ReLU
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # --- Путь обработки зашумленной метки (z_t_input for block t) ---
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

    # Input z_t_sampled is z_t sampled from q(z_t|y) for training block t+1 (paper index)
    # Or z_{t-1} generated during inference for inference step t (paper index)
    def forward(self, x, z_input):
        img_features_conv = self.img_conv(x)
        img_features = self.img_fc(img_features_conv)

        label_features = self.label_embed(z_input)

        combined_features = torch.cat([img_features, label_features], dim=1)
        predicted_u_y = self.combined(combined_features)
        return predicted_u_y


class NoPropNet(nn.Module):
    def __init__(self, num_blocks, embed_dim, num_classes, img_channels=1, img_size=28):
        super().__init__()
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim  # Store embed_dim
        # Создаем T независимых блоков новой архитектуры
        self.blocks = nn.ModuleList([
            DenoisingBlockPaper(embed_dim, img_channels, img_size)
            for _ in range(num_blocks)
        ])
        # Финальный классификатор остается тем же
        self.classifier = nn.Linear(embed_dim, num_classes)

    # Этот forward НЕ используется для обучения NoProp, только для примера/концепции
    def forward(self, x, z_0):
        # Conceptual inference pass (simplified, use run_inference instead)
        print("Warning: NoPropNet.forward is conceptual only. Use run_inference.")
        z_t = z_0
        for t in range(self.num_blocks):
            pred_u_y = self.blocks[t](x, z_t)
            z_t = pred_u_y  # Incorrect update for actual inference
        logits = self.classifier(z_t)
        return logits


# --- 4. Dataset ---
transform = transforms.Compose([transforms.ToTensor(
), transforms.Normalize((0.1307,), (0.3081,))])  # MNIST stats

trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=2, pin_memory=True if DEVICE == 'cuda' else False)

testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True if DEVICE == 'cuda' else False)


# --- 5. Initialization ---
model = NoPropNet(num_blocks=T, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES,
                  img_channels=IMG_CHANNELS, img_size=IMG_SIZE).to(DEVICE)

# Separate optimizer for each block + one for classifier/embeddings
optimizers = [optim.AdamW(block.parameters(), lr=LR, weight_decay=1e-3) for block in model.blocks]

optimizer_final = optim.AdamW(list(model.classifier.parameters()) + list(label_embedding.parameters()), lr=LR, weight_decay=1e-3)  # AdamW

mse_loss = nn.MSELoss()
ce_loss = nn.CrossEntropyLoss()


# --- 6. NOPROP Training Loop ---
print("Starting NoProp Training...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()  # Устанавливаем режим обучения
    running_loss_denoise_total = 0.0
    running_loss_classify = 0.0
    processed_samples = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        batch_size = inputs.shape[0]

        # Get target label embeddings u_y
        u_y = label_embedding(labels) # Requires grad if embedding is learned

        # --- Core NoProp: Train each block independently ---
        total_denoising_loss_batch = 0.0
        # According to Eq 56, block t takes z_{t-1} sampled from q(z_{t-1}|y)
        # According to Algorithm 1, we sample z_t ~ q(z_t|y) and compute loss for block t using z_{t-1}? Ambiguous.
        # Let's follow Eq 56: Train block 't' (paper index) using z_{t-1} sampled from q(z_{t-1}|y)
        # Our block index `t_idx` = 0..T-1 corresponds to paper index `t` = 1..T.
        # So block `t_idx` takes `z_{t_idx}` sampled from `q(z_{t_idx}|y)` as input z_input
        for t_idx in range(T):
            # Sample z_t (paper index t = t_idx + 1) from q(z_t | y)
            epsilon = torch.randn_like(u_y)
            sqrt_a_bar = torch.sqrt(alphas_bar[t_idx]).view(-1, 1).expand_as(u_y) # Используем alpha_bar_{t-1} (paper)
            sqrt_1_minus_a_bar = torch.sqrt(1.0 - alphas_bar[t_idx]).view(-1, 1).expand_as(u_y)

            # Это z_{t-1} (paper index) sampled from q(z_{t-1}|y)
            z_input_for_block = sqrt_a_bar * u_y.detach() + sqrt_1_minus_a_bar * epsilon # Используем detach для u_y здесь тоже

            block_to_train = model.blocks[t_idx]
            # Блок t (paper, t_idx) получает z_{t-1} (paper)
            predicted_u_y = block_to_train(inputs, z_input_for_block)

            loss_t = mse_loss(predicted_u_y, u_y.detach()) # Цель - чистый u_y (отсоединенный)

            # Вес для блока t (paper, t_idx) использует SNR(t) - SNR(t-1) (paper)
            # Это соответствует snr_loss_weight[t_idx]
            weighted_loss_t = T * ETA_LOSS_WEIGHT * snr_loss_weight[t_idx] * loss_t


            total_denoising_loss_batch += weighted_loss_t.item()

            # --- Independent Backpropagation for block t_idx ---
            optimizers[t_idx].zero_grad()
            weighted_loss_t.backward()  # Computes gradients ONLY for parameters in block t_idx
            optimizers[t_idx].step()  # Updates ONLY parameters in block t_idx

        # --- Train Classifier and Embeddings (if learnable) ---
        # This uses the CrossEntropy term from Eq. 8: E_q(z_T|y)[-log p_theta_out(y|z_T)]
        # Sample z_T from q(z_T|y)
        epsilon_final = torch.randn_like(u_y)
        # Use sqrt_..._train[T-1] which corresponds to paper time T
        sqrt_a_bar_T = sqrt_alphas_cumprod_train[T -
                                                 1].view(-1, 1).expand_as(u_y)
        sqrt_1_minus_a_bar_T = sqrt_one_minus_alphas_cumprod_train[T-1].view(-1, 1).expand_as(
            u_y)
        z_T_sample = sqrt_a_bar_T * u_y + sqrt_1_minus_a_bar_T * \
            epsilon_final  # Detach u_y here?

        # Get classification logits
        logits = model.classifier(z_T_sample)
        classify_loss = ce_loss(logits, labels)

        # Backprop for classifier and embeddings
        optimizer_final.zero_grad()
        classify_loss.backward()
        optimizer_final.step()

        # --- Logging ---
        running_loss_denoise_total += total_denoising_loss_batch
        running_loss_classify += classify_loss.item()
        processed_samples += batch_size

        if (i + 1) % 100 == 0:  # Print every 100 mini-batches
            avg_denoise_loss_print = running_loss_denoise_total / (100 * T)
            avg_classify_loss_print = running_loss_classify / 100
            print(f'[Epoch {epoch + 1}/{EPOCHS}, Batch {i + 1:5d}/{len(trainloader)}] AvgDenoiseLoss: {avg_denoise_loss_print:.4f}, ClassifyLoss: {avg_classify_loss_print:.4f}')
            running_loss_denoise_total = 0.0
            running_loss_classify = 0.0

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch + 1} finished. Time elapsed: {epoch_time:.2f}s")


print('Finished Training')
total_training_time = time.time() - start_time
print(f"Total Training Time: {total_training_time:.2f}s")


# --- 7. Inference Function ---
@torch.no_grad()  # Disable gradient calculations for inference
def run_inference(model, x_batch, T_steps, a_coeffs, b_coeffs, sqrt_c_coeffs, device):
    """
    Runs the NoProp inference process using Equation 3.
    """
    batch_size = x_batch.shape[0]
    embed_dim = model.embed_dim  # Get embed_dim from model

    # 1. Start with z_0 ~ N(0, I)
    z = torch.randn(batch_size, embed_dim, device=device)

    # 2. Iteratively apply blocks using Equation 3
    for t in range(T_steps):  # Loop t from 0 to T-1 (corresponds to steps t=1 to T)
        # model.blocks[t] corresponds to u_hat_theta_{t+1} in paper index
        # Input to block t (paper t+1) is z_t (paper t)
        u_hat = model.blocks[t](x_batch, z)  # Pass current state z

        # Get coefficients for this step t (index t)
        # Coeffs are indexed 0..T-1, corresponding to step t=1..T
        a_t = a_coeffs[t]
        b_t = b_coeffs[t]
        sqrt_c_t = sqrt_c_coeffs[t]

        # Sample noise epsilon_t ~ N(0, I)
        epsilon_t = torch.randn_like(z)
        # if t == T_steps - 1: # Option: No noise on the last step?
        #    epsilon_t.zero_()

        # Apply Equation 3: z_{t+1} = a_{t+1} * u_hat_{t+1} + b_{t+1} * z_t + sqrt(c_{t+1}) * epsilon_{t+1}
        # Our t is index 0..T-1, z is z_t (paper index), coeffs[t] are for step t+1 (paper index)
        z = a_t * u_hat + b_t * z + sqrt_c_t * epsilon_t  # z becomes z_{t+1}

    # 3. After T steps, z represents z_T. Use the classifier.
    logits = model.classifier(z)

    return logits


# --- 8. Evaluation after Training ---
print("\nRunning evaluation using inference process...")
eval_start_time = time.time()

correct = 0
total = 0
model.eval()  # Set model to evaluation mode

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Run inference process
        logits = run_inference(model, images, T, a_coeffs,
                               b_coeffs, sqrt_c_coeffs, DEVICE)

        # Get predictions
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
eval_time = time.time() - eval_start_time
print(
    f'Accuracy on the {total} test images using NoProp inference: {accuracy:.2f} %')
print(f"Evaluation Time: {eval_time:.2f}s")

# model.train() # Optional: return to training mode if needed later

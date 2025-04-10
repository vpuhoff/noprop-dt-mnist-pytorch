import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import math
import time # Для замера времени

# --- Hyperparameters ---
T = 10
EMBED_DIM = 20
NUM_CLASSES = 10
IMG_SIZE = 28
IMG_CHANNELS = 1
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-2 # Оставим пониженный LR
WEIGHT_DECAY = 1e-3 # WD для блоков и классификатора
EMBED_WD = 1e-5 # Маленький WD для эмбеддингов
MAX_NORM_EMBED = 50.0 # Порог для клиппинга нормы эмбеддингов
GRAD_CLIP_MAX_NORM = 1.0 # Порог для клиппинга градиентов
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ETA_LOSS_WEIGHT больше не нужен напрямую, используем равномерный вес для MSE(epsilon)
ETA_LOSS_WEIGHT = 1.0

print(f"Using device: {DEVICE}")
print(f"Parameters: T={T}, EmbedDim={EMBED_DIM}, Epochs={EPOCHS}, LR={LR}, WD={WEIGHT_DECAY}, EmbedWD={EMBED_WD}")
print(f"NormClipEmbed={MAX_NORM_EMBED}, GradClip={GRAD_CLIP_MAX_NORM}")
print("!!! Training Target: Predicting Noise (epsilon) !!!")

# --- 1. Noise Schedule (Cosine schedule) ---
def get_alpha_bar_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return torch.clip(alphas_cumprod, 0.0001, 0.9999)

alphas_bar = get_alpha_bar_schedule(T).to(DEVICE) # Size T+1, [alpha_bar_0, ..., alpha_bar_T]
alphas = alphas_bar[1:] / alphas_bar[:-1]       # Size T,   [alpha_1, ..., alpha_T]
betas = 1.0 - alphas                            # Size T,   [beta_1, ..., beta_T]

# --- Precompute values needed for sampling q(z_t|y) and inference ---
# Индексы 0..T соответствуют временным шагам статьи 0..T
sqrt_alphas_bar = torch.sqrt(alphas_bar)
sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)

# Для DDPM inference step (t идет от T до 1, используем индекс t-1 для доступа)
# Нужны alpha_t, beta_t, alpha_bar_t, alpha_bar_{t-1}
sqrt_recip_alphas = 1.0 / torch.sqrt(alphas) # index 0..T-1 -> 1/sqrt(alpha_1)...1/sqrt(alpha_T)
# Variance of q(z_{t-1} | z_t, z_0): beta_tilde_t = beta_t * (1-alpha_bar_{t-1}) / (1-alpha_bar_t)
# Индекс t-1 соответствует t шагу статьи (1..T)
posterior_variance = betas * (1.0 - alphas_bar[:-1]) / (1.0 - alphas_bar[1:]) # index 0..T-1 -> beta_tilde_1...beta_tilde_T

print("Noise schedule and necessary coefficients precomputed.")

# --- 2. Label Embeddings ---
label_embedding = nn.Embedding(NUM_CLASSES, EMBED_DIM).to(DEVICE)
print(f"Applying orthogonal initialization to {EMBED_DIM}-dim embeddings.")
try:
    nn.init.orthogonal_(label_embedding.weight)
except ValueError as e:
    print(f"Warning: Orthogonal init failed ({e}). Using default init.")


# --- 3. Model Architecture (Блок предсказывает epsilon) ---
class DenoisingBlockPaper(nn.Module):
    # Убираем num_classes, так как выход теперь embed_dim (предсказание шума)
    # Убираем W_embed из forward, так как он не нужен для предсказания шума напрямую
    def __init__(self, embed_dim, img_channels=1, img_size=28):
        super().__init__()
        self.embed_dim = embed_dim

        # --- Image processing path (x) ---
        self.img_conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten()
        )
        with torch.no_grad():
             dummy_input = torch.zeros(1, img_channels, img_size, img_size)
             conv_output_size = self.img_conv(dummy_input).shape[-1]
        self.img_fc = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2)
        )

        # --- Noisy embedding processing path (z_input) ---
        self.label_embed_proc = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2)
        )

        # --- Combined path -> outputs predicted epsilon (size embed_dim) ---
        self.combined = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, self.embed_dim) # Output predicted noise
        )

    # z_input это z_{t-1} (paper index) sampled from q(z_{t-1}|y)
    # Блок t (paper index) предсказывает epsilon, добавленный на шаге t-1->t
    # Убираем W_embed из аргументов
    def forward(self, x, z_input):
        img_features = self.img_fc(self.img_conv(x))
        label_features = self.label_embed_proc(z_input)
        combined_features = torch.cat([img_features, label_features], dim=1)
        predicted_epsilon = self.combined(combined_features)
        return predicted_epsilon

class NoPropNet(nn.Module):
    # Убираем num_classes из конструктора блоков
    def __init__(self, num_blocks, embed_dim, num_classes, img_channels=1, img_size=28):
        super().__init__()
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.blocks = nn.ModuleList([
            DenoisingBlockPaper(embed_dim, img_channels, img_size) # Убран num_classes
            for _ in range(num_blocks)
        ])
        self.classifier = nn.Linear(embed_dim, num_classes)

    # Conceptual forward (not used)
    def forward(self, x, z_0):
        print("Warning: NoPropNet.forward is conceptual only. Use run_inference.")
        # ... (старый концептуальный код нерелевантен)
        return None

# --- 4. Dataset ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True if DEVICE=='cuda' else False)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True if DEVICE=='cuda' else False)

# --- 5. Initialization ---
model = NoPropNet(num_blocks=T, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES, img_channels=IMG_CHANNELS, img_size=IMG_SIZE).to(DEVICE)

# Optimizers
optimizers_blocks = [optim.AdamW(block.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) for block in model.blocks]
params_final_classifier = model.classifier.parameters()
params_final_embedding = label_embedding.parameters()
optimizer_final = optim.AdamW([
    {'params': params_final_classifier, 'weight_decay': WEIGHT_DECAY},
    {'params': params_final_embedding, 'weight_decay': EMBED_WD}
], lr=LR)
print(f"Initialized optimizer_final: Classifier WD={WEIGHT_DECAY}, Embeddings WD={EMBED_WD}.")

T_max_epochs = EPOCHS # Количество эпох для одного цикла косинуса (можно взять меньше, если планируется ранняя остановка)
eta_min_lr = 1e-6 # Минимальное значение LR

schedulers_blocks = [optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max_epochs, eta_min=eta_min_lr) for opt in optimizers_blocks]
scheduler_final = optim.lr_scheduler.CosineAnnealingLR(optimizer_final, T_max=T_max_epochs, eta_min=eta_min_lr)


mse_loss = nn.MSELoss() # Теперь для шума
ce_loss = nn.CrossEntropyLoss()

# --- 7. Inference Function (ПЕРЕПИСАНА под DDPM-like step с предсказанием epsilon) ---
# Перемещаем определение ДО цикла обучения
@torch.no_grad()
def run_inference(model, x_batch, T_steps, alphas, alphas_bar, posterior_variance, device):
    """
    Запускает процесс вывода, используя DDPM-подобный шаг с предсказанным шумом epsilon.
    """
    batch_size = x_batch.shape[0]
    embed_dim = model.embed_dim

    # 1. Начинаем с z_T ~ N(0, I)
    # Обратите внимание: в NoProp z_0 был шумом, а z_T - выходом.
    # В DDPM наоборот: x_T - шум, x_0 - выход.
    # Сохраним нотацию NoProp где z_0 - шум, z_T - результат T шагов.
    # Значит, inference должен идти от z_0 к z_T? Нет, по логике DDPM/Sohl-Dickstein
    # обратный процесс идет от шума (z_T) к данным (z_0).
    # Давайте следовать DDPM: начинаем с z_T, идем к z_0.

    z = torch.randn(batch_size, embed_dim, device=device) # z_T (чистый шум)

    # 2. Итеративно идем назад от T до 1
    for t_idx_rev in range(T_steps - 1, -1, -1): # t_idx_rev = T-1, T-2, ..., 0
        # Это соответствует времени t = T, T-1, ..., 1 в DDPM
        current_t_paper = t_idx_rev + 1
        block_to_use = model.blocks[t_idx_rev] # Блок t_idx предсказывает шум для шага t=t_idx+1

        # Вход для блока DDPM - это z_t (текущее состояние)
        # Нам нужно передать и время/уровень шума? В VDM передают gamma_t.
        # В текущей архитектуре DenoisingBlockPaper нет входа для времени.
        # Это может быть проблемой! Блок не знает, для какого шага предсказывать шум.
        # Пока проигнорируем, но это важное ограничение.
        predicted_epsilon = block_to_use(x_batch, z) # Передаем текущий z (z_t)

        alpha_t = alphas[t_idx_rev]             # alpha_t = alpha_{t_idx+1}
        alpha_bar_t = alphas_bar[t_idx_rev + 1] # alpha_bar_t = alpha_bar_{t_idx+1}
        beta_t = 1.0 - alpha_t                  # beta_t = beta_{t_idx+1}
        sqrt_recip_alpha_t = 1.0 / torch.sqrt(alpha_t)
        sqrt_1m_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        # DDPM reverse step:
        # z_{t-1} = 1/sqrt(alpha_t) * (z_t - beta_t / sqrt(1-alpha_bar_t) * predicted_epsilon) + sigma_t * noise
        mean = sqrt_recip_alpha_t * (z - beta_t / sqrt_1m_alpha_bar_t * predicted_epsilon)

        if t_idx_rev == 0: # Последний шаг (t=1 -> t=0)
            noise = torch.zeros_like(z)
            # ИСПРАВЛЕНИЕ: Устанавливаем нулевой тензор вместо float
            variance = torch.tensor(0.0, device=device)
        else:
            noise = torch.randn_like(z)
            # Убедимся, что posterior_variance - это тензор
            variance = posterior_variance[t_idx_rev] # variance остается тензором

        # Также добавим clamp для гарантии неотрицательности перед корнем
        variance_stable = torch.clamp(variance + 1e-6, min=0.0)

        z = mean + torch.sqrt(variance_stable + 1e-6) * noise # Добавим epsilon для стабильности корня

    # 3. После T шагов, z представляет z_0 (предсказанный чистый эмбеддинг)
    # Используем классификатор на z_0
    logits = model.classifier(z)
    return logits

# --- 6. NOPROP Training Loop (Цель: шум epsilon, Классификатор: на предсказанном u_hat_T) ---
print(f"Starting NoProp Training for {EPOCHS} epochs...")
total_start_time = time.time()

best_test_accuracy = 0.0
epochs_no_improve = 0
patience = 15

for epoch in range(EPOCHS):
    model.train()
    epoch_start_time = time.time()
    print(f"--- Starting Epoch {epoch + 1}/{EPOCHS} ---")

    # Внешний цикл по временному шагу t
    for t_idx in range(T): # t_idx = 0..T-1 соответствует шагу t=1..T статьи
        block_to_train = model.blocks[t_idx]
        optimizer_t = optimizers_blocks[t_idx]

        running_loss_denoise_t = 0.0
        running_loss_classify_t = 0.0
        processed_batches_t = 0
        block_start_time = time.time()

        # Внутренний цикл по мини-батчам
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            batch_size = inputs.shape[0]

            u_y = label_embedding(labels)
            # current_W_embed больше не нужен блокам напрямую

            # --- Denoising Loss (Цель: шум epsilon) ---
            # 1. Сэмплируем epsilon
            epsilon_target = torch.randn_like(u_y)

            # 2. Сэмплируем z_{t-1} ~ q(z_{t-1} | y) используя epsilon_target
            # alpha_bar_{t-1} статьи соответствует alphas_bar[t_idx] кода
            alpha_bar_for_sample = alphas_bar[t_idx]
            sqrt_a_bar = sqrt_alphas_bar[t_idx].view(-1, 1).expand_as(u_y) # Используем предвычисленные
            sqrt_1_minus_a_bar = sqrt_one_minus_alphas_bar[t_idx].view(-1, 1).expand_as(u_y)
            # Вход для блока t_idx (предсказывает epsilon для шага t=t_idx+1)
            # Должен ли вход быть z_{t-1} или z_t?
            # Стандартный DDPM: epsilon_theta(z_t, t) предсказывает epsilon, который был добавлен для получения z_t
            # NoProp: блок t работает с z_{t-1} (исходная идея).
            # Давайте пока оставим вход z_{t-1}, как было раньше в NoProp.
            z_input_for_block = sqrt_a_bar * u_y.detach() + sqrt_1_minus_a_bar * epsilon_target

            # 3. Предсказываем epsilon с помощью блока t_idx
            # Убрали current_W_embed из вызова
            predicted_epsilon = block_to_train(inputs, z_input_for_block)

            # 4. Вычисляем потери MSE между предсказанным и реальным шумом
            loss_t = mse_loss(predicted_epsilon, epsilon_target)
            running_loss_denoise_t += loss_t.item() # Логируем невзвешенные

            # Используем равномерный вес (контролируемый ETA, который сейчас 1.0)
            weighted_loss_t = ETA_LOSS_WEIGHT * loss_t

            # --- Classification Loss (на основе предсказанного u_hat_T) ---
            # Для этого нужен z_{T-1} и предсказание шума epsilon_T от блока T-1
            with torch.no_grad():
                # Сэмплируем z_{T-1}
                alpha_bar_T_minus_1 = alphas_bar[T-1]
                sqrt_a_bar_Tm1 = sqrt_alphas_bar[T-1].view(-1, 1).expand_as(u_y)
                sqrt_1_minus_a_bar_Tm1 = sqrt_one_minus_alphas_bar[T-1].view(-1, 1).expand_as(u_y)
                epsilon_Tm1_target = torch.randn_like(u_y) # Шум для T-1
                z_T_minus_1_sample = sqrt_a_bar_Tm1 * u_y.detach() + sqrt_1_minus_a_bar_Tm1 * epsilon_Tm1_target

                # Получаем предсказание шума epsilon_T от последнего блока
                # Убрали current_W_embed
                predicted_epsilon_T = model.blocks[T-1](inputs, z_T_minus_1_sample)

                # Восстанавливаем предсказанный чистый u_hat_T
                # Используем формулу x_0 = (x_t - sqrt(1-alpha_bar_t)*epsilon) / sqrt(alpha_bar_t)
                # Здесь x_t это z_{T-1}, t это T-1
                predicted_u_y_final = (z_T_minus_1_sample - sqrt_1_minus_a_bar_Tm1 * predicted_epsilon_T) / (sqrt_a_bar_Tm1 + 1e-6) # Добавим epsilon

            # Классификатор на предсказанном чистом эмбеддинге (с detach)
            logits = model.classifier(predicted_u_y_final.detach())
            classify_loss = ce_loss(logits, labels)
            running_loss_classify_t += classify_loss.item()

            # --- Обновление параметров (Объединенное) ---
            # Вернемся к объединенному обновлению как в Алгоритме 1
            total_loss_batch = weighted_loss_t + classify_loss

            optimizer_t.zero_grad()
            optimizer_final.zero_grad()

            total_loss_batch.backward() # Градиенты от обеих потерь

            # Клиппинг градиентов
            torch.nn.utils.clip_grad_norm_(block_to_train.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            if label_embedding.weight.requires_grad:
                torch.nn.utils.clip_grad_norm_(label_embedding.parameters(), max_norm=GRAD_CLIP_MAX_NORM)

            # Шаг оптимизаторов
            optimizer_t.step()
            optimizer_final.step()

            # Клиппинг нормы эмбеддингов
            with torch.no_grad():
                current_norm = label_embedding.weight.data.norm()
                if current_norm > MAX_NORM_EMBED:
                    label_embedding.weight.data.mul_(MAX_NORM_EMBED / (current_norm + 1e-6))

            processed_batches_t += 1
            # ... (конец цикла по батчам) ...

        # Логирование после блока t_idx
        block_time = time.time() - block_start_time
        avg_denoise_loss_block = running_loss_denoise_t / processed_batches_t
        avg_classify_loss_block = running_loss_classify_t / processed_batches_t
        print(f'  Epoch {epoch + 1}, Block {t_idx + 1}/{T} trained. AvgLoss: Denoise(MSE_eps)={avg_denoise_loss_block:.7f}, Classify={avg_classify_loss_block:.4f}. Time: {block_time:.2f}s')
        # ... (конец цикла по t_idx) ...

    # --- Конец Эпохи Обучения ---
    epoch_time = time.time() - epoch_start_time
    with torch.no_grad():
        embedding_norm = torch.norm(label_embedding.weight.data)
        print(f"--- Epoch {epoch + 1} finished. Training Time: {epoch_time:.2f}s ---")
        print(f"--- End of Epoch {epoch + 1}. Embedding Norm: {embedding_norm:.4f} ---")

    # Шаг планировщиков
    for scheduler in schedulers_blocks:
        scheduler.step()
    scheduler_final.step()

    # Выводим текущий LR
    current_lr = optimizer_final.param_groups[0]['lr']
    print(f"--- End of Epoch {epoch + 1}. Current LR: {current_lr:.6f} ---")

    # --- ОЦЕНКА ПОСЛЕ КАЖДОЙ ЭПОХИ ---
    print(f"--- Running Evaluation for Epoch {epoch + 1} ---")
    eval_start_time = time.time()
    correct = 0
    total = 0
    model.eval()

    final_W_embed = label_embedding.weight.data.detach() # Не используется в инференсе теперь

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # ИЗМЕНЕНИЕ: Вызов нового инференса
            logits = run_inference(model, images, T, alphas, alphas_bar, posterior_variance, DEVICE)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    current_test_accuracy = 100 * correct / total
    eval_time = time.time() - eval_start_time
    print(f'>>> Epoch {epoch + 1} Test Accuracy: {current_test_accuracy:.2f} % ({correct}/{total}) <<<')
    print(f"Evaluation Time: {eval_time:.2f}s")

    # Логика ранней остановки
    if current_test_accuracy > best_test_accuracy:
        best_test_accuracy = current_test_accuracy
        epochs_no_improve = 0
        print(f"*** New best test accuracy: {best_test_accuracy:.2f} % ***")
    else:
        epochs_no_improve += 1
        print(f"Test accuracy did not improve for {epochs_no_improve} epochs.")

    if epochs_no_improve >= patience:
        print(f"Stopping early after {epoch + 1} epochs due to no improvement in test accuracy for {patience} epochs.")
        break

# --- Конец всего обучения ---
print('Finished Training')
total_training_time = time.time() - total_start_time
print(f"Total Training Time: {total_training_time:.2f}s")
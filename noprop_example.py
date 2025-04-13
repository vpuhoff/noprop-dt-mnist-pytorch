import torch
import torch.nn as nn
import torch.nn.functional as F # Добавлено для LayerNorm и др.
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import math
import time
from typing import Dict, Tuple, List, Any
import optuna # Добавляем импорт Optuna
from optuna.samplers import GridSampler # Для поиска по сетке
import plotly 
import kaleido  # для экспорта, если понадобится
from tqdm import tqdm

# --- 1. Базовая Конфигурация (Значения по умолчанию и не тюнингуемые) ---
# Эти значения будут использоваться, если Optuna их не переопределит
base_config: Dict[str, Any] = {
    # Diffusion settings
    "T": 5,
    "s_noise_schedule": 0.008,

    # Model architecture
    "EMBED_DIM": 20, # Это target_embed_dim для нового блока
    "NUM_CLASSES": 10,
    "IMG_SIZE": 28,
    "IMG_CHANNELS": 1,

    # Training settings (некоторые будут переопределены Optuna)
    "BATCH_SIZE": 128,
    "EPOCHS": 10, # <--- ФИКСИРОВАНО для HPO
    "LR": 1e-3,   # Будет переопределено
    "WEIGHT_DECAY": 1e-3, # Фиксировано для блоков/классификатора
    "EMBED_WD": 1e-5,     # Будет переопределено
    "MAX_NORM_EMBED": 50.0, # Фиксировано
    "GRAD_CLIP_MAX_NORM": 1.0, # Фиксировано
    "ETA_LOSS_WEIGHT": 1.0, # Будет переопределено
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    # Scheduler settings (НЕ ИСПОЛЬЗУЮТСЯ в HPO прогоне)
    # "T_max_epochs": 100,
    # "eta_min_lr": 1e-6,

    # Early stopping (НЕ ИСПОЛЬЗУЕТСЯ в HPO прогоне)
    # "patience": 15,

    # Data settings
    "data_root": './data',
    "num_workers": 2,
}

# --- Параметры для HPO (если используется) ---
STUDY_NAME = "find_params_attention_v3" # Новое имя study
LR_TRIALS = [0.002, 0.001, 0.0001] # Возможно, потребуются другие LR
ETA_LOSS_WEIGHT_TRIALS = [ 50.0, 100.0, 200.0]
EMBED_WD_TRIALS = [1e-6, 0.0]

# --- 2. Helper Functions (Без существенных изменений) ---
progress_bar = None

def tqdm_callback(study, trial):
    if progress_bar:
      progress_bar.update(1)

def get_alpha_bar_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return torch.clip(alphas_cumprod, 0.0001, 0.9999)

def precompute_diffusion_coefficients(T: int, device: torch.device, s: float = 0.008) -> Dict[str, torch.Tensor]:
    alphas_bar = get_alpha_bar_schedule(T, s).to(device)
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    betas = 1.0 - alphas
    sqrt_alphas_bar = torch.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)
    # Handle potential division by zero or instability if alpha_t is exactly 1.0
    # Clamping alpha_t slightly below 1.0 if needed for sqrt_recip_alphas
    alphas_clamped = torch.clamp(alphas, max=1.0 - 1e-9) 
    sqrt_recip_alphas = 1.0 / torch.sqrt(alphas_clamped)
    
    posterior_variance = betas * (1.0 - alphas_bar[:-1]) / torch.clamp(1.0 - alphas_bar[1:], min=1e-6)
    print("Noise schedule and necessary coefficients precomputed.")
    return {
        'alphas_bar': alphas_bar, 'alphas': alphas, 'betas': betas,
        'sqrt_alphas_bar': sqrt_alphas_bar, 'sqrt_one_minus_alphas_bar': sqrt_one_minus_alphas_bar,
        'sqrt_recip_alphas': sqrt_recip_alphas, 'posterior_variance': posterior_variance
    }

def initialize_embeddings(num_classes: int, embed_dim: int, device: torch.device) -> nn.Embedding:
    label_embedding = nn.Embedding(num_classes, embed_dim).to(device)
    print(f"Initializing {embed_dim}-dim embeddings.")
    try:
        if embed_dim >= num_classes:
             nn.init.orthogonal_(label_embedding.weight)
             print("Applied orthogonal initialization.")
        else:
             print("Default initialization (orthogonal not possible).")
    except ValueError as e:
        print(f"Warning: Orthogonal init failed ({e}). Using default init.")
    return label_embedding

def get_dataloaders(batch_size: int, root: str, num_workers: int, pin_memory: bool) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    testset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    print(f"Loaded MNIST: {len(trainset)} train, {len(testset)} test samples.")
    return trainloader, testloader

def initialize_training_components(
    model: nn.Module, label_embedding: nn.Embedding, config: Dict[str, Any], use_scheduler: bool = True
) -> Tuple[List[optim.Optimizer], optim.Optimizer, List[Any], Any, nn.Module, nn.Module]:
    """Initializes optimizers, optional schedulers, and loss functions."""
    LR = config['LR'] # Используем LR из config (может быть от Optuna)
    WEIGHT_DECAY = config['WEIGHT_DECAY']
    EMBED_WD = config['EMBED_WD']

    # Optimizers
    optimizers_blocks = [optim.AdamW(block.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) for block in model.blocks]
    params_final_classifier = model.classifier.parameters()
    params_final_embedding = label_embedding.parameters()
    optimizer_final = optim.AdamW([
        {'params': params_final_classifier, 'weight_decay': WEIGHT_DECAY},
        {'params': params_final_embedding, 'weight_decay': EMBED_WD}
    ], lr=LR)
    print(f"Initialized optimizers: LR={LR}, Blocks WD={WEIGHT_DECAY}, Classifier WD={WEIGHT_DECAY}, Embeddings WD={EMBED_WD}.")

    # Schedulers (Optional)
    schedulers_blocks = []
    scheduler_final = None
    if use_scheduler:
        # Убедимся что параметры планировщика есть в config
        if 'T_max_epochs' not in config or 'eta_min_lr' not in config:
             raise ValueError("Scheduler parameters 'T_max_epochs' and 'eta_min_lr' missing from config.")
        T_max_epochs = config['T_max_epochs']
        eta_min_lr = config['eta_min_lr']
        schedulers_blocks = [optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max_epochs, eta_min=eta_min_lr) for opt in optimizers_blocks]
        scheduler_final = optim.lr_scheduler.CosineAnnealingLR(optimizer_final, T_max=T_max_epochs, eta_min=eta_min_lr)
        print(f"Initialized CosineAnnealingLR schedulers with T_max={T_max_epochs}, eta_min={eta_min_lr}.")
    else:
        print("LR Scheduler is DISABLED for this run.")


    # Loss functions
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    return optimizers_blocks, optimizer_final, schedulers_blocks, scheduler_final, mse_loss, ce_loss

# --- 3. Model Architecture ---

# <<< НОВЫЙ БЛОК С ATTENTION >>>
class AttentionDenoisingBlock(nn.Module):
    def __init__(self,
                 target_embed_dim: int, # Размерность предсказываемого эпсилон (например, 20)
                 img_channels: int = 1,
                 img_size: int = 28,
                 patch_size: int = 4,      # Размер патча (должен делить img_size)
                 img_feature_dim: int = 128, # Размерность эмбеддинга для патчей и выхода self-attn
                 label_feature_dim: int = 128,# Размерность выхода обработки z_input
                 nhead: int = 4,            # Количество голов во внимании
                 num_encoder_layers: int = 1, # Количество слоев трансформера для self-attn
                 dim_feedforward: int = 256, # Размерность в MLP внутри трансформера
                 dropout: float = 0.1):
        super().__init__()

        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")

        self.target_embed_dim = target_embed_dim
        self.img_feature_dim = img_feature_dim
        self.label_feature_dim = label_feature_dim
        num_patches = (img_size // patch_size) ** 2
        patch_dim = img_channels * patch_size ** 2

        # --- 1. Self-Attention для обработки изображений ---

        # 1.1 Линейная проекция патчей (Conv2d)
        self.patch_projection = nn.Conv2d(img_channels, img_feature_dim,
                                          kernel_size=patch_size, stride=patch_size)

        # 1.2 Позиционное кодирование для патчей (Learnable Parameter)
        # Инициализация как Parameter требует, чтобы это было сделано внутри __init__
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, img_feature_dim))

        # 1.3 Трансформерный кодировщик (для self-attention)
        encoder_layer = nn.TransformerEncoderLayer(d_model=img_feature_dim,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='relu',
                                                   batch_first=True) # Удобнее для Conv2d выхода
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_encoder_layers)

        # Нормализация после трансформера
        self.ln_vision = nn.LayerNorm(img_feature_dim)

        # --- 2. Обработка входного эмбеддинга z_input ---
        # Выходной размерностью label_feature_dim
        self.label_embed_proc = nn.Sequential(
            nn.Linear(target_embed_dim, label_feature_dim // 2), # Вход target_embed_dim
            # Используем LayerNorm вместо BatchNorm, т.к. может быть малый батч в конце эпохи
            nn.LayerNorm(label_feature_dim // 2), nn.ReLU(), nn.Dropout(dropout), 
            nn.Linear(label_feature_dim // 2, label_feature_dim),
            nn.LayerNorm(label_feature_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        # self.ln_label = nn.LayerNorm(label_feature_dim) # Уже есть в Sequential

        # --- 3. Cross-Attention для слияния признаков ---
        if img_feature_dim != label_feature_dim:
             raise ValueError(f"img_feature_dim ({img_feature_dim}) must match "
                             f"label_feature_dim ({label_feature_dim}) for simple cross-attention.")

        self.cross_attention = nn.MultiheadAttention(embed_dim=label_feature_dim,
                                                     num_heads=nhead,
                                                     dropout=dropout,
                                                     batch_first=True) # Используем batch_first=True
        self.ln_cross = nn.LayerNorm(label_feature_dim)

        # --- 4. Финальный MLP для предсказания эпсилон ---
        self.final_mlp = nn.Sequential(
            nn.Linear(label_feature_dim, label_feature_dim // 2),
            nn.ReLU(),
            nn.Linear(label_feature_dim // 2, target_embed_dim) # Выходная размерность = target_embed_dim
        )

    def forward(self, x: torch.Tensor, z_input: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # 1. Обработка изображения (Self-Attention)
        patches = self.patch_projection(x) # -> (B, img_feature_dim, H/P, W/P)
        patches = patches.flatten(2)       # -> (B, img_feature_dim, NumPatches)
        patches = patches.transpose(1, 2)  # -> (B, NumPatches, img_feature_dim)

        # Добавляем позиционное кодирование
        patches = patches + self.positional_encoding

        # Пропускаем через трансформерный кодировщик
        img_features_seq = self.transformer_encoder(patches) # -> (B, NumPatches, img_feature_dim)

        # Берем среднее по последовательности патчей
        img_features = img_features_seq.mean(dim=1) # -> (B, img_feature_dim)
        img_features = self.ln_vision(img_features) # -> (B, img_feature_dim)

        # 2. Обработка эмбеддинга z_input
        label_features = self.label_embed_proc(z_input) # -> (B, label_feature_dim)
        # label_features = self.ln_label(label_features) # Уже есть LayerNorm внутри

        # 3. Слияние через Cross-Attention
        img_features_seq_for_attn = img_features.unsqueeze(1)    # -> (B, 1, img_feature_dim)
        label_features_seq_for_attn = label_features.unsqueeze(1) # -> (B, 1, label_feature_dim)

        # label_features (от z) делает запрос к img_features (от x)
        attn_input = label_features_seq_for_attn # Сохраняем для residual
        attn_output, _ = self.cross_attention(
            query=attn_input,
            key=img_features_seq_for_attn,
            value=img_features_seq_for_attn
        ) # -> (B, 1, label_feature_dim)

        # Применяем LayerNorm + residual
        merged_features = self.ln_cross(attn_input + attn_output) # -> (B, 1, label_feature_dim)
        merged_features = merged_features.squeeze(1) # -> (B, label_feature_dim)

        # 4. Финальное предсказание эпсилон
        predicted_epsilon = self.final_mlp(merged_features) # -> (B, target_embed_dim)

        return predicted_epsilon

# <<< МОДИФИЦИРОВАННЫЙ NoPropNet >>>
class NoPropNet(nn.Module):
    def __init__(self,
                 num_blocks: int,
                 block_class: nn.Module, # Класс блока
                 block_kwargs: dict,     # Аргументы для __init__ блока
                 classifier_input_dim: int, # Размерность входа для классификатора
                 num_classes: int):
        super().__init__()
        self.num_blocks = num_blocks
        # Определяем embed_dim (размерность эпсилон) из block_kwargs
        # Используем 'target_embed_dim' как ключ для нового блока
        self.embed_dim = block_kwargs.get('target_embed_dim', None)
        if self.embed_dim is None:
             raise ValueError("Key 'target_embed_dim' not found in block_kwargs")

        self.num_classes = num_classes

        self.blocks = nn.ModuleList([
            block_class(**block_kwargs) for _ in range(num_blocks)
        ])

        # Классификатор принимает на вход z0, который имеет размерность embed_dim
        self.classifier = nn.Linear(classifier_input_dim, num_classes)

    # forward остается концептуальным
    def forward(self, x: torch.Tensor, z_0: torch.Tensor) -> None:
        print("Warning: NoPropNet.forward is conceptual only. Use run_inference or training loop.")
        return None

# --- 4. Core Logic Functions (Без изменений в логике, но теперь работают с новой моделью) ---

@torch.no_grad()
def run_inference(
    model: NoPropNet,
    x_batch: torch.Tensor,
    T_steps: int,
    diff_coeffs: Dict[str, torch.Tensor],
    device: torch.device
) -> torch.Tensor:
    """Runs the reverse diffusion process (DDPM-like) for inference."""
    batch_size = x_batch.shape[0]
    embed_dim = model.embed_dim # Получаем из модели
    alphas = diff_coeffs['alphas']
    alphas_bar = diff_coeffs['alphas_bar']
    posterior_variance = diff_coeffs['posterior_variance']
    sqrt_recip_alphas = diff_coeffs['sqrt_recip_alphas'] # Получаем предвычисленное значение
    sqrt_one_minus_alphas_bar = diff_coeffs['sqrt_one_minus_alphas_bar'] # Получаем предвычисленное значение

    z = torch.randn(batch_size, embed_dim, device=device) # z_T

    for t_idx_rev in range(T_steps - 1, -1, -1):
        block_to_use = model.blocks[t_idx_rev]
        # Блок теперь принимает (x_batch, z)
        predicted_epsilon = block_to_use(x_batch, z) 

        alpha_t = alphas[t_idx_rev]
        alpha_bar_t = alphas_bar[t_idx_rev + 1] # alpha_bar на шаге t
        beta_t = 1.0 - alpha_t
        sqrt_recip_alpha_t = sqrt_recip_alphas[t_idx_rev] # Используем предвычисленное
        sqrt_1m_alpha_bar_t = sqrt_one_minus_alphas_bar[t_idx_rev + 1] # Используем предвычисленное

        # Формула среднего из DDPM
        mean = sqrt_recip_alpha_t * (z - beta_t / (sqrt_1m_alpha_bar_t + 1e-9) * predicted_epsilon)

        if t_idx_rev == 0:
            # На последнем шаге нет шума
            noise = torch.zeros_like(z)
            variance = torch.tensor(0.0, device=device) # Не используется, но для полноты
        else:
            variance = posterior_variance[t_idx_rev] # variance = posterior_variance[t-1] in some notations
            noise = torch.randn_like(z)

        # Избегаем корня из отрицательного числа из-за числовых неточностей
        variance_stable = torch.clamp(variance, min=1e-9) 
        z = mean + torch.sqrt(variance_stable) * noise # z_{t-1}

    # Классифицируем финальный z_0
    logits = model.classifier(z) 
    return logits

def train_epoch(
    epoch_num: int, # Only for logging
    model: NoPropNet,
    label_embedding: nn.Embedding,
    trainloader: DataLoader,
    optimizers_blocks: List[optim.Optimizer],
    optimizer_final: optim.Optimizer,
    mse_loss: nn.Module,
    ce_loss: nn.Module,
    diff_coeffs: Dict[str, torch.Tensor],
    config: Dict[str, Any]
) -> None:
    """Runs one epoch of training."""
    model.train() # Устанавливаем модель в режим обучения

    T = config['T']
    DEVICE = config['DEVICE']
    ETA_LOSS_WEIGHT = config['ETA_LOSS_WEIGHT'] # Weight from config
    GRAD_CLIP_MAX_NORM = config['GRAD_CLIP_MAX_NORM']
    MAX_NORM_EMBED = config['MAX_NORM_EMBED']

    alphas_bar = diff_coeffs['alphas_bar']
    sqrt_alphas_bar = diff_coeffs['sqrt_alphas_bar']
    sqrt_one_minus_alphas_bar = diff_coeffs['sqrt_one_minus_alphas_bar']

    # --- Основной цикл обучения по блокам ---
    for t_idx in tqdm(range(T)): # t_idx от 0 до T-1 (соответствует шагам 1 до T)
        block_to_train = model.blocks[t_idx]
        optimizer_t = optimizers_blocks[t_idx]

        running_loss_denoise_t = 0.0
        running_loss_classify_t = 0.0
        processed_batches_t = 0
        block_start_time = time.time()

        # --- Цикл по батчам данных ---
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            batch_size = inputs.shape[0]
            
            # Получаем "чистые" эмбеддинги для данного батча
            # Не используем .detach() здесь, т.к. эмбеддинги должны обучаться через optimizer_final
            u_y = label_embedding(labels) 

            # --- Denoising Loss для текущего блока t ---
            epsilon_target = torch.randn_like(u_y) # Целевой шум эпсилон
            
            # Индекс для alpha_bar соответствует t-1 в формуле z_{t-1}
            # В коде t_idx = 0 соответствует t=1, поэтому используем alphas_bar[t_idx]
            alpha_bar_tm1_val = alphas_bar[t_idx] 
            sqrt_a_bar = sqrt_alphas_bar[t_idx].view(-1, 1).expand_as(u_y)
            sqrt_1_minus_a_bar = sqrt_one_minus_alphas_bar[t_idx].view(-1, 1).expand_as(u_y)
            
            # Создаем вход z_{t-1} для блока t
            # Используем .detach() для u_y при создании входа для блока, 
            # т.к. градиенты для эмбеддингов пойдут только от classification loss
            z_input_for_block = sqrt_a_bar * u_y.detach() + sqrt_1_minus_a_bar * epsilon_target 
            
            # Предсказание шума блоком t
            predicted_epsilon = block_to_train(inputs, z_input_for_block) 
            
            # Считаем MSE лосс
            loss_t = mse_loss(predicted_epsilon, epsilon_target)
            running_loss_denoise_t += loss_t.item()
            
            # Взвешиваем лосс блока (простой вес ETA)
            weighted_loss_t = ETA_LOSS_WEIGHT * loss_t

            # --- Classification Loss (рассчитывается в каждом батче для обновления классификатора и эмбеддингов) ---
            with torch.no_grad(): # Не хотим считать градиенты для этой части внутри блока T
                # Используем последний блок (индекс T-1) для предсказания epsilon_T
                # Вход для него - z_{T-1}
                # Индекс для alpha_bar соответствует T-1
                alpha_bar_Tm1_val = alphas_bar[T-1] 
                sqrt_a_bar_Tm1 = sqrt_alphas_bar[T-1].view(-1, 1).expand_as(u_y)
                sqrt_1_minus_a_bar_Tm1 = sqrt_one_minus_alphas_bar[T-1].view(-1, 1).expand_as(u_y)
                
                epsilon_Tm1_target_for_clf = torch.randn_like(u_y) # Шум для z_{T-1}
                
                # Создаем z_{T-1}
                # Не используем detach для u_y здесь, так как градиенты для u_y пойдут от CE loss ниже
                z_T_minus_1_sample = sqrt_a_bar_Tm1 * u_y + sqrt_1_minus_a_bar_Tm1 * epsilon_Tm1_target_for_clf
                
                # Предсказание шума последним блоком T (индекс T-1)
                predicted_epsilon_T = model.blocks[T-1](inputs, z_T_minus_1_sample)
                
                # Реконструкция предсказанного u_y (на основе z_{T-1} и predicted_epsilon_T)
                # Это предсказанный z0
                predicted_u_y_final = (z_T_minus_1_sample - sqrt_1_minus_a_bar_Tm1 * predicted_epsilon_T) / (sqrt_a_bar_Tm1 + 1e-9)

            # Подаем реконструированный u_y в классификатор
            # Используем .detach(), чтобы градиенты отсюда не шли в последний блок T
            # Градиенты пойдут только в сам классификатор и в label_embedding (через z_T_minus_1_sample -> u_y)
            logits = model.classifier(predicted_u_y_final.detach()) 
            classify_loss = ce_loss(logits, labels)
            running_loss_classify_t += classify_loss.item()

            # --- Комбинированный шаг обновления ---
            # Общий лосс для вызова backward()
            total_loss_batch = weighted_loss_t + classify_loss
            
            # Обнуляем градиенты для обоих оптимизаторов
            optimizer_t.zero_grad()     # Оптимизатор текущего блока t
            optimizer_final.zero_grad() # Оптимизатор классификатора и эмбеддингов

            # Вычисляем градиенты
            total_loss_batch.backward()

            # Клиппинг градиентов (применяем ко всем обучаемым параметрам)
            torch.nn.utils.clip_grad_norm_(block_to_train.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            # Проверяем, есть ли градиенты у эмбеддингов перед клиппингом
            if label_embedding.weight.grad is not None:
                 torch.nn.utils.clip_grad_norm_(label_embedding.parameters(), max_norm=GRAD_CLIP_MAX_NORM)

            # Шаг оптимизаторов
            optimizer_t.step()     # Обновляем веса блока t
            optimizer_final.step() # Обновляем веса классификатора и эмбеддингов

            # Клиппинг нормы эмбеддингов (после шага оптимизатора)
            with torch.no_grad():
                current_norm = label_embedding.weight.data.norm()
                if current_norm > MAX_NORM_EMBED:
                    scale = MAX_NORM_EMBED / (current_norm + 1e-9)
                    label_embedding.weight.data.mul_(scale)

            processed_batches_t += 1
            # --- Конец цикла по батчам ---

        block_time = time.time() - block_start_time
        avg_denoise_loss_block = running_loss_denoise_t / processed_batches_t if processed_batches_t > 0 else 0
        avg_classify_loss_block = running_loss_classify_t / processed_batches_t if processed_batches_t > 0 else 0
        # Менее подробный лог во время HPO
        if epoch_num == 0 and t_idx == 0: # Лог только первого блока первой эпохи
             print(f'  (Trial Log Sample) Epoch {epoch_num + 1}, Block {t_idx + 1}/{T}. AvgLoss: Denoise(MSE_eps)={avg_denoise_loss_block:.7f}, Classify(CE)={avg_classify_loss_block:.4f}. Time: {block_time:.2f}s')
    # --- Конец цикла по блокам ---


@torch.no_grad()
def evaluate_model(
    model: NoPropNet,
    testloader: DataLoader,
    T_steps: int,
    diff_coeffs: Dict[str, torch.Tensor],
    device: torch.device
) -> Tuple[float, int, int]:
    """Evaluates the model on the test set using the inference function."""
    model.eval() # Переводим модель в режим оценки
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # Запускаем инференс
        logits = run_inference(model, images, T_steps, diff_coeffs, device)
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy, correct, total

# --- HPO Objective Function ---
def objective(trial: optuna.trial.Trial) -> float:
    """Optuna objective function to run one trial."""
    # Создаем конфиг для данного trial, копируя базовый
    run_config = base_config.copy()

    # --- Гиперпараметры для тюнинга ---
    run_config['LR'] = trial.suggest_categorical('LR', LR_TRIALS)
    run_config['ETA_LOSS_WEIGHT'] = trial.suggest_categorical('ETA_LOSS_WEIGHT', ETA_LOSS_WEIGHT_TRIALS)
    run_config['EMBED_WD'] = trial.suggest_categorical('EMBED_WD', EMBED_WD_TRIALS)
    # --- Фиксированные параметры для HPO ---
    run_config['EPOCHS'] = 10 # Короткий прогон для HPO (было 20, уменьшил для скорости)

    print(f"\n--- Starting Optuna Trial {trial.number} with config: ---")
    print(f"LR: {run_config['LR']:.1e}, ETA: {run_config['ETA_LOSS_WEIGHT']:.1f}, EmbedWD: {run_config['EMBED_WD']:.1e}")

    # --- Настройка специфичная для trial ---
    DEVICE = run_config['DEVICE']
    diff_coeffs = precompute_diffusion_coefficients(run_config['T'], DEVICE, run_config['s_noise_schedule'])
    label_embedding = initialize_embeddings(run_config['NUM_CLASSES'], run_config['EMBED_DIM'], DEVICE)

    # --- Инициализация МОДЕЛИ с Attention блоком ---
    block_kwargs_hpo = {
         "target_embed_dim": run_config['EMBED_DIM'], # e.g., 20
         "img_channels": run_config['IMG_CHANNELS'],
         "img_size": run_config['IMG_SIZE'],
         "patch_size": 4,          # Fixed for HPO
         "img_feature_dim": 64,    # Fixed & smaller for HPO
         "label_feature_dim": 64,  # Must match img_feature_dim
         "nhead": 4,               # Fixed for HPO
         "num_encoder_layers": 1,  # Fixed & smaller for HPO
         "dim_feedforward": 128,   # Fixed & smaller for HPO
         "dropout": 0.1            # Fixed dropout
    }
    try:
        model = NoPropNet(
             num_blocks=run_config['T'],
             block_class=AttentionDenoisingBlock, # Используем новый блок
             block_kwargs=block_kwargs_hpo,
             classifier_input_dim=run_config['EMBED_DIM'], # Классификатор принимает z0 (размерность эпсилон)
             num_classes=run_config['NUM_CLASSES']
        ).to(DEVICE)
    except ValueError as e:
        print(f"Error initializing model for trial {trial.number}: {e}")
        # Возвращаем очень низкое значение, чтобы Optuna проигнорировал этот trial
        return -1.0 


    trainloader, testloader = get_dataloaders(
        run_config['BATCH_SIZE'], run_config['data_root'], run_config['num_workers'], (DEVICE == torch.device('cuda'))
    )
    # Инициализируем оптимизаторы БЕЗ планировщика для HPO
    optimizers_blocks, optimizer_final, _, _, mse_loss, ce_loss = \
        initialize_training_components(model, label_embedding, run_config, use_scheduler=False)

    # --- Цикл обучения и оценки для trial ---
    best_trial_accuracy = 0.0
    try:
        for epoch in range(run_config['EPOCHS']):
            print(f"  Trial {trial.number}, Epoch {epoch + 1}/{run_config['EPOCHS']}")
            train_epoch(
                epoch, model, label_embedding, trainloader, optimizers_blocks, optimizer_final,
                mse_loss, ce_loss, diff_coeffs, run_config
            )
            current_test_accuracy, _, _ = evaluate_model(
                model, testloader, run_config['T'], diff_coeffs, DEVICE
            )
            print(f'  Trial {trial.number}, Epoch {epoch + 1} Test Acc: {current_test_accuracy:.2f}%')
            best_trial_accuracy = max(best_trial_accuracy, current_test_accuracy)

            # Optuna Pruning
            trial.report(current_test_accuracy, epoch)
            if trial.should_prune():
                print(f"--- Pruning Trial {trial.number} at epoch {epoch+1} ---")
                raise optuna.exceptions.TrialPruned()

    except Exception as e:
         print(f"!!! Error during training/evaluation in Trial {trial.number}, Epoch {epoch+1}: {e}")
         # Возвращаем текущую лучшую точность или 0, чтобы trial завершился
         return best_trial_accuracy 


    print(f"--- Trial {trial.number} Finished. Best Acc in {run_config['EPOCHS']} epochs: {best_trial_accuracy:.2f}% ---")
    return best_trial_accuracy

# --- Функции для визуализации и полного обучения ---
def write_plots(study):
    """Сохраняет графики Optuna, если возможно."""
    #You can visualize results if needed (requires pip install optuna-dashboard or plotly)
    try:
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_image("opt_history_attention.png") # Новое имя файла
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_image("param_importance_attention.png") # Новое имя файла
        print("Optuna plots saved to opt_history_attention.png and param_importance_attention.png")
    except ImportError:
        print("\nInstall plotly and kaleido to visualize Optuna results: pip install plotly kaleido")
    except ValueError as e_vis:
         if "Cannot evaluate parameter importances" in str(e_vis):
              print("Could not plot parameter importances (likely due to GridSampler or too few trials).")
         else:
              print(f"Could not plot Optuna results: {e_vis}")
    except Exception as e_vis:
        print(f"Could not plot Optuna results: {e_vis}")

def run_full_training(config: Dict[str, Any]):
    """Запускает полный цикл обучения с заданными параметрами, планировщиком и ранней остановкой."""
    DEVICE = config['DEVICE']
    print("--- Running Full Training with Attention Blocks ---")
    print(f"Using device: {DEVICE}")
    
    # --- Добавляем параметры Attention блока в конфиг для полного прогона ---
    config['patch_size'] = 4
    config['img_feature_dim'] = 128 # Можно увеличить для финального прогона (e.g., 192, 256)
    config['label_feature_dim'] = 128 # Должно совпадать с img_feature_dim
    config['nhead'] = 4             # Можно увеличить (e.g., 6, 8)
    config['num_encoder_layers'] = 2  # Можно увеличить (e.g., 2, 3, 4)
    config['dim_feedforward'] = 256   # Можно увеличить (e.g., 512)
    config['dropout'] = 0.1

    # Выводим финальную конфигурацию
    print("Full Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 30)

    # 1. Precompute diffusion coefficients
    diff_coeffs = precompute_diffusion_coefficients(config['T'], DEVICE, config['s_noise_schedule'])

    # 2. Initialize Embeddings
    label_embedding = initialize_embeddings(config['NUM_CLASSES'], config['EMBED_DIM'], DEVICE)

    # 3. Initialize Model с Attention блоком
    block_kwargs_full = {
         "target_embed_dim": config['EMBED_DIM'],
         "img_channels": config['IMG_CHANNELS'],
         "img_size": config['IMG_SIZE'],
         "patch_size": config['patch_size'],
         "img_feature_dim": config['img_feature_dim'],
         "label_feature_dim": config['label_feature_dim'],
         "nhead": config['nhead'],
         "num_encoder_layers": config['num_encoder_layers'],
         "dim_feedforward": config['dim_feedforward'],
         "dropout": config['dropout']
    }
    model = NoPropNet(
         num_blocks=config['T'],
         block_class=AttentionDenoisingBlock, # Используем новый блок
         block_kwargs=block_kwargs_full,
         classifier_input_dim=config['EMBED_DIM'], # Классификатор принимает z0
         num_classes=config['NUM_CLASSES']
    ).to(DEVICE)

    # 4. Get Dataloaders
    pin_memory = True if DEVICE == torch.device('cuda') else False
    trainloader, testloader = get_dataloaders(config['BATCH_SIZE'], config['data_root'], config['num_workers'], pin_memory)

    # 5. Initialize Training Components - ВКЛЮЧАЯ ПЛАНИРОВЩИК
    # Убедимся, что параметры планировщика добавлены в config перед вызовом
    if 'T_max_epochs' not in config or 'eta_min_lr' not in config:
         raise ValueError("Scheduler params 'T_max_epochs'/'eta_min_lr' missing for full run.")
    optimizers_blocks, optimizer_final, schedulers_blocks, scheduler_final, mse_loss, ce_loss = \
        initialize_training_components(model, label_embedding, config, use_scheduler=True) # use_scheduler=True

    # 6. Training Loop
    print(f"\n--- Starting Full Training for {config['EPOCHS']} epochs ---")
    total_start_time = time.time()
    best_test_accuracy = 0.0
    epochs_no_improve = 0

    train_history = {'epoch': [], 'train_denoise_loss': [], 'train_classify_loss': [], 'test_accuracy': [], 'lr': []}


    for epoch in range(config['EPOCHS']):
        print(f"\n--- Starting Epoch {epoch + 1}/{config['EPOCHS']} ---")
        epoch_start_time = time.time()

        # Запускаем обучение на одну эпоху
        # Модифицируем train_epoch чтобы возвращала средние лоссы
        # (Или можно оставить как есть, если логгирование внутри устраивает)
        train_epoch(epoch, model, label_embedding, trainloader, optimizers_blocks, optimizer_final,
                    mse_loss, ce_loss, diff_coeffs, config) 

        epoch_time = time.time() - epoch_start_time
        with torch.no_grad():
            embedding_norm = torch.norm(label_embedding.weight.data)
        print(f"--- Epoch {epoch + 1} finished. Training Time: {epoch_time:.2f}s ---")
        print(f"--- End of Epoch {epoch + 1}. Embedding Norm: {embedding_norm:.4f} ---")

        # Шаг планировщиков LR
        current_lr = optimizer_final.param_groups[0]['lr'] # Get LR before stepping scheduler
        print(f"--- End of Epoch {epoch + 1}. LR before step: {current_lr:.6f} ---")
        if scheduler_final: # Проверяем, что планировщик был создан
            for scheduler in schedulers_blocks:
                scheduler.step()
            scheduler_final.step()
            current_lr_after = optimizer_final.param_groups[0]['lr'] # Get LR after stepping scheduler
            print(f"--- End of Epoch {epoch + 1}. LR after step: {current_lr_after:.6f} ---")
        else:
            print("--- End of Epoch {epoch + 1}. LR Scheduler was not used. ---")


        # Оценка после каждой эпохи
        print(f"--- Running Evaluation for Epoch {epoch + 1} ---")
        eval_start_time = time.time()
        current_test_accuracy, correct, total = evaluate_model(
            model, testloader, config['T'], diff_coeffs, DEVICE
        )
        eval_time = time.time() - eval_start_time
        print(f'>>> Epoch {epoch + 1} Test Accuracy: {current_test_accuracy:.2f} % ({correct}/{total}) <<<')
        print(f"Evaluation Time: {eval_time:.2f}s")

        # Логгирование истории (средние лоссы нужно вынести из train_epoch если нужны точные)
        train_history['epoch'].append(epoch + 1)
        # train_history['train_denoise_loss'].append(avg_denoise_loss) # Нужно вернуть из train_epoch
        # train_history['train_classify_loss'].append(avg_classify_loss) # Нужно вернуть из train_epoch
        train_history['test_accuracy'].append(current_test_accuracy)
        train_history['lr'].append(current_lr) # Логгируем LR до шага планировщика


        # Логика Ранней Остановки
        if 'patience' in config: # Проверяем наличие ключа
            if current_test_accuracy > best_test_accuracy:
                best_test_accuracy = current_test_accuracy
                epochs_no_improve = 0
                print(f"*** New best test accuracy: {best_test_accuracy:.2f} % ***")
                # Можно сохранить лучшую модель здесь
                # torch.save(model.state_dict(), 'best_model_attention_run.pth')
            else:
                epochs_no_improve += 1
                print(f"Test accuracy did not improve for {epochs_no_improve} epochs.")

            if epochs_no_improve >= config['patience']:
                print(f"\nStopping early after {epoch + 1} epochs due to no improvement for {config['patience']} epochs.")
                print(f"Best test accuracy achieved: {best_test_accuracy:.2f} %")
                break # Выход из цикла по эпохам
        else:
             # Если patience не задан, просто обновляем лучшую точность
             if current_test_accuracy > best_test_accuracy:
                 best_test_accuracy = current_test_accuracy
                 print(f"*** New best test accuracy: {best_test_accuracy:.2f} % ***")

        # --- Конец цикла по эпохам ---

    # 7. Конец обучения
    print('\nFinished Full Training')
    total_training_time = time.time() - total_start_time
    print(f"Total Training Time: {total_training_time:.2f}s")
    print(f"Final Best Test Accuracy during run: {best_test_accuracy:.2f} %")

    # Можно построить график обучения, если история лоссов собиралась
    # import matplotlib.pyplot as plt
    # fig, ax1 = plt.subplots()
    # # ... код для отрисовки ...
    # plt.savefig("training_accuracy_attention_plot.png")


# --- Основной блок ---
if __name__ == "__main__":

    RUN_HPO = True # Установите True для запуска Optuna, False для полного обучения

    if RUN_HPO:
        print("--- Starting Hyperparameter Optimization using Optuna (Attention Blocks) ---")
        search_space = {
             'LR': LR_TRIALS,
             'ETA_LOSS_WEIGHT': ETA_LOSS_WEIGHT_TRIALS,
             'EMBED_WD': EMBED_WD_TRIALS
        }
        # Считаем количество комбинаций для GridSearch
        n_trials = 1
        for values in search_space.values():
             n_trials *= len(values)

        study = optuna.create_study(
            study_name=STUDY_NAME, # Новое имя
            direction='maximize',
            sampler=optuna.samplers.GridSampler(search_space), # Grid Search
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=4, interval_steps=1),
            storage="sqlite:///optuna_results_attention.db", # Новая база данных
            load_if_exists=True,
        )

        initial_trials = len(study.trials)
        remaining_trials = n_trials - initial_trials
        
        if remaining_trials <= 0:
             print(f"Grid search complete. Found {initial_trials} previous trials.")
        else:
             print(f"Found {initial_trials} previous trials. Starting {remaining_trials} new trials for Grid Search...")
             progress_bar = tqdm(total=remaining_trials) 
             start_hpo_time = time.time()
             try:
                 study.optimize(objective, n_trials=remaining_trials, timeout=60*60*24, callbacks=[tqdm_callback]) # Используем objective
             except KeyboardInterrupt:
                 print("\nПрерывание: сохраняю текущий прогресс Optuna...")
             except Exception as e:
                  print(f"An error occurred during HPO: {e}")
             finally:
                  end_hpo_time = time.time()
                  print(f"Total HPO time for this run: {end_hpo_time - start_hpo_time:.2f}s")
                  if progress_bar:
                      progress_bar.close()
                      progress_bar = None # Reset progress bar

        # Вывод результатов HPO
        print("\n--- HPO Finished ---")
        if study.best_trial:
            print(f"Best trial number: {study.best_trial.number}")
            print(f"Best accuracy (in {base_config['EPOCHS']} epochs): {study.best_value:.2f}%")
            print("Best hyperparameters found:")
            for key, value in study.best_params.items():
                print(f"  {key}: {value}")
            print("\n--- Recommendations ---")
            print(f"Use these 'Best hyperparameters' for a full training run...")
            # Сохраняем лучшие параметры для использования в full run
            best_params_from_hpo = study.best_params
        else:
            print("No successful HPO trials completed.")
            best_params_from_hpo = None # Не удалось найти лучшие параметры

        write_plots(study) # Функция для отрисовки графиков Optuna

    else: # Запуск полного обучения
        print("--- Starting Full Training Run with Attention Blocks ---")
        # Используем базовый конфиг и переопределяем параметры
        full_run_config = base_config.copy()

        # << ВАЖНО: Установите здесь либо лучшие параметры из HPO, либо выбранные вручную >>
        # Пример использования лучших из HPO (если HPO запускался ранее и нашел что-то):
        # if best_params_from_hpo: # Проверка, что HPO был успешен
        #     full_run_config['LR'] = best_params_from_hpo['LR']
        #     full_run_config['ETA_LOSS_WEIGHT'] = best_params_from_hpo['ETA_LOSS_WEIGHT']
        #     full_run_config['EMBED_WD'] = best_params_from_hpo['EMBED_WD']
        #     print("Using best parameters found from HPO.")
        # else: # Параметры по умолчанию или выбранные вручную, если HPO не было
        full_run_config['LR'] = 0.002 # Пример: Выбрано вручную или из предыдущего опыта
        full_run_config['ETA_LOSS_WEIGHT'] = 0.01   # Пример
        full_run_config['EMBED_WD'] = 1e-08 # Пример
        print("Using manually selected/default parameters for full run.")

        # Устанавливаем параметры для полного прогона (эпохи, планировщик, ранняя остановка)
        full_run_config['EPOCHS'] = 100
        full_run_config['T_max_epochs'] = 100 # Для планировщика LR (должно совпадать с EPOCHS для полного цикла)
        full_run_config['eta_min_lr'] = 1e-6  # Для планировщика LR
        full_run_config['patience'] = 15      # Для ранней остановки

        # Запускаем полный прогон
        run_full_training(full_run_config)
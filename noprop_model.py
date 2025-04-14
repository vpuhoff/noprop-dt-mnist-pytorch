# noprop_model.py
"""
Модуль, определяющий архитектуру NoPropNet, ее компоненты
и связанные с диффузией вспомогательные функции и логику инференса.
"""

import torch
import torch.nn as nn
import math
from typing import Dict

# --- Вспомогательные функции для диффузии ---

def get_alpha_bar_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Генерирует расписание cumulative product of alphas (alphas_bar) по косинусоиде.

    Args:
        timesteps (int): Общее количество временных шагов T.
        s (float): Небольшое смещение для предотвращения слишком близких к 1 значений в начале.

    Returns:
        torch.Tensor: Тензор alphas_bar формы (timesteps + 1).
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0] # Нормализуем, чтобы alpha_bar_0 = 1
    # Ограничиваем значения для численной стабильности
    return torch.clip(alphas_cumprod, 0.0001, 0.9999)

def precompute_diffusion_coefficients(T: int, device: torch.device, s: float = 0.008) -> Dict[str, torch.Tensor]:
    """
    Предварительно вычисляет коэффициенты, необходимые для диффузионного процесса.

    Args:
        T (int): Общее количество временных шагов.
        device (torch.device): Устройство ('cuda' или 'cpu').
        s (float): Параметр смещения для get_alpha_bar_schedule.

    Returns:
        Dict[str, torch.Tensor]: Словарь с предварительно вычисленными тензорами:
            'alphas_bar', 'alphas', 'betas', 'sqrt_alphas_bar',
            'sqrt_one_minus_alphas_bar', 'sqrt_recip_alphas', 'posterior_variance'.
    """
    alphas_bar = get_alpha_bar_schedule(T, s).to(device)
    # alphas_bar имеет T+1 элементов (от 0 до T)
    # alphas и betas имеют T элементов (от 1 до T)
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    betas = 1.0 - alphas
    sqrt_alphas_bar = torch.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)

    # Для обратного процесса:
    # reciprocal of sqrt(alpha_t), используется для вычисления среднего значения mu_t(z_t, x)
    # Убедимся, что alpha_t не равно 1 для избежания деления на ноль в 1/sqrt(alpha_t)
    alphas_clamped = torch.clamp(alphas, max=1.0 - 1e-9)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alphas_clamped)

    # posterior variance: beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    # Размерность T (от t=1 до T)
    # Убедимся, что знаменатель не равен нулю
    posterior_variance = betas * (1.0 - alphas_bar[:-1]) / torch.clamp(1.0 - alphas_bar[1:], min=1e-6)

    print("Noise schedule and necessary coefficients precomputed.")
    return {
        'alphas_bar': alphas_bar, # Форма (T+1)
        'alphas': alphas,         # Форма (T)
        'betas': betas,           # Форма (T)
        'sqrt_alphas_bar': sqrt_alphas_bar, # Форма (T+1)
        'sqrt_one_minus_alphas_bar': sqrt_one_minus_alphas_bar, # Форма (T+1)
        'sqrt_recip_alphas': sqrt_recip_alphas, # Форма (T)
        'posterior_variance': posterior_variance # Форма (T)
    }

# --- Компоненты Модели ---

class DenoisingBlockPaper(nn.Module):
    """
    Блок, предсказывающий шум (эпсилон) на основе входного изображения (x)
    и зашумленного представления метки (z_input) на определенном временном шаге.
    Архитектура следует описанию в статье NoProp.
    """
    def __init__(self, embed_dim: int, img_channels: int = 1, img_size: int = 28):
        super().__init__()
        self.embed_dim = embed_dim
        # Сверточная часть для обработки изображения
        self.img_conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)), # Приводим к фиксированному размеру 3x3
            nn.Flatten() # 128 * 3 * 3 = 1152
        )
        # Динамическое определение размера выхода сверточной части
        with torch.no_grad():
            dummy_input = torch.zeros(1, img_channels, img_size, img_size)
            conv_output_size = self.img_conv(dummy_input).shape[-1]

        # Полносвязная часть для признаков изображения
        self.img_fc = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2)
        )
        # Сеть для обработки входного эмбеддинга z_input
        self.label_embed_proc = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2)
        )
        # Комбинирующая сеть для предсказания эпсилон
        self.combined = nn.Sequential(
            nn.Linear(256 + 256, 256), # Вход: конкатенация признаков img и label
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, self.embed_dim) # Выход: предсказанный эпсилон (размер эмбеддинга)
        )

    def forward(self, x: torch.Tensor, z_input: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход блока.

        Args:
            x (torch.Tensor): Батч изображений (B, C, H, W).
            z_input (torch.Tensor): Батч зашумленных эмбеддингов (B, embed_dim).

        Returns:
            torch.Tensor: Предсказанный шум эпсилон (B, embed_dim).
        """
        img_features = self.img_fc(self.img_conv(x))
        label_features = self.label_embed_proc(z_input)
        combined_features = torch.cat([img_features, label_features], dim=1)
        predicted_epsilon = self.combined(combined_features)
        return predicted_epsilon

# --- Основная Модель ---

class NoPropNet(nn.Module):
    """
    Основная модель NoPropNet, состоящая из T блоков DenoisingBlockPaper
    и финального классификатора.
    """
    def __init__(self, num_blocks: int, embed_dim: int, num_classes: int, img_channels: int = 1, img_size: int = 28):
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("Number of blocks (T) must be positive.")
        self.num_blocks = num_blocks # Равно T
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Создаем T независимых блоков
        self.blocks = nn.ModuleList([
            DenoisingBlockPaper(embed_dim, img_channels, img_size)
            for _ in range(num_blocks)
        ])

        # Финальный классификатор, применяемый к предсказанному z_0
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, z_0: torch.Tensor) -> None:
        """
        Этот метод forward() намеренно оставлен пустым или с предупреждением,
        так как прямой проход всей модели не используется напрямую.
        Вместо этого используется `run_inference` для обратного процесса
        и специфическая логика в `train_epoch` для обучения.
        """
        print("Warning: NoPropNet.forward is conceptual only. Use run_inference for evaluation or the specific training loop.")
        return None

# --- Функция Инференса (Обратный процесс) ---

@torch.no_grad()
def run_inference(
    model: NoPropNet, x_batch: torch.Tensor, T_steps: int, diff_coeffs: Dict[str, torch.Tensor], device: torch.device
) -> torch.Tensor:
    """
    Выполняет обратный диффузионный процесс для получения предсказаний (логитов).

    Args:
        model (NoPropNet): Обученная модель.
        x_batch (torch.Tensor): Батч входных изображений.
        T_steps (int): Количество шагов диффузии (должно совпадать с model.num_blocks).
        diff_coeffs (Dict[str, torch.Tensor]): Предвычисленные коэффициенты диффузии.
        device (torch.device): Устройство для вычислений.

    Returns:
        torch.Tensor: Логиты классификации для батча.
    """
    model.eval() # Убедимся, что модель в режиме оценки
    batch_size = x_batch.shape[0]
    embed_dim = model.embed_dim

    if T_steps != model.num_blocks:
         print(f"Warning: T_steps ({T_steps}) for inference differs from model.num_blocks ({model.num_blocks}). Using {model.num_blocks}.")
         T_steps = model.num_blocks # Используем количество блоков в модели

    # Получаем нужные коэффициенты
    alphas = diff_coeffs['alphas']                             # (T)
    alphas_bar = diff_coeffs['alphas_bar']                     # (T+1)
    posterior_variance = diff_coeffs['posterior_variance']     # (T)
    sqrt_recip_alphas = diff_coeffs['sqrt_recip_alphas']       # (T)
    sqrt_one_minus_alphas_bar = diff_coeffs['sqrt_one_minus_alphas_bar'] # (T+1)

    # 1. Начинаем с шума z_T ~ N(0, I)
    z = torch.randn(batch_size, embed_dim, device=device)

    # 2. Итеративно применяем блоки в обратном порядке (от T до 1)
    for t_idx_rev in range(T_steps - 1, -1, -1): # t_idx_rev = T-1, T-2, ..., 0
        block_to_use = model.blocks[t_idx_rev]

        # Предсказываем эпсилон с помощью соответствующего блока
        # Вход для блока t: изображение x и текущий z (который является z_{t+1})
        predicted_epsilon = block_to_use(x_batch, z)

        # Получаем коэффициенты для текущего шага t = t_idx_rev + 1
        # Индексы в тензорах: alphas[t-1], alphas_bar[t], betas[t-1], etc.
        # соответствуют t_idx_rev
        alpha_t = alphas[t_idx_rev]
        beta_t = 1.0 - alpha_t
        sqrt_recip_alpha_t = sqrt_recip_alphas[t_idx_rev]
        # Используем alpha_bar_t (соответствует t=t_idx_rev+1)
        sqrt_1m_alpha_bar_t = sqrt_one_minus_alphas_bar[t_idx_rev + 1]

        # Вычисляем среднее значение для z_{t-1} по формуле обратного процесса
        # mean = 1/sqrt(alpha_t) * (z_t - (beta_t / sqrt(1 - alpha_bar_t)) * predicted_epsilon)
        mean = sqrt_recip_alpha_t * (z - beta_t / (sqrt_1m_alpha_bar_t + 1e-9) * predicted_epsilon)

        # Добавляем шум, если t > 0
        if t_idx_rev == 0:
            # На последнем шаге (t=1 -> t=0), дисперсия равна 0
            noise = torch.zeros_like(z)
            variance_stable = torch.tensor(0.0, device=device)
        else:
            # Используем posterior_variance для t
            variance = posterior_variance[t_idx_rev]
            variance_stable = torch.clamp(variance, min=0.0) # Гарантируем неотрицательность
            noise = torch.randn_like(z)

        # Сэмплируем z_{t-1}
        z = mean + torch.sqrt(variance_stable + 1e-6) * noise # Добавляем малое число для стабильности корня

    # 3. После цикла z содержит предсказанный z_0 (u_hat_0)
    # Применяем классификатор к z_0
    logits = model.classifier(z)
    return logits
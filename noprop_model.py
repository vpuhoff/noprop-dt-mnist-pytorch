"""
Основной скрипт для обучения, оценки и подбора гиперпараметров
модели NoPropNet, определенной в noprop_model.py.
"""

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
# import math # Math теперь не нужен здесь напрямую
import time
from typing import Dict, Tuple, List, Any
from optuna.samplers import GridSampler
import plotly
import kaleido # Для сохранения графиков Optuna
from tqdm import tqdm
import matplotlib.pyplot as plt

# Импортируем компоненты из нашего модуля модели
from noprop_model import (
    NoPropNet,
    precompute_diffusion_coefficients,
    run_inference
)

# --- 1. Базовая Конфигурация ---
config: Dict[str, Any] = {
    # Diffusion settings
    "T": 10, # Количество блоков/шагов диффузии
    "s_noise_schedule": 0.008, # Параметр для косинусного расписания шума
    # Model architecture
    "EMBED_DIM": 20, # Размерность эмбеддингов меток/z
    "NUM_CLASSES": 10, # Количество классов (MNIST)
    "IMG_SIZE": 28, # Размер изображения (MNIST)
    "IMG_CHANNELS": 1, # Количество каналов изображения (MNIST)
    # Training settings
    "BATCH_SIZE": 128,
    "EPOCHS": 10, # Эпохи для HPO или коротких тестов
    "LR": 1e-3,
    "WEIGHT_DECAY": 1e-3, # WD для параметров блоков и классификатора
    "EMBED_WD": 1e-5,     # WD для эмбеддингов меток
    "MAX_NORM_EMBED": 50.0, # Максимальная норма для эмбеддингов меток
    "GRAD_CLIP_MAX_NORM": 1.0, # Максимальная норма градиента
    "ETA_LOSS_WEIGHT": 0.5,   # Вес для Denoise Loss (MSE эпсилон) в гибридной потере
    "LAMBDA_GLOBAL": 1.0,   # Вес для Global Target Loss (MSE u_y) в гибридной потере
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # Scheduler settings (для полного прогона)
    "T_max_epochs": 100, # Общее кол-во эпох для косинусного планировщика LR
    "eta_min_lr": 1e-6,  # Минимальный LR для планировщика
    # Early stopping (для полного прогона)
    "patience": 15,
    # Data settings
    "data_root": './data',
    "num_workers": 2,
}

# --- Константы для HPO ---
STUDY_NAME = "find_params_v7_refactored" # Новое имя для рефакторинга
LR_TRIALS = [0.1, 1e-2, 1e-3, 1e-4]
ETA_LOSS_WEIGHT_TRIALS = [0.5, 1.0, 1.5, 2.0] # Вес для MSE(eps)
LAMBDA_GLOBAL_TRIALS = [0.5, 1.0, 1.5, 2.0] # Вес для MSE(u_y) - НОВЫЙ ПАРАМЕТР HPO
EMBED_WD_TRIALS = [1e-5, 1e-6, 1e-7]

# --- 2. Вспомогательные функции (Optuna, Data, Init) ---

progress_bar = None # Глобальный для HPO callback

def tqdm_callback(study, trial):
    """Callback для обновления progress bar Optuna."""
    if progress_bar:
        progress_bar.update(1)

def initialize_embeddings(num_classes: int, embed_dim: int, device: torch.device) -> nn.Embedding:
    """Инициализирует эмбеддинги меток."""
    label_embedding = nn.Embedding(num_classes, embed_dim).to(device)
    print(f"Initializing {embed_dim}-dim label embeddings.")
    # Попытка ортогональной инициализации, если возможно
    try:
        if embed_dim >= num_classes:
            nn.init.orthogonal_(label_embedding.weight)
            print("Applied orthogonal initialization to label embeddings.")
        else:
            print("Default initialization for label embeddings (orthogonal not possible).")
    except ValueError as e:
        print(f"Warning: Orthogonal init for embeddings failed ({e}). Using default init.")
    return label_embedding

def get_dataloaders(batch_size: int, root: str, num_workers: int, pin_memory: bool) -> Tuple[DataLoader, DataLoader]:
    """Загружает и подготавливает датасет MNIST."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Стандартная нормализация для MNIST
    ])
    trainset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    testset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    print(f"Loaded MNIST: {len(trainset)} train, {len(testset)} test samples.")
    return trainloader, testloader

def initialize_training_components(
    model: NoPropNet, label_embedding: nn.Embedding, config: Dict[str, Any], use_scheduler: bool = True
) -> Tuple[List[optim.Optimizer], optim.Optimizer, List[Any], Any, nn.Module, nn.Module]:
    """Инициализирует оптимизаторы, планировщики (опционально) и функции потерь."""
    LR = config['LR']
    WEIGHT_DECAY = config['WEIGHT_DECAY'] # WD для блоков и классификатора
    EMBED_WD = config['EMBED_WD']         # WD для эмбеддингов

    # Оптимизатор для каждого блока DenoisingBlockPaper
    optimizers_blocks = [
        optim.AdamW(block.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        for block in model.blocks
    ]
    # Отдельный оптимизатор для классификатора и эмбеддингов меток
    params_final_classifier = model.classifier.parameters()
    params_final_embedding = label_embedding.parameters()
    optimizer_final = optim.AdamW([
        {'params': params_final_classifier, 'weight_decay': WEIGHT_DECAY},
        {'params': params_final_embedding, 'weight_decay': EMBED_WD} # Используем отдельный WD для эмбеддингов
    ], lr=LR)
    print(f"Initialized optimizers: LR={LR}, Blocks WD={WEIGHT_DECAY}, Classifier WD={WEIGHT_DECAY}, Embeddings WD={EMBED_WD}.")

    schedulers_blocks = []
    scheduler_final = None
    if use_scheduler:
        if 'T_max_epochs' not in config or 'eta_min_lr' not in config:
            raise ValueError("Scheduler parameters 'T_max_epochs' and 'eta_min_lr' missing from config.")
        T_max_epochs = config['T_max_epochs']
        eta_min_lr = config['eta_min_lr']
        # Планировщик для каждого блока
        schedulers_blocks = [
            optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max_epochs, eta_min=eta_min_lr)
            for opt in optimizers_blocks
        ]
        # Планировщик для классификатора и эмбеддингов
        scheduler_final = optim.lr_scheduler.CosineAnnealingLR(optimizer_final, T_max=T_max_epochs, eta_min=eta_min_lr)
        print(f"Initialized CosineAnnealingLR schedulers with T_max={T_max_epochs}, eta_min={eta_min_lr}.")
    else:
        print("LR Scheduler is DISABLED for this run (e.g., during HPO).")

    # Функции потерь
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    return optimizers_blocks, optimizer_final, schedulers_blocks, scheduler_final, mse_loss, ce_loss

# --- 3. Логика Обучения и Оценки ---

# МОДИФИЦИРОВАННАЯ ФУНКЦИЯ TRAIN_EPOCH (логика осталась прежней)
def train_epoch(
    epoch_num: int, # Только для логирования
    model: NoPropNet,
    label_embedding: nn.Embedding,
    trainloader: DataLoader,
    optimizers_blocks: List[optim.Optimizer],
    optimizer_final: optim.Optimizer,
    mse_loss: nn.Module,
    ce_loss: nn.Module,
    diff_coeffs: Dict[str, torch.Tensor],
    config: Dict[str, Any]
) -> Tuple[float, float]: # Возвращает средние потери за эпоху
    """Выполняет одну эпоху обучения с использованием гибридной функции потерь."""
    model.train() # Переводим модель в режим обучения

    # Параметры из конфигурации
    T = config['T']
    DEVICE = config['DEVICE']
    # Получаем веса для компонентов потерь (с дефолтами на всякий случай)
    ETA_LOSS_WEIGHT = config.get('ETA_LOSS_WEIGHT', 1.0) # Вес для локальной MSE(eps)
    LAMBDA_GLOBAL = config.get('LAMBDA_GLOBAL', 1.0) # Вес для глобальной цели MSE(u_y)
    GRAD_CLIP_MAX_NORM = config['GRAD_CLIP_MAX_NORM']
    MAX_NORM_EMBED = config['MAX_NORM_EMBED']

    # Доступ к предвычисленным коэффициентам
    alphas_bar = diff_coeffs['alphas_bar']
    sqrt_alphas_bar = diff_coeffs['sqrt_alphas_bar']
    sqrt_one_minus_alphas_bar = diff_coeffs['sqrt_one_minus_alphas_bar']

    # Аккумуляторы для средних потерь за эпоху
    epoch_total_denoise_loss = 0.0        # MSE(eps) loss
    epoch_total_global_target_loss = 0.0  # MSE(u_y) loss
    epoch_total_classify_loss = 0.0       # CE loss
    epoch_total_batches = 0               # Общее количество батчей (для усреднения)

    print(f"Starting Epoch {epoch_num + 1} Training Phase...")
    # --- Внешний цикл по временным шагам t (обучение каждого блока) ---
    for t_idx in range(T): # t_idx = 0..T-1 соответствует шагам t=1..T в статье
        block_to_train = model.blocks[t_idx]
        optimizer_t = optimizers_blocks[t_idx]

        # Progress bar для батчей внутри прохода обучения блока t
        pbar_desc = f"Epoch {epoch_num+1}, Block {t_idx+1}/{T}"
        pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=pbar_desc, leave=False)

        # --- Внутренний цикл по мини-батчам ---
        for i, data in pbar:
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            batch_size = inputs.shape[0]

            # --- Общая подготовка ---
            # Получаем "чистые" эмбеддинги u_y = z_0. Нужен grad для optimizer_final.
            u_y = label_embedding(labels)

            # Получаем значения расписания шума для z_{t-1} (индекс t_idx в alphas_bar)
            # Коэффициенты для сэмплирования z_{t-1} из u_y
            sqrt_a_bar = sqrt_alphas_bar[t_idx].view(-1, 1).expand_as(u_y)
            sqrt_1_minus_a_bar = sqrt_one_minus_alphas_bar[t_idx].view(-1, 1).expand_as(u_y)

            # --- 1. Локальная потеря Denoising (Предсказание Эпсилон) ---
            # Генерируем целевой шум epsilon_target ~ N(0, I)
            epsilon_target = torch.randn_like(u_y)
            # Создаем зашумленный вход z_{t-1} для блока t, используя ОТСОЕДИНЕННЫЙ u_y
            # z_{t-1} = sqrt(alpha_bar_{t-1}) * u_y + sqrt(1 - alpha_bar_{t-1}) * epsilon
            z_input_for_block = sqrt_a_bar * u_y.detach() + sqrt_1_minus_a_bar * epsilon_target
            # Предсказываем эпсилон с помощью текущего блока
            predicted_epsilon = block_to_train(inputs, z_input_for_block)
            # Считаем MSE между предсказанным и целевым шумом
            loss_denoise = mse_loss(predicted_epsilon, epsilon_target)
            epoch_total_denoise_loss += loss_denoise.item() # Аккумулируем не взвешенную потерю

            # --- 2. Глобальная целевая потеря (Неявное предсказание u_y) ---
            # Восстанавливаем предсказанный "чистый" эмбеддинг u_hat_t из predicted_epsilon
            # Формула: u_hat_t = (z_{t-1} - sqrt(1-alpha_bar_{t-1}) * predicted_epsilon) / sqrt(alpha_bar_{t-1})
            # predicted_u_t требует градиента от predicted_epsilon
            predicted_u_t = (z_input_for_block - sqrt_1_minus_a_bar * predicted_epsilon) / (sqrt_a_bar + 1e-9)
            # Считаем MSE между предсказанным u_hat_t и ОТСОЕДИНЕННЫМ чистым u_y
            loss_global_target = mse_loss(predicted_u_t, u_y.detach())
            epoch_total_global_target_loss += loss_global_target.item() # Аккумулируем не взвешенную потерю

            # --- 3. Взвешенная комбинированная потеря для обновления БЛОКА t ---
            # Эта потеря используется для вычисления градиентов ТОЛЬКО для блока t
            weighted_loss_block_t = ETA_LOSS_WEIGHT * loss_denoise + LAMBDA_GLOBAL * loss_global_target

            # --- 4. Потеря классификации (используя финальный предсказанный эмбеддинг u_hat_T) ---
            # Пересчитывается на каждой итерации батча для обновления классификатора/эмбеддингов
            # Используем .no_grad(), так как этот путь не должен влиять на градиенты блоков (кроме последнего)
            with torch.no_grad():
                # Сэмплируем z_{T-1} как вход для последнего блока T
                # Используем коэффициенты для шага T-1 (индекс T-1 в alphas_bar)
                sqrt_a_bar_Tm1 = sqrt_alphas_bar[T-1].view(-1, 1).expand_as(u_y)
                sqrt_1_minus_a_bar_Tm1 = sqrt_one_minus_alphas_bar[T-1].view(-1, 1).expand_as(u_y)
                # Генерируем новый шум для сэмплирования z_{T-1}
                epsilon_Tm1_target_for_clf = torch.randn_like(u_y)
                # Сэмплируем z_{T-1} из ОТСОЕДИНЕННОГО u_y
                z_T_minus_1_sample = sqrt_a_bar_Tm1 * u_y.detach() + sqrt_1_minus_a_bar_Tm1 * epsilon_Tm1_target_for_clf

                # Получаем предсказанный шум epsilon_T от ПОСЛЕДНЕГО блока (индекс T-1)
                # Передаем ОТСОЕДИНЕННЫЙ z_{T-1}
                predicted_epsilon_T = model.blocks[T-1](inputs, z_T_minus_1_sample.detach())

                # Восстанавливаем предсказанный финальный "чистый" эмбеддинг u_hat_T (предсказанный z_0)
                predicted_u_y_final = (z_T_minus_1_sample - sqrt_1_minus_a_bar_Tm1 * predicted_epsilon_T) / (sqrt_a_bar_Tm1 + 1e-9)
                # Отсоединяем градиенты от u_hat_T перед классификатором, так как CE loss должна влиять только на классификатор и эмбеддинги u_y
                predicted_u_y_final_detached = predicted_u_y_final.detach()

            # Теперь вычисляем потерю классификации с градиентами
            # Вход для классификатора - это u_y (эмбеддинг из Embedding слоя)
            # Но мы хотим, чтобы классификатор учился на основе предсказаний модели.
            # Используем предсказанный u_y_final, но позволяем градиенту течь обратно к u_y
            # Пересмотр: Мы должны использовать предсказанный эмбеддинг для классификации,
            # но градиент от CE Loss должен идти к классификатору и ИСХОДНОМУ эмбеддингу u_y.
            # Поэтому используем u_y для классификатора в forward pass, но loss считается по предсказанию? Нет.
            # Правильный подход: используем predicted_u_y_final как вход для классификатора,
            # градиент от CE loss пойдет на параметры классификатора и на predicted_u_y_final.
            # Но predicted_u_y_final зависит от epsilon_T, который зависит от блока T.
            # Чтобы градиент CE влиял только на классификатор и ЭМБЕДДИНГИ (u_y),
            # нужно либо передавать u_y в классификатор, либо detach predicted_u_y_final.

            # --- Пересмотр логики CE Loss ---
            # Идея: CE Loss должна обновлять классификатор И эмбеддинги u_y, чтобы они становились лучше.
            # Градиент от CE Loss НЕ должен идти обратно в блоки предсказания шума.

            # Получаем логиты от классификатора, примененного к НАСТОЯЩИМ эмбеддингам u_y
            # Это гарантирует, что градиент CE пойдет только на классификатор и Embedding слой.
            logits = model.classifier(u_y) # Используем u_y, чтобы градиент шел к Embedding
            classify_loss = ce_loss(logits, labels)
            epoch_total_classify_loss += classify_loss.item() # Аккумулируем CE потерю

            # --- 5. Комбинированная общая потеря и обратное распространение ---
            # Потеря для блока t + Потеря для классификатора/эмбеддингов
            total_loss_batch = weighted_loss_block_t + classify_loss

            # Обнуляем градиенты для ОБОИХ оптимизаторов
            optimizer_t.zero_grad()     # Градиенты для блока t
            optimizer_final.zero_grad() # Градиенты для классификатора и эмбеддингов

            # Вычисляем градиенты от общей потери
            total_loss_batch.backward()
            # Градиенты теперь рассчитаны:
            # - weighted_loss_block_t повлияла только на параметры block_to_train
            # - classify_loss повлияла только на параметры model.classifier и label_embedding

            # --- 6. Клиппинг градиентов ---
            torch.nn.utils.clip_grad_norm_(block_to_train.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            # Клиппинг для параметров, обновляемых optimizer_final
            # Проверяем наличие градиентов перед клиппингом
            params_to_clip_final = []
            params_to_clip_final.extend(model.classifier.parameters())
            params_to_clip_final.extend(label_embedding.parameters())
            # Фильтруем параметры без градиентов (хотя они должны быть)
            params_with_grad_final = [p for p in params_to_clip_final if p.grad is not None]
            if params_with_grad_final:
                 torch.nn.utils.clip_grad_norm_(params_with_grad_final, max_norm=GRAD_CLIP_MAX_NORM)

            # --- 7. Шаг оптимизаторов ---
            optimizer_t.step()     # Обновляем параметры блока t
            optimizer_final.step() # Обновляем параметры классификатора и эмбеддингов

            # --- 8. Клиппинг нормы эмбеддингов ---
            with torch.no_grad():
                current_norm = label_embedding.weight.data.norm()
                if current_norm > MAX_NORM_EMBED:
                    label_embedding.weight.data.mul_(MAX_NORM_EMBED / (current_norm + 1e-9))

            epoch_total_batches += 1 # Считаем общее количество обновлений за эпоху

            # --- Обновление progress bar (опционально) ---
            if i % 100 == 0:
                pbar.set_description(
                    f"Epoch {epoch_num+1}, Block {t_idx+1}/{T} | "
                    f"Denoise L: {loss_denoise.item():.4f} | "
                    f"GlobalTgt L: {loss_global_target.item():.4f} | "
                    f"Classify L: {classify_loss.item():.4f}"
                )
            # --- Конец цикла по батчам ---
        pbar.close()
        # --- Конец цикла по блокам (t_idx) ---

    # Вычисляем средние потери за эпоху
    avg_epoch_denoise = epoch_total_denoise_loss / epoch_total_batches if epoch_total_batches > 0 else 0
    #avg_epoch_global_target = epoch_total_global_target_loss / epoch_total_batches if epoch_total_batches > 0 else 0 # Можно добавить, если нужно
    avg_epoch_classify = epoch_total_classify_loss / epoch_total_batches if epoch_total_batches > 0 else 0

    # Возвращаем средние потери для логирования вне функции
    return avg_epoch_denoise, avg_epoch_classify


@torch.no_grad()
def evaluate_model(
    model: NoPropNet, testloader: DataLoader, T_steps: int, diff_coeffs: Dict[str, torch.Tensor], device: torch.device
) -> Tuple[float, int, int]:
    """
    Оценивает модель на тестовом наборе данных, используя run_inference.

    Args:
        model (NoPropNet): Модель для оценки.
        testloader (DataLoader): Загрузчик тестовых данных.
        T_steps (int): Количество шагов диффузии для инференса.
        diff_coeffs (Dict[str, torch.Tensor]): Предвычисленные коэффициенты.
        device (torch.device): Устройство для вычислений.

    Returns:
        Tuple[float, int, int]: Точность (в %), количество правильных предсказаний, общее количество примеров.
    """
    model.eval() # Переводим модель в режим оценки
    correct = 0
    total = 0
    print("Running evaluation...")
    for data in tqdm(testloader, desc="Evaluating", leave=False):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        # Используем импортированную функцию run_inference
        logits = run_inference(model, images, T_steps, diff_coeffs, device)

        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Evaluation finished. Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy, correct, total

# --- 4. Подбор Гиперпараметров (Optuna) ---

def objective(trial: optuna.trial.Trial) -> float:
    """Целевая функция Optuna для одного запуска (trial)."""
    # Создаем копию базовой конфигурации для этого запуска
    run_config = config.copy()

    # --- Гиперпараметры для подбора ---
    run_config['LR'] = trial.suggest_categorical('LR', LR_TRIALS)
    run_config['ETA_LOSS_WEIGHT'] = trial.suggest_categorical('ETA_LOSS_WEIGHT', ETA_LOSS_WEIGHT_TRIALS)
    run_config['LAMBDA_GLOBAL'] = trial.suggest_categorical('LAMBDA_GLOBAL', LAMBDA_GLOBAL_TRIALS)
    run_config['EMBED_WD'] = trial.suggest_categorical('EMBED_WD', EMBED_WD_TRIALS)
    # --- Фиксированные параметры для HPO ---
    run_config['EPOCHS'] = 10 # Короткий прогон для HPO

    print(f"\n--- Starting Optuna Trial {trial.number} with config: ---")
    print(f"  LR: {run_config['LR']:.1e}, ETA_L: {run_config['ETA_LOSS_WEIGHT']:.1f}, LAMBDA_G: {run_config['LAMBDA_GLOBAL']:.1f}, EmbedWD: {run_config['EMBED_WD']:.1e}")

    # --- Настройка, специфичная для этого trial ---
    DEVICE = run_config['DEVICE']
    # Пересчитываем коэффициенты для каждого trial (хотя они зависят только от T и s)
    diff_coeffs = precompute_diffusion_coefficients(run_config['T'], DEVICE, run_config['s_noise_schedule'])
    label_embedding = initialize_embeddings(run_config['NUM_CLASSES'], run_config['EMBED_DIM'], DEVICE)
    # Создаем модель
    model = NoPropNet(
        num_blocks=run_config['T'], embed_dim=run_config['EMBED_DIM'], num_classes=run_config['NUM_CLASSES'],
        img_channels=run_config['IMG_CHANNELS'], img_size=run_config['IMG_SIZE']
    ).to(DEVICE)
    # Загружаем данные
    trainloader, testloader = get_dataloaders(
        run_config['BATCH_SIZE'], run_config['data_root'], run_config['num_workers'], (DEVICE == torch.device('cuda'))
    )
    # Инициализируем компоненты обучения БЕЗ планировщика для HPO
    optimizers_blocks, optimizer_final, _, _, mse_loss, ce_loss = \
        initialize_training_components(model, label_embedding, run_config, use_scheduler=False)

    # --- Цикл обучения и оценки для этого trial ---
    best_trial_accuracy = 0.0
    for epoch in range(run_config['EPOCHS']):
        print(f"  Trial {trial.number}, Epoch {epoch + 1}/{run_config['EPOCHS']}")
        # Обучаем одну эпоху
        train_denoise_loss, train_classify_loss = train_epoch(
            epoch, model, label_embedding, trainloader, optimizers_blocks, optimizer_final,
            mse_loss, ce_loss, diff_coeffs, run_config
        )
        # Оцениваем модель
        current_test_accuracy, _, _ = evaluate_model(
            model, testloader, run_config['T'], diff_coeffs, DEVICE
        )
        print(f'  Trial {trial.number}, Epoch {epoch + 1} Test Acc: {current_test_accuracy:.2f}%')
        best_trial_accuracy = max(best_trial_accuracy, current_test_accuracy)

        # Отчет Optuna для возможного прунинга (ранней остановки неперспективных trial)
        trial.report(current_test_accuracy, epoch)
        if trial.should_prune():
            print(f"--- Pruning Trial {trial.number} at epoch {epoch+1} ---")
            raise optuna.exceptions.TrialPruned()

    print(f"--- Trial {trial.number} Finished. Best Acc in {run_config['EPOCHS']} epochs: {best_trial_accuracy:.2f}% ---")
    # Optuna ожидает, что функция вернет значение для максимизации/минимизации
    return best_trial_accuracy

def write_optuna_plots(study):
    """Сохраняет графики результатов Optuna."""
    try:
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_image("optuna_history.png")
        print("Saved Optuna optimization history plot to optuna_history.png")
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_image("optuna_param_importance.png")
        print("Saved Optuna parameter importance plot to optuna_param_importance.png")
    except ImportError:
        print("\nPlotly or Kaleido not found. Cannot save Optuna plots. Install: pip install plotly kaleido")
    except Exception as e:
        print(f"\nError generating Optuna plots: {e}")


# --- 5. Функция для Полного Обучения ---
def run_full_training(full_run_config: Dict[str, Any]):
    """Запускает полный цикл обучения с заданными параметрами, планировщиком и ранней остановкой."""
    DEVICE = full_run_config['DEVICE']
    print("\n" + "="*40)
    print("--- Running Full Training ---")
    print(f"Using device: {DEVICE}")
    print("Full Training Configuration:")
    for key, value in full_run_config.items():
        print(f"  {key}: {value}")
    print("="*40 + "\n")

    # 1. Предвычисление коэффициентов диффузии
    diff_coeffs = precompute_diffusion_coefficients(full_run_config['T'], DEVICE, full_run_config['s_noise_schedule'])
    # 2. Инициализация эмбеддингов
    label_embedding = initialize_embeddings(full_run_config['NUM_CLASSES'], full_run_config['EMBED_DIM'], DEVICE)
    # 3. Создание модели
    model = NoPropNet(
        num_blocks=full_run_config['T'], embed_dim=full_run_config['EMBED_DIM'], num_classes=full_run_config['NUM_CLASSES'],
        img_channels=full_run_config['IMG_CHANNELS'], img_size=full_run_config['IMG_SIZE']
    ).to(DEVICE)
    # 4. Загрузка данных
    pin_memory = (DEVICE == torch.device('cuda'))
    trainloader, testloader = get_dataloaders(full_run_config['BATCH_SIZE'], full_run_config['data_root'], full_run_config['num_workers'], pin_memory)
    # 5. Инициализация компонентов обучения (с планировщиком LR)
    optimizers_blocks, optimizer_final, schedulers_blocks, scheduler_final, mse_loss, ce_loss = \
        initialize_training_components(model, label_embedding, full_run_config, use_scheduler=True)

    # 6. Настройка цикла обучения и логирования
    print(f"\n--- Starting Full Training for {full_run_config['EPOCHS']} epochs ---")
    total_start_time = time.time()
    best_test_accuracy = 0.0
    epochs_no_improve = 0
    train_history = {'epoch': [], 'train_denoise_loss': [], 'train_classify_loss': [], 'test_accuracy': [], 'lr': []}

    # 7. Основной цикл обучения по эпохам
    for epoch in range(full_run_config['EPOCHS']):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch + 1}/{full_run_config['EPOCHS']} ---")

        # Запускаем обучение на одну эпоху
        avg_epoch_denoise, avg_epoch_classify = train_epoch(
            epoch, model, label_embedding, trainloader, optimizers_blocks, optimizer_final,
            mse_loss, ce_loss, diff_coeffs, full_run_config
        )

        epoch_time = time.time() - epoch_start_time
        with torch.no_grad():
            embedding_norm = torch.norm(label_embedding.weight.data)

        print(f">>> Epoch {epoch + 1} Training Finished. Time: {epoch_time:.2f}s")
        print(f"    Avg Train Losses: Denoise(MSE_eps)={avg_epoch_denoise:.7f}, Classify(CE)={avg_epoch_classify:.4f}")
        print(f"    End of Epoch Embedding Norm: {embedding_norm:.4f}")

        # Шаг планировщиков LR
        current_lr = optimizer_final.param_groups[0]['lr'] # Получаем LR до шага
        print(f"    LR before step: {current_lr:.6f}")
        if scheduler_final: # Проверяем, что планировщики были инициализированы
            for scheduler in schedulers_blocks: scheduler.step()
            scheduler_final.step()
            current_lr_after = optimizer_final.param_groups[0]['lr']
            print(f"    LR after step: {current_lr_after:.6f}")
        else:
             print("    LR Scheduler was not used.")
             current_lr_after = current_lr # Если планировщика нет, LR не меняется

        # Оценка модели на тестовом наборе
        eval_start_time = time.time()
        current_test_accuracy, correct, total = evaluate_model(
            model, testloader, full_run_config['T'], diff_coeffs, DEVICE
        )
        eval_time = time.time() - eval_start_time
        print(f">>> Epoch {epoch + 1} Evaluation Finished. Time: {eval_time:.2f}s")
        print(f'    Test Accuracy: {current_test_accuracy:.2f} % ({correct}/{total})')

        # Запись истории для графика
        train_history['epoch'].append(epoch + 1)
        train_history['train_denoise_loss'].append(avg_epoch_denoise)
        train_history['train_classify_loss'].append(avg_epoch_classify)
        train_history['test_accuracy'].append(current_test_accuracy)
        train_history['lr'].append(current_lr) # Логгируем LR *до* шага

        # Логика Ранней Остановки (Early Stopping)
        if 'patience' in full_run_config and full_run_config['patience'] > 0:
            if current_test_accuracy > best_test_accuracy:
                best_test_accuracy = current_test_accuracy
                epochs_no_improve = 0
                print(f"    *** New best test accuracy: {best_test_accuracy:.2f} % ***")
                # Раскомментируйте для сохранения лучшей модели:
                # try:
                #     torch.save(model.state_dict(), 'best_noprop_model.pth')
                #     print("    Saved best model state dict to best_noprop_model.pth")
                # except Exception as e:
                #     print(f"    Error saving model: {e}")
            else:
                epochs_no_improve += 1
                print(f"    Test accuracy did not improve for {epochs_no_improve} epochs.")

            if epochs_no_improve >= full_run_config['patience']:
                print(f"\nStopping early after {epoch + 1} epochs due to no improvement for {full_run_config['patience']} epochs.")
                print(f"Best test accuracy achieved: {best_test_accuracy:.2f} %")
                break # Выход из цикла обучения
        else: # Если ранняя остановка не используется, просто отслеживаем лучший результат
            if current_test_accuracy > best_test_accuracy:
                best_test_accuracy = current_test_accuracy
                print(f"    *** New best test accuracy: {best_test_accuracy:.2f} % ***")

    # --- Конец обучения ---
    print('\n' + "="*40)
    print('Finished Full Training')
    total_training_time = time.time() - total_start_time
    print(f"Total Training Time: {total_training_time:.2f}s")
    print(f"Final Best Test Accuracy during run: {best_test_accuracy:.2f} %")
    print("="*40)

    # --- Построение Графиков Обучения ---
    print("\nGenerating training plots...")
    try:
        epochs_ran = train_history['epoch']
        if epochs_ran: # Проверяем, были ли завершены эпохи
            fig, ax1 = plt.subplots(figsize=(12, 8))

            # График потерь (левая ось Y)
            color = 'tab:red'
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Avg Train Loss', color=color)
            # Логарифмическая шкала для потерь
            ax1.semilogy(epochs_ran, train_history['train_classify_loss'], color=color, linestyle='-', marker='x', markersize=4, label='Avg Classify Loss (CE)')
            ax1.semilogy(epochs_ran, train_history['train_denoise_loss'], color='tab:orange', linestyle=':', marker='+', markersize=4, label='Avg Denoise Loss (MSE eps)')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.legend(loc='upper left')
            ax1.grid(True, which='both', linestyle='--', linewidth=0.5) # Добавим сетку

            # График точности (правая ось Y)
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Test Accuracy (%)', color=color)
            ax2.plot(epochs_ran, train_history['test_accuracy'], color=color, linestyle='-', marker='.', label='Test Accuracy')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.legend(loc='upper right')
            # Установим пределы для точности для лучшей визуализации
            min_acc = min(train_history['test_accuracy']) if train_history['test_accuracy'] else 0
            max_acc = max(train_history['test_accuracy']) if train_history['test_accuracy'] else 100
            ax2.set_ylim(max(0, min_acc - 5), min(100, max_acc + 5))

            # График LR (третья ось Y, справа)
            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('outward', 60)) # Сдвигаем третью ось правее
            color = 'tab:green'
            ax3.set_ylabel('Learning Rate', color=color)
            ax3.plot(epochs_ran, train_history['lr'], color=color, linestyle='--', label='Learning Rate (Start of Epoch)')
            ax3.tick_params(axis='y', labelcolor=color)
            ax3.set_yscale('log') # Лог шкала для LR
            ax3.legend(loc='lower left')

            plt.title('NoProp Training Progress')
            fig.tight_layout() # Автоматически подгоняет элементы графика
            plt.savefig('training_progress_noprop.png')
            print("Saved training progress plot to training_progress_noprop.png")
            # plt.show() # Раскомментируйте, чтобы показать график
            plt.close(fig) # Закрываем фигуру после сохранения
        else:
             print("No epochs completed, skipping training plot generation.")
    except ImportError:
        print("\nMatplotlib not found. Install: pip install matplotlib")
    except Exception as e:
        print(f"\nError generating training plot: {e}")


# --- 6. Основной блок запуска ---
if __name__ == "__main__":

    RUN_HPO = False # Установите True для запуска Optuna HPO

    if RUN_HPO:
        print("--- Starting Hyperparameter Optimization using Optuna ---")
        # Определяем пространство поиска для GridSampler
        search_space = {
             'LR': LR_TRIALS,
             'ETA_LOSS_WEIGHT': ETA_LOSS_WEIGHT_TRIALS,
             'LAMBDA_GLOBAL': LAMBDA_GLOBAL_TRIALS, # Добавлен новый параметр
             'EMBED_WD': EMBED_WD_TRIALS
        }
        n_trials = 1
        for key in search_space:
             n_trials *= len(search_space[key])

        study = optuna.create_study(
            study_name=STUDY_NAME,
            direction='maximize', # Мы хотим максимизировать точность
            sampler=optuna.samplers.GridSampler(search_space), # Используем Grid Search
            # Используем прунер для ранней остановки плохих trial
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=4, interval_steps=1),
            storage="sqlite:///optuna_results.db", # Сохраняем результаты в БД
            load_if_exists=True, # Продолжаем предыдущее исследование, если оно есть
        )

        # Инициализируем progress bar для HPO
        # Учитываем уже завершенные trial, если загружаем исследование
        completed_trials = len(study.trials)
        progress_bar = tqdm(total=n_trials, initial=completed_trials, desc="Optuna Trials")

        print(f"\nStarting HPO Grid Search with {n_trials} potential trials ({config['EPOCHS']} epochs each)...")
        print(f"Study storage: sqlite:///optuna_results.db")
        print(f"Completed trials loaded: {completed_trials}")

        start_hpo_time = time.time()
        try:
            # Запускаем оптимизацию
            study.optimize(
                objective, # Наша целевая функция
                n_trials=n_trials - completed_trials, # Запускаем только недостающие trial
                timeout=60*60*24, # Максимальное время HPO (24 часа)
                callbacks=[tqdm_callback] # Callback для progress bar
            )
        except KeyboardInterrupt:
            print("\n--- HPO Interrupted by User ---")
            print("Saving current Optuna progress...")
        except Exception as e:
             print(f"\n--- An error occurred during HPO: {e} ---")
        finally:
            end_hpo_time = time.time()
            print(f"\nTotal HPO time for this session: {end_hpo_time - start_hpo_time:.2f}s")
            if progress_bar:
                progress_bar.close() # Закрываем progress bar

        # Вывод результатов HPO
        print("\n" + "="*40)
        print("--- HPO Finished ---")
        try:
             if study.best_trial:
                 print(f"Best trial number: {study.best_trial.number}")
                 print(f"Best accuracy (in {config['EPOCHS']} epochs): {study.best_value:.2f}%")
                 print("Best hyperparameters found:")
                 for key, value in study.best_params.items():
                      print(f"  {key}: {value}")

                 # Сохраняем лучшие параметры в файл для удобства
                 try:
                     with open("best_hpo_params.txt", "w") as f:
                         f.write(f"Best Trial: {study.best_trial.number}\n")
                         f.write(f"Best Value (Accuracy %): {study.best_value}\n")
                         f.write("Best Params:\n")
                         for key, value in study.best_params.items():
                             f.write(f"  '{key}': {value},\n")
                     print("Saved best hyperparameters to best_hpo_params.txt")
                 except Exception as e:
                     print(f"Error saving best params to file: {e}")

                 # Генерируем графики Optuna
                 write_optuna_plots(study)

                 print("\n--- Recommendation ---")
                 print("Update the 'full_run_config' in the 'else' block below")
                 print("with these 'Best hyperparameters' for a full training run.")

             else:
                 print("No successful HPO trials completed or study was empty.")
        except Exception as e:
             print(f"An error occurred while processing HPO results: {e}")
        print("="*40)

    else:
        # --- ЗАПУСК ПОЛНОГО ОБУЧЕНИЯ (без HPO) ---
        print("--- Skipping HPO, running Full Training with specified config ---")

        # Создаем конфигурацию для полного прогона
        # Можно использовать базовую 'config' или задать параметры вручную/из файла
        full_run_config = config.copy()

        # --- !!! ВАЖНО: Установите здесь лучшие параметры (найденные HPO или выбранные вручную) !!! ---
        # Пример использования лучших параметров из файла best_hpo_params.txt (если он есть)
        # Либо задайте вручную:
        best_params_for_full_run = {
            'LR': 1e-3,             # Пример значения
            'ETA_LOSS_WEIGHT': 1.0, # Пример значения
            'LAMBDA_GLOBAL': 1.0,   # Пример значения
            'EMBED_WD': 1e-6,       # Пример значения
        }
        full_run_config.update(best_params_for_full_run)

        # Параметры, специфичные для полного прогона (эпохи, планировщик, ранняя остановка)
        full_run_config['EPOCHS'] = 100 # Больше эпох для полного обучения
        full_run_config['T_max_epochs'] = 100 # Совпадает с EPOCHS для косинусного планировщика
        full_run_config['eta_min_lr'] = 1e-6  # Минимальный LR для планировщика
        full_run_config['patience'] = 15      # Количество эпох без улучшения для ранней остановки (0 для отключения)

        # Запускаем полный прогон с выбранной конфигурацией
        run_full_training(full_run_config)
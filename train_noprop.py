# train_noprop.py
"""
Основной скрипт для обучения, оценки и подбора гиперпараметров
модели NoPropNet (из noprop_model.py) на датасете Fashion-MNIST.
"""

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
# import math
import time
from typing import Dict, Tuple, List, Any
from optuna.samplers import GridSampler
import plotly
import kaleido
from tqdm import tqdm
import matplotlib.pyplot as plt

# Импортируем компоненты из нашего модуля модели
from noprop_model import (
    NoPropNet,
    precompute_diffusion_coefficients,
    run_inference
)

# --- 1. Базовая Конфигурация (остается в основном прежней) ---
config: Dict[str, Any] = {
    # Diffusion settings
    "T": 10,
    "s_noise_schedule": 0.008,
    # Model architecture (параметры подходят для Fashion-MNIST)
    "EMBED_DIM": 40,
    "NUM_CLASSES": 10,
    "IMG_SIZE": 28,
    "IMG_CHANNELS": 1,
    # Training settings (гиперпараметры могут потребовать тюнинга)
    "BATCH_SIZE": 128,
    "EPOCHS": 10, # Эпохи для HPO или коротких тестов
    "LR": 1e-3,
    "WEIGHT_DECAY": 1e-3,
    "EMBED_WD": 1e-5,
    "MAX_NORM_EMBED": 50.0,
    "GRAD_CLIP_MAX_NORM": 1.0,
    "ETA_LOSS_WEIGHT": 0.5,
    "LAMBDA_GLOBAL": 1.0,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # Scheduler settings (для полного прогона)
    "T_max_epochs": 100,
    "eta_min_lr": 1e-6,
    # Early stopping (для полного прогона)
    "patience": 15,
    # Data settings
    "data_root": './data', # Можно оставить тот же каталог
    "num_workers": 2,
}

# --- Константы для HPO ---
# <--- ИЗМЕНЕНИЕ: Обновлено имя исследования для Fashion-MNIST
STUDY_NAME = "fmnist_find_params_v1"
LR_TRIALS = [0.1, 1e-2, 1e-3, 1e-4]
ETA_LOSS_WEIGHT_TRIALS = [0.5, 1.0, 1.5, 2.0]
LAMBDA_GLOBAL_TRIALS = [0.5, 1.0, 1.5, 2.0]
EMBED_WD_TRIALS = [1e-5, 1e-6, 1e-7]

# --- 2. Вспомогательные функции (Optuna, Init - без изменений) ---

progress_bar = None

def tqdm_callback(study, trial):
    if progress_bar:
        progress_bar.update(1)

def initialize_embeddings(num_classes: int, embed_dim: int, device: torch.device) -> nn.Embedding:
    """Инициализирует эмбеддинги меток."""
    label_embedding = nn.Embedding(num_classes, embed_dim).to(device)
    print(f"Initializing {embed_dim}-dim label embeddings.")
    try:
        if embed_dim >= num_classes:
            nn.init.orthogonal_(label_embedding.weight)
            print("Applied orthogonal initialization to label embeddings.")
        else:
            print("Default initialization for label embeddings (orthogonal not possible).")
    except ValueError as e:
        print(f"Warning: Orthogonal init for embeddings failed ({e}). Using default init.")
    return label_embedding

# --- ИЗМЕНЕННАЯ ФУНКЦИЯ ЗАГРУЗКИ ДАННЫХ ---
def get_dataloaders(batch_size: int, root: str, num_workers: int, pin_memory: bool) -> Tuple[DataLoader, DataLoader]:
    """Загружает и подготавливает датасет Fashion-MNIST."""
    # <--- ИЗМЕНЕНИЕ: Используем среднее и стд. отклонение для FashionMNIST
    # Значения взяты из общедоступных источников, можно использовать (0.5,), (0.5,) для простоты
    fmnist_mean = (0.2860,)
    fmnist_std = (0.3530,)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(fmnist_mean, fmnist_std)
    ])
    # <--- ИЗМЕНЕНИЕ: Загружаем FashionMNIST вместо MNIST
    trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    # <--- ИЗМЕНЕНИЕ: Обновлено сообщение в логе
    print(f"Loaded FashionMNIST: {len(trainset)} train, {len(testset)} test samples.")
    print(f"Data normalized with Mean: {fmnist_mean}, Std: {fmnist_std}")
    return trainloader, testloader

def initialize_training_components(
    model: NoPropNet, label_embedding: nn.Embedding, config: Dict[str, Any], use_scheduler: bool = True
) -> Tuple[List[optim.Optimizer], optim.Optimizer, List[Any], Any, nn.Module, nn.Module]:
    """Инициализирует оптимизаторы, планировщики (опционально) и функции потерь."""
    LR = config['LR']
    WEIGHT_DECAY = config['WEIGHT_DECAY']
    EMBED_WD = config['EMBED_WD']

    optimizers_blocks = [
        optim.AdamW(block.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        for block in model.blocks
    ]
    params_final_classifier = model.classifier.parameters()
    params_final_embedding = label_embedding.parameters()
    optimizer_final = optim.AdamW([
        {'params': params_final_classifier, 'weight_decay': WEIGHT_DECAY},
        {'params': params_final_embedding, 'weight_decay': EMBED_WD}
    ], lr=LR)
    print(f"Initialized optimizers: LR={LR}, Blocks WD={WEIGHT_DECAY}, Classifier WD={WEIGHT_DECAY}, Embeddings WD={EMBED_WD}.")

    schedulers_blocks = []
    scheduler_final = None
    if use_scheduler:
        if 'T_max_epochs' not in config or 'eta_min_lr' not in config:
            raise ValueError("Scheduler parameters 'T_max_epochs' and 'eta_min_lr' missing from config.")
        T_max_epochs = config['T_max_epochs']
        eta_min_lr = config['eta_min_lr']
        schedulers_blocks = [
            optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max_epochs, eta_min=eta_min_lr)
            for opt in optimizers_blocks
        ]
        scheduler_final = optim.lr_scheduler.CosineAnnealingLR(optimizer_final, T_max=T_max_epochs, eta_min=eta_min_lr)
        print(f"Initialized CosineAnnealingLR schedulers with T_max={T_max_epochs}, eta_min={eta_min_lr}.")
    else:
        print("LR Scheduler is DISABLED for this run (e.g., during HPO).")

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    return optimizers_blocks, optimizer_final, schedulers_blocks, scheduler_final, mse_loss, ce_loss

# --- 3. Логика Обучения и Оценки (без изменений в логике) ---

# Функция train_epoch остается без изменений в своей логике
def train_epoch(
    epoch_num: int,
    model: NoPropNet,
    label_embedding: nn.Embedding,
    trainloader: DataLoader,
    optimizers_blocks: List[optim.Optimizer],
    optimizer_final: optim.Optimizer,
    mse_loss: nn.Module,
    ce_loss: nn.Module,
    diff_coeffs: Dict[str, torch.Tensor],
    config: Dict[str, Any]
) -> Tuple[float, float]:
    model.train()
    T = config['T']
    DEVICE = config['DEVICE']
    ETA_LOSS_WEIGHT = config.get('ETA_LOSS_WEIGHT', 1.0)
    LAMBDA_GLOBAL = config.get('LAMBDA_GLOBAL', 1.0)
    GRAD_CLIP_MAX_NORM = config['GRAD_CLIP_MAX_NORM']
    MAX_NORM_EMBED = config['MAX_NORM_EMBED']
    alphas_bar = diff_coeffs['alphas_bar']
    sqrt_alphas_bar = diff_coeffs['sqrt_alphas_bar']
    sqrt_one_minus_alphas_bar = diff_coeffs['sqrt_one_minus_alphas_bar']

    epoch_total_denoise_loss = 0.0
    epoch_total_global_target_loss = 0.0 # Можно добавить логирование, если нужно
    epoch_total_classify_loss = 0.0
    epoch_total_batches = 0

    # print(f"Starting Epoch {epoch_num + 1} Training Phase...") # Закомментировано для краткости
    for t_idx in range(T):
        block_to_train = model.blocks[t_idx]
        optimizer_t = optimizers_blocks[t_idx]
        pbar_desc = f"Epoch {epoch_num+1}, Block {t_idx+1}/{T}"
        pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=pbar_desc, leave=False)

        for i, data in pbar:
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            batch_size = inputs.shape[0]
            u_y = label_embedding(labels)
            sqrt_a_bar = sqrt_alphas_bar[t_idx].view(-1, 1).expand_as(u_y)
            sqrt_1_minus_a_bar = sqrt_one_minus_alphas_bar[t_idx].view(-1, 1).expand_as(u_y)

            # Denoise Loss
            epsilon_target = torch.randn_like(u_y)
            z_input_for_block = sqrt_a_bar * u_y.detach() + sqrt_1_minus_a_bar * epsilon_target
            predicted_epsilon = block_to_train(inputs, z_input_for_block)
            loss_denoise = mse_loss(predicted_epsilon, epsilon_target)
            epoch_total_denoise_loss += loss_denoise.item()

            # Global Target Loss
            predicted_u_t = (z_input_for_block - sqrt_1_minus_a_bar * predicted_epsilon) / (sqrt_a_bar + 1e-9)
            loss_global_target = mse_loss(predicted_u_t, u_y.detach())
            epoch_total_global_target_loss += loss_global_target.item()

            # Weighted Loss for Block
            weighted_loss_block_t = ETA_LOSS_WEIGHT * loss_denoise + LAMBDA_GLOBAL * loss_global_target

            # Classification Loss (using original u_y as input to classifier for grad separation)
            logits = model.classifier(u_y)
            classify_loss = ce_loss(logits, labels)
            epoch_total_classify_loss += classify_loss.item()

            # Combined Loss and Backprop
            total_loss_batch = weighted_loss_block_t + classify_loss
            optimizer_t.zero_grad()
            optimizer_final.zero_grad()
            total_loss_batch.backward()

            # Grad Clipping
            torch.nn.utils.clip_grad_norm_(block_to_train.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            params_to_clip_final = list(model.classifier.parameters()) + list(label_embedding.parameters())
            params_with_grad_final = [p for p in params_to_clip_final if p.grad is not None]
            if params_with_grad_final:
                 torch.nn.utils.clip_grad_norm_(params_with_grad_final, max_norm=GRAD_CLIP_MAX_NORM)

            # Optimizer Step
            optimizer_t.step()
            optimizer_final.step()

            # Embed Norm Clipping
            with torch.no_grad():
                current_norm = label_embedding.weight.data.norm()
                if current_norm > MAX_NORM_EMBED:
                    label_embedding.weight.data.mul_(MAX_NORM_EMBED / (current_norm + 1e-9))

            epoch_total_batches += 1
            if i % 100 == 0:
                 pbar.set_description(
                     f"Epoch {epoch_num+1}, Block {t_idx+1}/{T} | "
                     f"Denoise L: {loss_denoise.item():.4f} | "
                     # f"GlobalTgt L: {loss_global_target.item():.4f} | " # Раскомментировать при необходимости
                     f"Classify L: {classify_loss.item():.4f}"
                 )
        pbar.close()

    avg_epoch_denoise = epoch_total_denoise_loss / epoch_total_batches if epoch_total_batches > 0 else 0
    avg_epoch_classify = epoch_total_classify_loss / epoch_total_batches if epoch_total_batches > 0 else 0
    return avg_epoch_denoise, avg_epoch_classify

# Функция evaluate_model остается без изменений в своей логике
@torch.no_grad()
def evaluate_model(
    model: NoPropNet, testloader: DataLoader, T_steps: int, diff_coeffs: Dict[str, torch.Tensor], device: torch.device
) -> Tuple[float, int, int]:
    model.eval()
    correct = 0
    total = 0
    print("Running evaluation...")
    for data in tqdm(testloader, desc="Evaluating", leave=False):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        logits = run_inference(model, images, T_steps, diff_coeffs, device)
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Evaluation finished. Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy, correct, total

# --- 4. Подбор Гиперпараметров (Optuna - логика та же, но работает с FashionMNIST) ---

def objective(trial: optuna.trial.Trial) -> float:
    """Целевая функция Optuna для одного запуска (trial) на FashionMNIST."""
    run_config = config.copy()
    run_config['LR'] = trial.suggest_categorical('LR', LR_TRIALS)
    run_config['ETA_LOSS_WEIGHT'] = trial.suggest_categorical('ETA_LOSS_WEIGHT', ETA_LOSS_WEIGHT_TRIALS)
    run_config['LAMBDA_GLOBAL'] = trial.suggest_categorical('LAMBDA_GLOBAL', LAMBDA_GLOBAL_TRIALS)
    run_config['EMBED_WD'] = trial.suggest_categorical('EMBED_WD', EMBED_WD_TRIALS)
    run_config['EPOCHS'] = 10 # Короткий прогон для HPO

    print(f"\n--- Starting Optuna Trial {trial.number} with config: ---")
    print(f"  LR: {run_config['LR']:.1e}, ETA_L: {run_config['ETA_LOSS_WEIGHT']:.1f}, LAMBDA_G: {run_config['LAMBDA_GLOBAL']:.1f}, EmbedWD: {run_config['EMBED_WD']:.1e}")

    DEVICE = run_config['DEVICE']
    diff_coeffs = precompute_diffusion_coefficients(run_config['T'], DEVICE, run_config['s_noise_schedule'])
    label_embedding = initialize_embeddings(run_config['NUM_CLASSES'], run_config['EMBED_DIM'], DEVICE)
    model = NoPropNet(
        num_blocks=run_config['T'], embed_dim=run_config['EMBED_DIM'], num_classes=run_config['NUM_CLASSES'],
        img_channels=run_config['IMG_CHANNELS'], img_size=run_config['IMG_SIZE']
    ).to(DEVICE)
    # Загружаем FashionMNIST
    trainloader, testloader = get_dataloaders(
        run_config['BATCH_SIZE'], run_config['data_root'], run_config['num_workers'], (DEVICE == torch.device('cuda'))
    )
    optimizers_blocks, optimizer_final, _, _, mse_loss, ce_loss = \
        initialize_training_components(model, label_embedding, run_config, use_scheduler=False)

    best_trial_accuracy = 0.0
    for epoch in range(run_config['EPOCHS']):
        # print(f"  Trial {trial.number}, Epoch {epoch + 1}/{run_config['EPOCHS']}") # Закомментировано для краткости
        train_denoise_loss, train_classify_loss = train_epoch(
            epoch, model, label_embedding, trainloader, optimizers_blocks, optimizer_final,
            mse_loss, ce_loss, diff_coeffs, run_config
        )
        current_test_accuracy, _, _ = evaluate_model(
            model, testloader, run_config['T'], diff_coeffs, DEVICE
        )
        # print(f'  Trial {trial.number}, Epoch {epoch + 1} Test Acc: {current_test_accuracy:.2f}%') # Закомментировано для краткости
        best_trial_accuracy = max(best_trial_accuracy, current_test_accuracy)

        trial.report(current_test_accuracy, epoch)
        if trial.should_prune():
            print(f"--- Pruning Trial {trial.number} at epoch {epoch+1} ---")
            raise optuna.exceptions.TrialPruned()

    print(f"--- Trial {trial.number} Finished. Best Acc in {run_config['EPOCHS']} epochs: {best_trial_accuracy:.2f}% ---")
    return best_trial_accuracy

def write_optuna_plots(study):
    """Сохраняет графики результатов Optuna."""
    try:
        figname_hist = f"{study.study_name}_history.png" # <--- ИЗМЕНЕНИЕ
        figname_imp = f"{study.study_name}_param_importance.png" # <--- ИЗМЕНЕНИЕ

        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_image(figname_hist)
        print(f"Saved Optuna optimization history plot to {figname_hist}")
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_image(figname_imp)
        print(f"Saved Optuna parameter importance plot to {figname_imp}")
    except ImportError:
        print("\nPlotly or Kaleido not found. Cannot save Optuna plots. Install: pip install plotly kaleido")
    except Exception as e:
        print(f"\nError generating Optuna plots: {e}")


# --- 5. Функция для Полного Обучения (логика та же, работает с FashionMNIST) ---
def run_full_training(full_run_config: Dict[str, Any]):
    """Запускает полный цикл обучения на FashionMNIST."""
    DEVICE = full_run_config['DEVICE']
    print("\n" + "="*40)
    print("--- Running Full Training on FashionMNIST ---") # <--- ИЗМЕНЕНИЕ
    print(f"Using device: {DEVICE}")
    print("Full Training Configuration:")
    for key, value in full_run_config.items():
        print(f"  {key}: {value}")
    print("="*40 + "\n")

    diff_coeffs = precompute_diffusion_coefficients(full_run_config['T'], DEVICE, full_run_config['s_noise_schedule'])
    label_embedding = initialize_embeddings(full_run_config['NUM_CLASSES'], full_run_config['EMBED_DIM'], DEVICE)
    model = NoPropNet(
        num_blocks=full_run_config['T'], embed_dim=full_run_config['EMBED_DIM'], num_classes=full_run_config['NUM_CLASSES'],
        img_channels=full_run_config['IMG_CHANNELS'], img_size=full_run_config['IMG_SIZE']
    ).to(DEVICE)
    # Загружаем FashionMNIST
    pin_memory = (DEVICE == torch.device('cuda'))
    trainloader, testloader = get_dataloaders(full_run_config['BATCH_SIZE'], full_run_config['data_root'], full_run_config['num_workers'], pin_memory)
    optimizers_blocks, optimizer_final, schedulers_blocks, scheduler_final, mse_loss, ce_loss = \
        initialize_training_components(model, label_embedding, full_run_config, use_scheduler=True)

    print(f"\n--- Starting Full Training for {full_run_config['EPOCHS']} epochs ---")
    total_start_time = time.time()
    best_test_accuracy = 0.0
    epochs_no_improve = 0
    train_history = {'epoch': [], 'train_denoise_loss': [], 'train_classify_loss': [], 'test_accuracy': [], 'lr': []}

    for epoch in range(full_run_config['EPOCHS']):
        epoch_start_time = time.time()
        # print(f"\n--- Epoch {epoch + 1}/{full_run_config['EPOCHS']} ---") # Закомментировано для краткости

        avg_epoch_denoise, avg_epoch_classify = train_epoch(
            epoch, model, label_embedding, trainloader, optimizers_blocks, optimizer_final,
            mse_loss, ce_loss, diff_coeffs, full_run_config
        )
        epoch_time = time.time() - epoch_start_time
        # Логирование и оценка... (без изменений в логике)
        with torch.no_grad(): embedding_norm = torch.norm(label_embedding.weight.data)
        print(f">>> Epoch {epoch + 1} | Time: {epoch_time:.2f}s | Lr: {optimizer_final.param_groups[0]['lr']:.6f} | EmbNorm: {embedding_norm:.4f}")
        print(f"    Avg Losses: Denoise={avg_epoch_denoise:.7f}, Classify={avg_epoch_classify:.4f}")

        if scheduler_final:
            for scheduler in schedulers_blocks: scheduler.step()
            scheduler_final.step()

        eval_start_time = time.time()
        current_test_accuracy, correct, total = evaluate_model(
            model, testloader, full_run_config['T'], diff_coeffs, DEVICE
        )
        eval_time = time.time() - eval_start_time
        # print(f">>> Epoch {epoch + 1} Evaluation Finished. Time: {eval_time:.2f}s") # Закомментировано для краткости
        print(f'    Test Accuracy: {current_test_accuracy:.2f} % ({correct}/{total})')

        train_history['epoch'].append(epoch + 1)
        train_history['train_denoise_loss'].append(avg_epoch_denoise)
        train_history['train_classify_loss'].append(avg_epoch_classify)
        train_history['test_accuracy'].append(current_test_accuracy)
        train_history['lr'].append(optimizer_final.param_groups[0]['lr']) # Логгируем LR *после* шага для CosineAnnealing

        # Early Stopping
        patience = full_run_config.get('patience', 0) # Используем get с дефолтом 0
        if patience > 0:
            if current_test_accuracy > best_test_accuracy:
                best_test_accuracy = current_test_accuracy
                epochs_no_improve = 0
                print(f"    *** New best test accuracy: {best_test_accuracy:.2f} % ***")
                # --- ИЗМЕНЕНИЕ: Имя файла для сохранения лучшей модели ---
                best_model_filename = 'best_fmnist_noprop_model.pth'
                try:
                    torch.save(model.state_dict(), best_model_filename)
                    # print(f"    Saved best model state dict to {best_model_filename}") # Закомментировано для краткости
                except Exception as e:
                    print(f"    Error saving model: {e}")
            else:
                epochs_no_improve += 1
                # print(f"    Test accuracy did not improve for {epochs_no_improve} epochs.") # Закомментировано для краткости

            if epochs_no_improve >= patience:
                print(f"\nStopping early after {epoch + 1} epochs due to no improvement for {patience} epochs.")
                print(f"Best test accuracy achieved: {best_test_accuracy:.2f} %")
                break
        else: # Если ранняя остановка выключена (patience=0)
             if current_test_accuracy > best_test_accuracy:
                best_test_accuracy = current_test_accuracy
                # print(f"    *** New best test accuracy: {best_test_accuracy:.2f} % ***") # Закомментировано для краткости

    # --- Конец обучения ---
    print('\n' + "="*40)
    print('Finished Full Training on FashionMNIST') # <--- ИЗМЕНЕНИЕ
    total_training_time = time.time() - total_start_time
    print(f"Total Training Time: {total_training_time:.2f}s")
    print(f"Final Best Test Accuracy during run: {best_test_accuracy:.2f} %")
    print("="*40)

    # --- Построение Графиков Обучения ---
    print("\nGenerating training plots...")
    try:
        epochs_ran = train_history['epoch']
        if epochs_ran:
            fig, ax1 = plt.subplots(figsize=(12, 8))
            # Графики... (логика та же)
            color = 'tab:red'
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Avg Train Loss', color=color)
            ax1.semilogy(epochs_ran, train_history['train_classify_loss'], color=color, linestyle='-', marker='x', markersize=4, label='Avg Classify Loss (CE)')
            ax1.semilogy(epochs_ran, train_history['train_denoise_loss'], color='tab:orange', linestyle=':', marker='+', markersize=4, label='Avg Denoise Loss (MSE eps)')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.legend(loc='upper left'); ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax2 = ax1.twinx(); color = 'tab:blue'
            ax2.set_ylabel('Test Accuracy (%)', color=color)
            ax2.plot(epochs_ran, train_history['test_accuracy'], color=color, linestyle='-', marker='.', label='Test Accuracy')
            ax2.tick_params(axis='y', labelcolor=color); ax2.legend(loc='upper right')
            min_acc = min(train_history['test_accuracy']) if train_history['test_accuracy'] else 0; max_acc = max(train_history['test_accuracy']) if train_history['test_accuracy'] else 100
            ax2.set_ylim(max(0, min_acc - 5), min(100, max_acc + 5))
            ax3 = ax1.twinx(); ax3.spines['right'].set_position(('outward', 60)); color = 'tab:green'
            ax3.set_ylabel('Learning Rate', color=color)
            ax3.plot(epochs_ran, train_history['lr'], color=color, linestyle='--', label='Learning Rate (End of Epoch)')
            ax3.tick_params(axis='y', labelcolor=color); ax3.set_yscale('log'); ax3.legend(loc='lower left')

            plt.title('FashionMNIST NoProp Training Progress') # <--- ИЗМЕНЕНИЕ
            fig.tight_layout()
            # --- ИЗМЕНЕНИЕ: Имя файла для сохранения графика ---
            plot_filename = 'training_progress_fmnist.png'
            plt.savefig(plot_filename)
            print(f"Saved training progress plot to {plot_filename}")
            plt.close(fig)
        else:
             print("No epochs completed, skipping training plot generation.")
    except ImportError:
        print("\nMatplotlib not found. Install: pip install matplotlib")
    except Exception as e:
        print(f"\nError generating training plot: {e}")


# --- 6. Основной блок запуска ---
if __name__ == "__main__":

    # Установите True для запуска Optuna HPO на FashionMNIST
    # Установите False для запуска полного обучения с заданными параметрами
    RUN_HPO = False # <--- РЕКОМЕНДУЕТСЯ УСТАНОВИТЬ True для первого запуска на FashionMNIST

    if RUN_HPO:
        print("--- Starting Hyperparameter Optimization for FashionMNIST ---") # <--- ИЗМЕНЕНИЕ
        search_space = {
             'LR': LR_TRIALS,
             'ETA_LOSS_WEIGHT': ETA_LOSS_WEIGHT_TRIALS,
             'LAMBDA_GLOBAL': LAMBDA_GLOBAL_TRIALS,
             'EMBED_WD': EMBED_WD_TRIALS
        }
        n_trials = 1
        for key in search_space: n_trials *= len(search_space[key])

        study = optuna.create_study(
            study_name=STUDY_NAME, # Используем обновленное имя
            direction='maximize',
            sampler=optuna.samplers.GridSampler(search_space),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=4, interval_steps=1),
            storage="sqlite:///optuna_results_fmnist.db", # <--- ИЗМЕНЕНИЕ: Отдельная БД для FMNIST
            load_if_exists=True,
        )

        completed_trials = len(study.trials)
        progress_bar = tqdm(total=n_trials, initial=completed_trials, desc="Optuna Trials FMNIST")

        print(f"\nStarting HPO Grid Search with {n_trials} potential trials ({config['EPOCHS']} epochs each)...")
        print(f"Study storage: sqlite:///optuna_results_fmnist.db")
        print(f"Completed trials loaded: {completed_trials}")

        start_hpo_time = time.time()
        try:
            study.optimize(
                objective,
                n_trials=n_trials - completed_trials,
                timeout=60*60*24, # Ограничение по времени (24ч)
                callbacks=[tqdm_callback]
            )
        except KeyboardInterrupt: print("\n--- HPO Interrupted by User ---")
        except Exception as e: print(f"\n--- An error occurred during HPO: {e} ---")
        finally:
            end_hpo_time = time.time()
            print(f"\nTotal HPO time for this session: {end_hpo_time - start_hpo_time:.2f}s")
            if progress_bar: progress_bar.close()

        print("\n" + "="*40)
        print("--- HPO Finished ---")
        try:
             if study.best_trial:
                 print(f"Best trial number: {study.best_trial.number}")
                 print(f"Best accuracy (in {config['EPOCHS']} epochs): {study.best_value:.2f}%")
                 print("Best hyperparameters found:")
                 best_params_hpo = study.best_params
                 for key, value in best_params_hpo.items(): print(f"  {key}: {value}")
                 # <--- ИЗМЕНЕНИЕ: Имя файла для сохранения параметров ---
                 params_filename = "best_hpo_params_fmnist.txt"
                 try:
                     with open(params_filename, "w") as f:
                         f.write(f"# Results for {STUDY_NAME}\n")
                         f.write(f"Best Trial: {study.best_trial.number}\n")
                         f.write(f"Best Value (Accuracy %): {study.best_value}\n")
                         f.write("Best Params:\n")
                         for key, value in best_params_hpo.items(): f.write(f"  '{key}': {value},\n")
                     print(f"Saved best hyperparameters to {params_filename}")
                 except Exception as e: print(f"Error saving best params to file: {e}")

                 write_optuna_plots(study)
                 print("\n--- Recommendation ---")
                 print("Update the 'best_params_for_full_run' in the 'else' block below")

             else: print("No successful HPO trials completed or study was empty.")
        except Exception as e: print(f"An error occurred while processing HPO results: {e}")
        print("="*40)

    else:
        # --- ЗАПУСК ПОЛНОГО ОБУЧЕНИЯ (без HPO) ---
        print("--- Skipping HPO, running Full Training on FashionMNIST ---") # <--- ИЗМЕНЕНИЕ

        full_run_config = config.copy()

        # --- !!! ВАЖНО: Установите здесь лучшие параметры для FashionMNIST !!! ---
        # Используйте найденные HPO или начните с базовых/MNIST параметров и тюньте
        best_params_for_full_run = {
            'LR': 1e-3,             # Пример: начать с этого или с лучших от MNIST/HPO
            'ETA_LOSS_WEIGHT': 1.0, # Пример
            'LAMBDA_GLOBAL': 1.0,   # Пример
            'EMBED_WD': 1e-6,       # Пример
        }
        print("*"*10 + " WARNING: Using default parameters for full run. " + "*"*10)
        print("==> Consider running HPO first or manually setting 'best_params_for_full_run' based on previous results.")
        full_run_config.update(best_params_for_full_run)

        # Параметры, специфичные для полного прогона
        full_run_config['EPOCHS'] = 100
        full_run_config['T_max_epochs'] = 100
        full_run_config['eta_min_lr'] = 1e-6
        full_run_config['patience'] = 15 # Можно увеличить/уменьшить

        run_full_training(full_run_config)
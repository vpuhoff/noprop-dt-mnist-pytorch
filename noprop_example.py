import torch
import torch.nn as nn
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

# --- 1. Базовая Конфигурация (Значения по умолчанию и не тюнингуемые) ---
# Эти значения будут использоваться, если Optuna их не переопределит
base_config: Dict[str, Any] = {
    # Diffusion settings
    "T": 10,
    "s_noise_schedule": 0.008,

    # Model architecture
    "EMBED_DIM": 20,
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

# --- 2. Helper Functions (Без изменений) ---

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
    sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)
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

# --- 3. Model Architecture (Block predicts epsilon) ---
class DenoisingBlockPaper(nn.Module):
    def __init__(self, embed_dim: int, img_channels: int = 1, img_size: int = 28):
        super().__init__()
        self.embed_dim = embed_dim
        # ... (внутренняя архитектура без изменений) ...
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
        self.label_embed_proc = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2)
        )
        self.combined = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, self.embed_dim) # Output predicted noise
        )

    def forward(self, x: torch.Tensor, z_input: torch.Tensor) -> torch.Tensor:
        img_features = self.img_fc(self.img_conv(x))
        label_features = self.label_embed_proc(z_input)
        combined_features = torch.cat([img_features, label_features], dim=1)
        predicted_epsilon = self.combined(combined_features)
        return predicted_epsilon

class NoPropNet(nn.Module):
    def __init__(self, num_blocks: int, embed_dim: int, num_classes: int, img_channels: int = 1, img_size: int = 28):
        super().__init__()
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.blocks = nn.ModuleList([
            DenoisingBlockPaper(embed_dim, img_channels, img_size)
            for _ in range(num_blocks)
        ])
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, z_0: torch.Tensor) -> None:
        print("Warning: NoPropNet.forward is conceptual only. Use run_inference or training loop.")
        return None

# --- 4. Core Logic Functions ---

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
    embed_dim = model.embed_dim
    alphas = diff_coeffs['alphas']
    alphas_bar = diff_coeffs['alphas_bar']
    posterior_variance = diff_coeffs['posterior_variance']

    z = torch.randn(batch_size, embed_dim, device=device) # z_T

    for t_idx_rev in range(T_steps - 1, -1, -1):
        block_to_use = model.blocks[t_idx_rev]
        predicted_epsilon = block_to_use(x_batch, z)

        alpha_t = alphas[t_idx_rev]
        alpha_bar_t = alphas_bar[t_idx_rev + 1]
        beta_t = 1.0 - alpha_t
        sqrt_recip_alpha_t = 1.0 / torch.sqrt(alpha_t)
        sqrt_1m_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        mean = sqrt_recip_alpha_t * (z - beta_t / (sqrt_1m_alpha_bar_t + 1e-6) * predicted_epsilon)

        if t_idx_rev == 0:
            variance = torch.tensor(0.0, device=device)
            noise = torch.zeros_like(z)
        else:
            variance = posterior_variance[t_idx_rev]
            noise = torch.randn_like(z)

        variance_stable = torch.clamp(variance, min=0.0)
        z = mean + torch.sqrt(variance_stable + 1e-6) * noise

    logits = model.classifier(z) # Classify final z_0
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
    model.train()

    T = config['T']
    DEVICE = config['DEVICE']
    ETA_LOSS_WEIGHT = config['ETA_LOSS_WEIGHT'] # Weight from config
    GRAD_CLIP_MAX_NORM = config['GRAD_CLIP_MAX_NORM']
    MAX_NORM_EMBED = config['MAX_NORM_EMBED']

    alphas_bar = diff_coeffs['alphas_bar']
    sqrt_alphas_bar = diff_coeffs['sqrt_alphas_bar']
    sqrt_one_minus_alphas_bar = diff_coeffs['sqrt_one_minus_alphas_bar']

    # Get noise-variance based weight (precomputed preferable if fixed T)
    noise_variance_t_minus_1 = 1.0 - alphas_bar[:-1]
    denoise_loss_weight_B = (1.0 / torch.clamp(noise_variance_t_minus_1, min=1e-5)).to(DEVICE)

    for t_idx in range(T):
        block_to_train = model.blocks[t_idx]
        optimizer_t = optimizers_blocks[t_idx]

        running_loss_denoise_t = 0.0
        running_loss_classify_t = 0.0
        processed_batches_t = 0
        block_start_time = time.time()

        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            batch_size = inputs.shape[0]
            u_y = label_embedding(labels)

            # --- Denoising Loss ---
            epsilon_target = torch.randn_like(u_y)
            alpha_bar_tm1_sample = alphas_bar[t_idx]
            sqrt_a_bar = sqrt_alphas_bar[t_idx].view(-1, 1).expand_as(u_y)
            sqrt_1_minus_a_bar = sqrt_one_minus_alphas_bar[t_idx].view(-1, 1).expand_as(u_y)
            z_input_for_block = sqrt_a_bar * u_y.detach() + sqrt_1_minus_a_bar * epsilon_target
            predicted_epsilon = block_to_train(inputs, z_input_for_block)
            loss_t = mse_loss(predicted_epsilon, epsilon_target)
            running_loss_denoise_t += loss_t.item()
            # weighted_loss_t = ETA_LOSS_WEIGHT * denoise_loss_weight_B[t_idx] * loss_t # Using inverse variance weight
            weighted_loss_t = ETA_LOSS_WEIGHT * loss_t # Using simple ETA weight for HPO? Let's stick to original plan: Uniform weight * ETA

            # --- Classification Loss ---
            with torch.no_grad():
                alpha_bar_T_minus_1 = alphas_bar[T-1]
                sqrt_a_bar_Tm1 = sqrt_alphas_bar[T-1].view(-1, 1).expand_as(u_y)
                sqrt_1_minus_a_bar_Tm1 = sqrt_one_minus_alphas_bar[T-1].view(-1, 1).expand_as(u_y)
                epsilon_Tm1_target_for_clf = torch.randn_like(u_y)
                z_T_minus_1_sample = sqrt_a_bar_Tm1 * u_y.detach() + sqrt_1_minus_a_bar_Tm1 * epsilon_Tm1_target_for_clf
                predicted_epsilon_T = model.blocks[T-1](inputs, z_T_minus_1_sample)
                predicted_u_y_final = (z_T_minus_1_sample - sqrt_1_minus_a_bar_Tm1 * predicted_epsilon_T) / (sqrt_a_bar_Tm1 + 1e-6)
            logits = model.classifier(predicted_u_y_final.detach())
            classify_loss = ce_loss(logits, labels)
            running_loss_classify_t += classify_loss.item()

            # --- Combined Update ---
            total_loss_batch = weighted_loss_t + classify_loss
            optimizer_t.zero_grad()
            optimizer_final.zero_grad()
            total_loss_batch.backward()

            # Grad Clipping
            torch.nn.utils.clip_grad_norm_(block_to_train.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            if label_embedding.weight.grad is not None:
                 torch.nn.utils.clip_grad_norm_(label_embedding.parameters(), max_norm=GRAD_CLIP_MAX_NORM)

            # Optimizer Step
            optimizer_t.step()
            optimizer_final.step()

            # Embed Norm Clipping
            with torch.no_grad():
                current_norm = label_embedding.weight.data.norm()
                if current_norm > MAX_NORM_EMBED:
                    label_embedding.weight.data.mul_(MAX_NORM_EMBED / (current_norm + 1e-6))

            processed_batches_t += 1

        block_time = time.time() - block_start_time
        avg_denoise_loss_block = running_loss_denoise_t / processed_batches_t if processed_batches_t > 0 else 0
        avg_classify_loss_block = running_loss_classify_t / processed_batches_t if processed_batches_t > 0 else 0
        # Less verbose logging during HPO
        if epoch_num == 0 and t_idx == 0: # Log first block of first epoch only
             print(f'  (Trial Log Sample) Epoch {epoch_num + 1}, Block {t_idx + 1}/{T}. AvgLoss: Denoise(MSE_eps)={avg_denoise_loss_block:.7f}, Classify(CE)={avg_classify_loss_block:.4f}. Time: {block_time:.2f}s')

@torch.no_grad()
def evaluate_model(
    model: NoPropNet,
    testloader: DataLoader,
    T_steps: int,
    diff_coeffs: Dict[str, torch.Tensor],
    device: torch.device
) -> Tuple[float, int, int]:
    """Evaluates the model on the test set using the inference function."""
    model.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        logits = run_inference(model, images, T_steps, diff_coeffs, device)
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy, correct, total

# --- HPO Objective Function ---
def objective(trial: optuna.trial.Trial) -> float:
    """Optuna objective function to run one trial."""
    # Create config for this trial, starting with base
    run_config = base_config.copy()

    # --- Hyperparameters to tune ---
    run_config['LR'] = trial.suggest_categorical('LR', [5e-4, 1e-3, 2e-3])
    run_config['ETA_LOSS_WEIGHT'] = trial.suggest_categorical('ETA_LOSS_WEIGHT', [0.1, 1.0, 10.0])
    run_config['EMBED_WD'] = trial.suggest_categorical('EMBED_WD', [0.0, 1e-5, 1e-4])
    # --- Fixed parameters for HPO ---
    run_config['EPOCHS'] = 10 # Short run for HPO

    print(f"\n--- Starting Optuna Trial {trial.number} with config: ---")
    print(f"LR: {run_config['LR']:.1e}, ETA: {run_config['ETA_LOSS_WEIGHT']:.1f}, EmbedWD: {run_config['EMBED_WD']:.1e}")

    # --- Setup specific to this trial ---
    DEVICE = run_config['DEVICE']
    diff_coeffs = precompute_diffusion_coefficients(run_config['T'], DEVICE, run_config['s_noise_schedule'])
    label_embedding = initialize_embeddings(run_config['NUM_CLASSES'], run_config['EMBED_DIM'], DEVICE)
    model = NoPropNet(
        num_blocks=run_config['T'], embed_dim=run_config['EMBED_DIM'], num_classes=run_config['NUM_CLASSES'],
        img_channels=run_config['IMG_CHANNELS'], img_size=run_config['IMG_SIZE']
    ).to(DEVICE)
    trainloader, testloader = get_dataloaders(
        run_config['BATCH_SIZE'], run_config['data_root'], run_config['num_workers'], (DEVICE == torch.device('cuda'))
    )
    # Initialize optimizers etc. WITHOUT scheduler for HPO runs
    optimizers_blocks, optimizer_final, _, _, mse_loss, ce_loss = \
        initialize_training_components(model, label_embedding, run_config, use_scheduler=False)

    # --- Training and Evaluation Loop for this trial ---
    best_trial_accuracy = 0.0
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

        # Optuna Pruning (optional) - stops unpromising trials early
        trial.report(current_test_accuracy, epoch)
        if trial.should_prune():
            print(f"--- Pruning Trial {trial.number} at epoch {epoch+1} ---")
            raise optuna.exceptions.TrialPruned()

    print(f"--- Trial {trial.number} Finished. Best Acc in {run_config['EPOCHS']} epochs: {best_trial_accuracy:.2f}% ---")
    return best_trial_accuracy

def write_plots(study):
    #You can visualize results if needed (requires pip install optuna-dashboard or plotly)
    try:
        fig1 = optuna.visualization.plot_optimization_history(study)
        #fig1.show(renderer="browser")  # откроется в браузере
        fig1.write_image("opt_history.png")
        fig2 = optuna.visualization.plot_param_importances(study)
        #fig2.show(renderer="browser")
        fig2.write_image("param_importance.png")
    except ImportError:
        print("\nInstall plotly and kaleido to visualize Optuna results: pip install plotly kaleido")
    except Exception as e_vis:
        print(f"Could not plot Optuna results: {e_vis}")

# --- Main Execution Block for HPO ---
if __name__ == "__main__":
    # Define the grid search space for Optuna
    search_space = {
         'LR': [5e-4, 1e-3, 2e-3],
         'ETA_LOSS_WEIGHT': [0.1, 1.0, 10.0],
         'EMBED_WD': [0.0, 1e-5, 1e-4]
    }
    n_trials = len(search_space['LR']) * len(search_space['ETA_LOSS_WEIGHT']) * len(search_space['EMBED_WD']) # 3x3x3 = 27

    # Create the Optuna study with GridSampler
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.GridSampler(search_space),
        # Optional: Add pruning to stop bad trials early
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=4, interval_steps=1), # Prune after epoch 4 if worse than median,
        storage="sqlite:///optuna_results.db",
        load_if_exists=True,
    )

    # Run the hyperparameter optimization
    print(f"\nStarting HPO Grid Search with {n_trials} trials ({base_config['EPOCHS']} epochs each)...")
    start_hpo_time = time.time()
    try:
        try:
            study.optimize(objective, n_trials=n_trials, timeout=60*60*24) # Added 24-hour timeout
        except KeyboardInterrupt:
            print("Прерывание: сохраняю текущий прогресс Optuna...")
            print(f"Проведено итераций: {len(study.trials)}")
            print("Лучшие параметры на текущий момент:", study.best_params)
            write_plots(study)
    except Exception as e:
        print(f"An error occurred during HPO: {e}")
    finally:
        end_hpo_time = time.time()
        print(f"Total HPO time: {end_hpo_time - start_hpo_time:.2f}s")

    # Print the results
    print("\n--- HPO Finished ---")
    if study.best_trial:
        print(f"Best trial number: {study.best_trial.number}")
        print(f"Best accuracy (in {base_config['EPOCHS']} epochs): {study.best_value:.2f}%")
        print("Best hyperparameters found:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        print("\n--- Recommendations ---")
        print(f"Use these 'Best hyperparameters' for a full training run (e.g., {base_config['T_max_epochs']} epochs) with the LR scheduler enabled.")
    else:
        print("No successful trials completed.")


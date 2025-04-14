import optuna
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
from optuna.samplers import GridSampler
import plotly
import kaleido
from tqdm import tqdm

# --- 1. Базовая Конфигурация ---
config: Dict[str, Any] = {
    # Diffusion settings
    "T": 10,
    "s_noise_schedule": 0.008,
    # Model architecture
    "EMBED_DIM": 20,
    "NUM_CLASSES": 10,
    "IMG_SIZE": 28,
    "IMG_CHANNELS": 1,
    # Training settings
    "BATCH_SIZE": 128,
    "EPOCHS": 10, # Устанавливаем для полного прогона
    "LR": 1e-3,    # Начальный LR для полного прогона (или лучший из HPO)
    "WEIGHT_DECAY": 1e-3,
    "EMBED_WD": 1e-5,
    "MAX_NORM_EMBED": 50.0,
    "GRAD_CLIP_MAX_NORM": 1.0,
    "ETA_LOSS_WEIGHT": 0.5,  # Вес для Denoise Loss (MSE эпсилон)
    "LAMBDA_GLOBAL": 1.0,  # Вес для Global Target Loss (MSE u_y) - НОВЫЙ
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # Scheduler settings
    "T_max_epochs": 100, # Должно совпадать с EPOCHS для полного прогона
    "eta_min_lr": 1e-6,
    # Early stopping
    "patience": 15,
    # Data settings
    "data_root": './data',
    "num_workers": 2,
}

STUDY_NAME = "find_params_v6"
LR_TRIALS = [0.1, 1e-2, 1e-3, 1e-4]
ETA_LOSS_WEIGHT_TRIALS = [0.5, 1.0, 1.5, 2.0]
EMBED_WD_TRIALS = [1e-5, 1e-6, 1e-7]

# --- 2. Helper Functions ---
progress_bar = None

def tqdm_callback(study, trial):
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
    LR = config['LR']
    WEIGHT_DECAY = config['WEIGHT_DECAY']
    EMBED_WD = config['EMBED_WD']

    optimizers_blocks = [optim.AdamW(block.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) for block in model.blocks]
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
        schedulers_blocks = [optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max_epochs, eta_min=eta_min_lr) for opt in optimizers_blocks]
        scheduler_final = optim.lr_scheduler.CosineAnnealingLR(optimizer_final, T_max=T_max_epochs, eta_min=eta_min_lr)
        print(f"Initialized CosineAnnealingLR schedulers with T_max={T_max_epochs}, eta_min={eta_min_lr}.")
    else:
        print("LR Scheduler is DISABLED for this run.")

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    return optimizers_blocks, optimizer_final, schedulers_blocks, scheduler_final, mse_loss, ce_loss

# --- 3. Model Architecture ---
class DenoisingBlockPaper(nn.Module):
    def __init__(self, embed_dim: int, img_channels: int = 1, img_size: int = 28):
        super().__init__()
        self.embed_dim = embed_dim
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
            nn.Linear(128, self.embed_dim)
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
    model: NoPropNet, x_batch: torch.Tensor, T_steps: int, diff_coeffs: Dict[str, torch.Tensor], device: torch.device
) -> torch.Tensor:
    # ... (Код функции run_inference без изменений) ...
    batch_size = x_batch.shape[0]
    embed_dim = model.embed_dim
    alphas = diff_coeffs['alphas']
    alphas_bar = diff_coeffs['alphas_bar']
    posterior_variance = diff_coeffs['posterior_variance']
    sqrt_recip_alphas = diff_coeffs['sqrt_recip_alphas']
    sqrt_one_minus_alphas_bar = diff_coeffs['sqrt_one_minus_alphas_bar']

    z = torch.randn(batch_size, embed_dim, device=device) # z_T
    for t_idx_rev in range(T_steps - 1, -1, -1):
        block_to_use = model.blocks[t_idx_rev]
        predicted_epsilon = block_to_use(x_batch, z)
        alpha_t = alphas[t_idx_rev]
        alpha_bar_t = alphas_bar[t_idx_rev + 1]
        beta_t = 1.0 - alpha_t
        sqrt_recip_alpha_t = sqrt_recip_alphas[t_idx_rev]
        sqrt_1m_alpha_bar_t = sqrt_one_minus_alphas_bar[t_idx_rev + 1]
        mean = sqrt_recip_alpha_t * (z - beta_t / (sqrt_1m_alpha_bar_t + 1e-9) * predicted_epsilon)
        if t_idx_rev == 0:
            variance = torch.tensor(0.0, device=device)
            noise = torch.zeros_like(z)
        else:
            variance = posterior_variance[t_idx_rev]
            noise = torch.randn_like(z)
        variance_stable = torch.clamp(variance, min=0.0)
        z = mean + torch.sqrt(variance_stable + 1e-6) * noise
    logits = model.classifier(z)
    return logits


# --- МОДИФИЦИРОВАННАЯ ФУНКЦИЯ TRAIN_EPOCH ---
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
) -> Tuple[float, float]: # Возвращает средние потери за эпоху
    """Runs one epoch of training using the HYBRID loss approach."""
    model.train() # Set model to training mode

    T = config['T']
    DEVICE = config['DEVICE']
    ETA_LOSS_WEIGHT = config.get('ETA_LOSS_WEIGHT', 1.0) # Вес для локальной MSE(eps)
    LAMBDA_GLOBAL = config.get('LAMBDA_GLOBAL', 1.0) # Вес для глобальной цели MSE(u_y)
    GRAD_CLIP_MAX_NORM = config['GRAD_CLIP_MAX_NORM']
    MAX_NORM_EMBED = config['MAX_NORM_EMBED']

    # Access precomputed coefficients
    alphas_bar = diff_coeffs['alphas_bar']
    sqrt_alphas_bar = diff_coeffs['sqrt_alphas_bar']
    sqrt_one_minus_alphas_bar = diff_coeffs['sqrt_one_minus_alphas_bar']

    # Аккумуляторы для средних потерь за всю эпоху
    epoch_total_denoise_loss = 0.0
    epoch_total_global_target_loss = 0.0
    epoch_total_classify_loss = 0.0
    epoch_total_batches = 0

    print(f"Starting Epoch {epoch_num + 1} Training Phase...")
    # --- Outer loop for time step t (training each block) ---
    for t_idx in range(T): # t_idx = 0..T-1 corresponds to paper step t=1..T
        block_to_train = model.blocks[t_idx]
        optimizer_t = optimizers_blocks[t_idx]

        # Progress bar for batches within a block training pass
        pbar_desc = f"Epoch {epoch_num+1}, Block {t_idx+1}/{T}"
        pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=pbar_desc, leave=False)

        # --- Inner loop over mini-batches ---
        for i, data in pbar:
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            batch_size = inputs.shape[0]

            # --- Common Setup ---
            u_y = label_embedding(labels) # Needs grad for optimizer_final

            # Get noise schedule values for z_{t-1} (index t_idx)
            alpha_bar_tm1_val = alphas_bar[t_idx]
            sqrt_a_bar = sqrt_alphas_bar[t_idx].view(-1, 1).expand_as(u_y)
            sqrt_1_minus_a_bar = sqrt_one_minus_alphas_bar[t_idx].view(-1, 1).expand_as(u_y)

            # --- 1. Local Denoising Loss Calculation (Predict Epsilon) ---
            epsilon_target = torch.randn_like(u_y)
            z_input_for_block = sqrt_a_bar * u_y.detach() + sqrt_1_minus_a_bar * epsilon_target
            predicted_epsilon = block_to_train(inputs, z_input_for_block)
            loss_t = mse_loss(predicted_epsilon, epsilon_target)
            epoch_total_denoise_loss += loss_t.item() # Accumulate unweighted loss

            # --- 2. Global Target Loss Calculation (Predict u_y implicitly) ---
            # Reconstruct predicted clean u_t from predicted_epsilon (requires grad on predicted_epsilon)
            predicted_u_t = (z_input_for_block - sqrt_1_minus_a_bar * predicted_epsilon) / (sqrt_a_bar + 1e-9)
            # Calculate MSE between predicted u_t and detached clean u_y
            loss_global_target = mse_loss(predicted_u_t, u_y.detach())
            epoch_total_global_target_loss += loss_global_target.item() # Accumulate unweighted loss

            # --- 3. Weighted Combined Loss for Block Update ---
            weighted_loss_block_t = ETA_LOSS_WEIGHT * loss_t + LAMBDA_GLOBAL * loss_global_target

            # --- 4. Classification Loss Calculation (using final predicted embedding u_hat_T) ---
            # Recalculated each batch iteration to update classifier/embeddings
            with torch.no_grad():
                # Sample z_{T-1} for input to the last block
                alpha_bar_T_minus_1 = alphas_bar[T-1]
                sqrt_a_bar_Tm1 = sqrt_alphas_bar[T-1].view(-1, 1).expand_as(u_y)
                sqrt_1_minus_a_bar_Tm1 = sqrt_one_minus_alphas_bar[T-1].view(-1, 1).expand_as(u_y)
                epsilon_Tm1_target_for_clf = torch.randn_like(u_y)
                # Sample z_{T-1} using u_y.detach() as base
                z_T_minus_1_sample = sqrt_a_bar_Tm1 * u_y.detach() + sqrt_1_minus_a_bar_Tm1 * epsilon_Tm1_target_for_clf

                # Get predicted noise epsilon_T from the *last* block
                predicted_epsilon_T = model.blocks[T-1](inputs, z_T_minus_1_sample)

                # Reconstruct the predicted clean embedding u_hat_T (predicted z_0)
                predicted_u_y_final = (z_T_minus_1_sample - sqrt_1_minus_a_bar_Tm1 * predicted_epsilon_T) / (sqrt_a_bar_Tm1 + 1e-9)

            # Calculate classification loss using the reconstructed clean embedding
            # Detach input to classifier
            logits = model.classifier(predicted_u_y_final.detach())
            classify_loss = ce_loss(logits, labels)
            epoch_total_classify_loss += classify_loss.item() # Accumulate classify loss

            # --- 5. Combined Loss and Backpropagation ---
            total_loss_batch = weighted_loss_block_t + classify_loss

            optimizer_t.zero_grad()
            optimizer_final.zero_grad()

            total_loss_batch.backward() # Calculate gradients from combined loss

            # --- 6. Gradient Clipping ---
            torch.nn.utils.clip_grad_norm_(block_to_train.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            # Check if classifier/embedding grads exist before clipping
            if any(p.grad is not None for p in model.classifier.parameters()):
                 torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            if label_embedding.weight.grad is not None:
                 torch.nn.utils.clip_grad_norm_(label_embedding.parameters(), max_norm=GRAD_CLIP_MAX_NORM)

            # --- 7. Optimizer Step ---
            optimizer_t.step()
            optimizer_final.step()

            # --- 8. Embed Norm Clipping ---
            with torch.no_grad():
                current_norm = label_embedding.weight.data.norm()
                if current_norm > MAX_NORM_EMBED:
                    label_embedding.weight.data.mul_(MAX_NORM_EMBED / (current_norm + 1e-9))

            epoch_total_batches += 1 # Count total batch updates in epoch

            # Update progress bar description (optional)
            if i % 100 == 0:
                 pbar.set_description(
                     f"Epoch {epoch_num+1}, Block {t_idx+1}/{T} | "
                     f"Denoise L: {loss_t.item():.4f} | "
                     f"GlobalTgt L: {loss_global_target.item():.4f} | "
                     f"Classify L: {classify_loss.item():.4f}"
                 )
            # --- End of batch loop ---
        pbar.close()
        # --- End of block training loop (t_idx) ---

    # Calculate average losses for the epoch
    avg_epoch_denoise = epoch_total_denoise_loss / epoch_total_batches if epoch_total_batches > 0 else 0
    avg_epoch_classify = epoch_total_classify_loss / epoch_total_batches if epoch_total_batches > 0 else 0

    # Return average losses for logging outside the function
    return avg_epoch_denoise, avg_epoch_classify

@torch.no_grad()
def evaluate_model(
    model: NoPropNet, testloader: DataLoader, T_steps: int, diff_coeffs: Dict[str, torch.Tensor], device: torch.device
) -> Tuple[float, int, int]:
    model.eval()
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
    run_config = config.copy()

    # --- Hyperparameters to tune ---
    run_config['LR'] = trial.suggest_categorical('LR', LR_TRIALS)
    run_config['ETA_LOSS_WEIGHT'] = trial.suggest_categorical('ETA_LOSS_WEIGHT', ETA_LOSS_WEIGHT_TRIALS)
    run_config['EMBED_WD'] = trial.suggest_categorical('EMBED_WD', EMBED_WD_TRIALS)
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
    fig1 = optuna.visualization.plot_optimization_history(study)
    #fig1.show(renderer="browser")  # откроется в браузере
    fig1.write_image("opt_history.png")
    fig2 = optuna.visualization.plot_param_importances(study)
    #fig2.show(renderer="browser")
    fig2.write_image("param_importance.png")

# --- Функция для полного обучения (Запускает основной цикл) ---
def run_full_training(config: Dict[str, Any]):
    """Запускает полный цикл обучения с заданными параметрами, планировщиком и ранней остановкой."""
    DEVICE = config['DEVICE']
    print("--- Running Full Training (Hybrid Loss) ---")
    print(f"Using device: {DEVICE}")
    print("Full Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 30)

    # 1. Precompute
    diff_coeffs = precompute_diffusion_coefficients(config['T'], DEVICE, config['s_noise_schedule'])
    # 2. Embeddings
    label_embedding = initialize_embeddings(config['NUM_CLASSES'], config['EMBED_DIM'], DEVICE)
    # 3. Model
    model = NoPropNet(
        num_blocks=config['T'], embed_dim=config['EMBED_DIM'], num_classes=config['NUM_CLASSES'],
        img_channels=config['IMG_CHANNELS'], img_size=config['IMG_SIZE']
    ).to(DEVICE)
    # 4. Data
    pin_memory = True if DEVICE == torch.device('cuda') else False
    trainloader, testloader = get_dataloaders(config['BATCH_SIZE'], config['data_root'], config['num_workers'], pin_memory)
    # 5. Training Components (with scheduler)
    optimizers_blocks, optimizer_final, schedulers_blocks, scheduler_final, mse_loss, ce_loss = \
        initialize_training_components(model, label_embedding, config, use_scheduler=True)

    # 6. Training Loop & Logging Setup
    print(f"\n--- Starting Full Training for {config['EPOCHS']} epochs ---")
    total_start_time = time.time()
    best_test_accuracy = 0.0
    epochs_no_improve = 0
    train_history = {'epoch': [], 'train_denoise_loss': [], 'train_classify_loss': [], 'test_accuracy': [], 'lr': []}

    for epoch in range(config['EPOCHS']):
        epoch_start_time = time.time()

        # Запускаем обучение на одну эпоху и получаем средние потери
        avg_epoch_denoise, avg_epoch_classify = train_epoch(
            epoch, model, label_embedding, trainloader, optimizers_blocks, optimizer_final,
            mse_loss, ce_loss, diff_coeffs, config
        )

        epoch_time = time.time() - epoch_start_time
        with torch.no_grad():
            embedding_norm = torch.norm(label_embedding.weight.data)
        print(f"--- Epoch {epoch + 1} finished. Training Time: {epoch_time:.2f}s ---")
        print(f"--- Avg Epoch Losses: Denoise(MSE_eps)={avg_epoch_denoise:.7f}, Classify(CE)={avg_epoch_classify:.4f} ---")
        print(f"--- End of Epoch {epoch + 1}. Embedding Norm: {embedding_norm:.4f} ---")

        # Шаг планировщиков LR
        current_lr = optimizer_final.param_groups[0]['lr']
        print(f"--- End of Epoch {epoch + 1}. LR before step: {current_lr:.6f} ---")
        if scheduler_final:
            for scheduler in schedulers_blocks: scheduler.step()
            scheduler_final.step()
            current_lr_after = optimizer_final.param_groups[0]['lr']
            print(f"--- End of Epoch {epoch + 1}. LR after step: {current_lr_after:.6f} ---")
        else:
            print("--- End of Epoch {epoch + 1}. LR Scheduler was not used. ---")
            current_lr_after = current_lr # Если планировщика нет

        # Оценка после каждой эпохи
        print(f"--- Running Evaluation for Epoch {epoch + 1} ---")
        eval_start_time = time.time()
        current_test_accuracy, correct, total = evaluate_model(
            model, testloader, config['T'], diff_coeffs, DEVICE
        )
        eval_time = time.time() - eval_start_time
        print(f'>>> Epoch {epoch + 1} Test Accuracy: {current_test_accuracy:.2f} % ({correct}/{total}) <<<')
        print(f"Evaluation Time: {eval_time:.2f}s")

        # Запись истории для графика
        train_history['epoch'].append(epoch + 1)
        train_history['train_denoise_loss'].append(avg_epoch_denoise)
        train_history['train_classify_loss'].append(avg_epoch_classify)
        train_history['test_accuracy'].append(current_test_accuracy)
        train_history['lr'].append(current_lr) # Логгируем LR до шага

        # Логика Ранней Остановки
        if 'patience' in config:
            if current_test_accuracy > best_test_accuracy:
                best_test_accuracy = current_test_accuracy
                epochs_no_improve = 0
                print(f"*** New best test accuracy: {best_test_accuracy:.2f} % ***")
                # torch.save(model.state_dict(), 'best_model_hybrid.pth') # Сохраняем лучшую модель
            else:
                epochs_no_improve += 1
                print(f"Test accuracy did not improve for {epochs_no_improve} epochs.")

            if epochs_no_improve >= config['patience']:
                print(f"\nStopping early after {epoch + 1} epochs due to no improvement for {config['patience']} epochs.")
                print(f"Best test accuracy achieved: {best_test_accuracy:.2f} %")
                break
        else: # Если patience не задан
             if current_test_accuracy > best_test_accuracy:
                 best_test_accuracy = current_test_accuracy
                 print(f"*** New best test accuracy: {best_test_accuracy:.2f} % ***")

    # 7. Конец обучения
    print('\nFinished Full Training')
    total_training_time = time.time() - total_start_time
    print(f"Total Training Time: {total_training_time:.2f}s")
    print(f"Final Best Test Accuracy during run: {best_test_accuracy:.2f} %")

    # --- Построение Графиков ---
    print("\nGenerating training plots...")
    try:
        import matplotlib.pyplot as plt
        epochs_ran = train_history['epoch']
        if epochs_ran:
            fig, ax1 = plt.subplots(figsize=(12, 8)) # Сделаем повыше для 3х графиков

            # График потерь
            color = 'tab:red'
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Avg Train Loss', color=color)
            # Используем логарифмическую шкалу для потерь, т.к. они могут сильно различаться
            ax1.semilogy(epochs_ran, train_history['train_classify_loss'], color=color, linestyle='-', label='Avg Classify Loss (CE)')
            ax1.semilogy(epochs_ran, train_history['train_denoise_loss'], color='tab:orange', linestyle=':', label='Avg Denoise Loss (MSE eps)')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.legend(loc='upper left')
            ax1.grid(True)

            # График точности
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Test Accuracy (%)', color=color)
            ax2.plot(epochs_ran, train_history['test_accuracy'], color=color, linestyle='-', marker='.', label='Test Accuracy')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.legend(loc='upper right')

            #График LR (на третьей оси Y, если нужно)
            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('outward', 60)) # Сдвигаем третью ось
            color = 'tab:green'
            ax3.set_ylabel('Learning Rate', color=color)
            ax3.plot(epochs_ran, train_history['lr'], color=color, linestyle='--', label='Learning Rate')
            ax3.tick_params(axis='y', labelcolor=color)
            ax3.set_yscale('log') # Лог шкала для LR
            ax3.legend(loc='lower left')


            plt.title('Hybrid NoProp Training Progress')
            fig.tight_layout()
            plt.savefig('training_progress_hybrid.png') # Новое имя файла
            print("Saved training progress plot to training_progress_hybrid.png")
            # plt.show()
        else:
             print("No epochs completed, skipping plot generation.")
    except ImportError:
        print("\nMatplotlib not found. Install: pip install matplotlib")
    except Exception as e:
        print(f"\nError generating plot: {e}")


# --- Основной блок ---
if __name__ == "__main__":

    RUN_HPO = False # Установите True для запуска Optuna

    if RUN_HPO:
        print("--- Starting Hyperparameter Optimization using Optuna ---")
        search_space = {
             'LR': LR_TRIALS, 
             'ETA_LOSS_WEIGHT': ETA_LOSS_WEIGHT_TRIALS,
             'EMBED_WD': EMBED_WD_TRIALS
        }
        n_trials = len(search_space['LR']) * len(search_space['ETA_LOSS_WEIGHT']) * len(search_space['EMBED_WD'])

        study = optuna.create_study(
            study_name=STUDY_NAME, 
            direction='maximize',
            sampler=optuna.samplers.GridSampler(search_space),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=4, interval_steps=1),
            storage="sqlite:///optuna_results.db", 
            load_if_exists=True,
        )

        progress_bar = tqdm(total=n_trials, initial=len(study.trials))
        print(f"\nStarting HPO Grid Search with {n_trials} trials ({config['EPOCHS']} epochs each)...")
        start_hpo_time = time.time()
        try:
            study.optimize(objective, n_trials=n_trials, timeout=60*60*24, callbacks=[tqdm_callback]) # Используем objective
        except KeyboardInterrupt:
            print("Прерывание: сохраняю текущий прогресс Optuna...")
            print("Прерывание: сохраняю текущий прогресс Optuna...")
            print(f"Проведено итераций: {len(study.trials)}")
            print("Лучшие параметры на текущий момент:", study.best_params)
            write_plots(study)
        except Exception as e:
             print(f"An error occurred during HPO: {e}")
        finally:
             end_hpo_time = time.time()
             print(f"Total HPO time: {end_hpo_time - start_hpo_time:.2f}s")
             progress_bar.close()

        # Вывод результатов HPO
        print("\n--- HPO Finished ---")
        if study.best_trial:
            print(f"Best trial number: {study.best_trial.number}")
            print(f"Best accuracy (in {config['EPOCHS']} epochs): {study.best_value:.2f}%")
            print("Best hyperparameters found:")
            for key, value in study.best_params.items():
                print(f"  {key}: {value}")
            print("\n--- Recommendations ---")
            print(f"Use these 'Best hyperparameters' for a full training run...")
        else:
            print("No successful HPO trials completed.")
        write_plots(study) # Функция для отрисовки графиков Optuna
    else:
        # Запускаем полный прогон с выбранной конфигурацией
        # Берем базовую и переопределяем параметры для гибридного прогона
        hybrid_config = config.copy()

        # --- Параметры для полного прогона гибридной модели ---
        # Основываемся на лучших параметрах предыдущего успешного запуска
        # и добавляем LAMBDA_GLOBAL
        hybrid_config['LR'] = 0.01 # Попробуем снова 1e-3 с планировщиком
        hybrid_config['ETA_LOSS_WEIGHT'] = 2.0 # Вес для MSE(eps)
        hybrid_config['LAMBDA_GLOBAL'] = 1.0 # Вес для MSE(u_y) - НОВЫЙ, можно тюнить
        hybrid_config['EMBED_WD'] = 1e-7 # Маленький WD для эмбеддингов
        hybrid_config['WEIGHT_DECAY'] = 1e-3 # WD для остального

        # Параметры полного прогона
        hybrid_config['EPOCHS'] = 100
        hybrid_config['T_max_epochs'] = 100 # Для планировщика LR
        hybrid_config['eta_min_lr'] = 1e-6
        hybrid_config['patience'] = 15

        # Запускаем полный прогон
        run_full_training(hybrid_config)
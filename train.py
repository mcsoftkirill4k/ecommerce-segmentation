"""
BiRefNet Fine-tuning Script

"""

import os
import gc
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
from transformers import AutoModelForImageSegmentation, get_cosine_schedule_with_warmup
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def atomic_torch_save(obj, path: Path):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise

# =====================
# CONFIGURATION - TWO-PHASE TRAINING
# =====================
UNIFIED_ROOT = Path(r"D:\DataSet\unified_matting")
TRAIN_IMG_DIR = UNIFIED_ROOT / "train" / "images"
TRAIN_ALPHA_DIR = UNIFIED_ROOT / "train" / "alpha"
VAL_IMG_DIR = UNIFIED_ROOT / "val" / "images"
VAL_ALPHA_DIR = UNIFIED_ROOT / "val" / "alpha"

# LMDB paths
TRAIN_LMDB = Path(r"D:\DataSet\unified_train.lmdb")
VAL_LMDB = Path(r"D:\DataSet\unified_val.lmdb")

USE_LMDB = True  # True для LMDB, False для PNG

# SINGLE PHASE: Keep encoder FROZEN throughout (no unfreeze)
PHASE1_EPOCHS = 999  # Disabled phase switching
PHASE1_BATCH_SIZE = 2
PHASE1_ACCUM_STEPS = 2

# PHASE 2: DISABLED to avoid instability from encoder unfreeze
PHASE2_BATCH_SIZE = 2  # Same as Phase1
PHASE2_ACCUM_STEPS = 2

NUM_WORKERS = 0  # 0 = без multiprocessing, чтобы избежать memory leak на Windows
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_EVERY_ITERS = 1000
TRAIN_LOG_EVERY_ITERS = 100

CHECKPOINT_DIR = Path(r"D:\\birefnet_checkpoints")

RESUME_CHECKPOINT_PATH = None

USE_EDGE_WEIGHTED_LOSS = False  # Отключено: фокус на краях приводит к переобучению на текстуры
EDGE_WEIGHT_FACTOR = 1.0
EARLY_STOPPING_METRIC = 'val_mse'
PATIENCE = 6  # Increased for more stable convergence
EXPECTED_SIZE = 1024

# Background augmentation extras: shadows + white-on-white low contrast (REDUCED 2-3x)
ENABLE_SYNTHETIC_SHADOWS = True
SHADOW_PROB = 0.10
SHADOW_STRENGTH_RANGE = (0.05, 0.12)
SHADOW_BLUR_SIGMA_RANGE = (8.0, 20.0)
SHADOW_OFFSET_RANGE = (-20, 20)

ENABLE_WHITE_ON_WHITE = True
WHITE_ON_WHITE_PROB = 0.10
WHITE_ON_WHITE_STRENGTH_RANGE = (0.03, 0.10)
WHITE_ON_WHITE_BG_MEAN_THRESH = 200


def apply_synthetic_shadow(bg: np.ndarray, alpha_float: np.ndarray) -> np.ndarray:
    """Add soft shadow to background using shifted/blurred alpha (GT unchanged)."""
    if not ENABLE_SYNTHETIC_SHADOWS or np.random.rand() > SHADOW_PROB:
        return bg

    h, w = bg.shape[:2]
    shadow = alpha_float.squeeze()
    sigma = np.random.uniform(*SHADOW_BLUR_SIGMA_RANGE)
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=sigma, sigmaY=sigma)

    dx = np.random.randint(SHADOW_OFFSET_RANGE[0], SHADOW_OFFSET_RANGE[1] + 1)
    dy = np.random.randint(SHADOW_OFFSET_RANGE[0], SHADOW_OFFSET_RANGE[1] + 1)
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shadow = cv2.warpAffine(shadow, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    shadow = np.clip(shadow, 0.0, 1.0)

    strength = np.random.uniform(*SHADOW_STRENGTH_RANGE)
    bg_float = bg.astype(np.float32)
    bg_float = bg_float * (1.0 - strength * shadow[..., None])
    return np.clip(bg_float, 0, 255).astype(np.uint8)


def apply_white_on_white_fg(fg: np.ndarray, bg_mean: float) -> np.ndarray:
    """Lighten FG to simulate low-contrast white-on-white cases."""
    if not ENABLE_WHITE_ON_WHITE or bg_mean < WHITE_ON_WHITE_BG_MEAN_THRESH:
        return fg
    if np.random.rand() > WHITE_ON_WHITE_PROB:
        return fg

    strength = np.random.uniform(*WHITE_ON_WHITE_STRENGTH_RANGE)
    fg_float = fg.astype(np.float32)
    fg_float = fg_float * (1.0 - strength) + 255.0 * strength
    return np.clip(fg_float, 0, 255).astype(np.uint8)


# =====================
# DATASET CLASSES
# =====================
class UnifiedMattingDataset(Dataset):
    def __init__(self, img_dir, alpha_dir, transform=None, size=1024,
                 use_bg_augmentation=True, p_white_bg=0.5):
        self.img_dir = Path(img_dir)
        self.alpha_dir = Path(alpha_dir)
        self.transform = transform
        self.size = size
        self.use_bg_augmentation = use_bg_augmentation
        self.p_white_bg = p_white_bg
        
        self.img_paths = sorted(
            list(self.img_dir.glob("*.jpg")) + 
            list(self.img_dir.glob("*.jpeg")) + 
            list(self.img_dir.glob("*.png"))
        )
        
        self.alpha_map = {}
        for alpha_path in self.alpha_dir.glob("*.png"):
            self.alpha_map[alpha_path.stem] = alpha_path
        
        self.samples = []
        for img_path in self.img_paths:
            if img_path.stem in self.alpha_map:
                self.samples.append({
                    'img_path': img_path,
                    'alpha_path': self.alpha_map[img_path.stem]
                })
        
        print(f"UnifiedMattingDataset: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def composite_with_white_bg(self, image, alpha):
        """DIVERSE background augmentation: white/gray/colored/textured"""
        if np.random.rand() > self.p_white_bg:
            return image
        
        h, w = image.shape[:2]
        alpha_float = alpha.astype(np.float32) / 255.0
        if alpha_float.ndim == 2:
            alpha_float = alpha_float[:, :, np.newaxis]
        
        r = np.random.rand()
        
        # 25% - чистый белый
        if r < 0.25:
            bg = np.full((h, w, 3), 255, dtype=np.uint8)
        
        # 25% - серый (120-200) чтобы модель не учила "белое=фон"
        elif r < 0.5:
            gray_val = np.random.randint(120, 200)
            bg = np.full((h, w, 3), gray_val, dtype=np.uint8)
            # Добавляем небольшой шум
            noise = np.random.normal(0, 5.0, size=(h, w, 3))
            bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # 25% - цветной фон (голубой/бежевый/зеленоватый)
        elif r < 0.75:
            # Случайный светлый оттенок
            base_r = np.random.randint(200, 245)
            base_g = np.random.randint(200, 245)
            base_b = np.random.randint(200, 245)
            bg = np.full((h, w, 3), [base_r, base_g, base_b], dtype=np.uint8)
            noise = np.random.normal(0, 8.0, size=(h, w, 3))
            bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # 25% - градиент (светлый в темный или наоборот)
        else:
            top_val = np.random.randint(180, 255)
            bottom_val = np.random.randint(120, 220)
            if np.random.rand() > 0.5:
                top_val, bottom_val = bottom_val, top_val
            gradient = np.linspace(top_val / 255.0, bottom_val / 255.0, h).reshape(-1, 1, 1)
            bg = (np.full((h, w, 3), 200, dtype=np.float32) * gradient).astype(np.uint8)
        
        bg = apply_synthetic_shadow(bg, alpha_float)
        bg_mean = float(bg.mean())
        fg = apply_white_on_white_fg(image, bg_mean)
        
        img_float = fg.astype(np.float32)
        bg_float = bg.astype(np.float32)
        out = img_float * alpha_float + bg_float * (1.0 - alpha_float)
        return np.clip(out, 0, 255).astype(np.uint8)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = cv2.imread(str(sample['img_path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        alpha = cv2.imread(str(sample['alpha_path']), cv2.IMREAD_GRAYSCALE)
        
        if image.shape[:2] != (self.size, self.size):
            image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            alpha = cv2.resize(alpha, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        
        if self.use_bg_augmentation:
            image = self.composite_with_white_bg(image, alpha)
        
        if self.transform:
            augmented = self.transform(image=image, mask=alpha)
            image = augmented['image']
            alpha = augmented['mask']
        
        alpha = alpha.float() / 255.0
        return {'image': image, 'mask': alpha.unsqueeze(0)}

class UnifiedLMDBDataset(Dataset):
    def __init__(self, lmdb_path, transform=None, size=1024,
                 use_bg_augmentation=True, p_white_bg=0.5):
        import lmdb
        lmdb_path = Path(lmdb_path).resolve()
        
        if not lmdb_path.exists():
            raise FileNotFoundError(f"LMDB не найден: {lmdb_path}")
        
        self.lmdb_path = str(lmdb_path)
        self.env = None
        
        temp_env = lmdb.open(self.lmdb_path, readonly=True, lock=False, max_dbs=0, readahead=False, meminit=False)
        with temp_env.begin() as txn:
            self.length = txn.stat()['entries']
        temp_env.close()
        
        self.transform = transform
        self.size = size
        self.use_bg_augmentation = use_bg_augmentation
        self.p_white_bg = p_white_bg
        print(f"LMDB Dataset: {self.length} samples")
    
    def _open_lmdb(self):
        if self.env is None:
            import lmdb
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, 
                                max_dbs=0, readahead=False, meminit=False)
    
    def __len__(self):
        return self.length
    
    def composite_with_white_bg(self, image, alpha):
        """DIVERSE background augmentation: white/gray/colored/textured"""
        if np.random.rand() > self.p_white_bg:
            return image
        
        h, w = image.shape[:2]
        alpha_float = alpha.astype(np.float32) / 255.0
        if alpha_float.ndim == 2:
            alpha_float = alpha_float[:, :, np.newaxis]
        
        r = np.random.rand()
        
        # 25% - чистый белый
        if r < 0.25:
            bg = np.full((h, w, 3), 255, dtype=np.uint8)
        
        # 25% - серый (120-200) чтобы модель не учила "белое=фон"
        elif r < 0.5:
            gray_val = np.random.randint(120, 200)
            bg = np.full((h, w, 3), gray_val, dtype=np.uint8)
            # Добавляем небольшой шум
            noise = np.random.normal(0, 5.0, size=(h, w, 3))
            bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # 25% - цветной фон (голубой/бежевый/зеленоватый)
        elif r < 0.75:
            # Случайный светлый оттенок
            base_r = np.random.randint(200, 245)
            base_g = np.random.randint(200, 245)
            base_b = np.random.randint(200, 245)
            bg = np.full((h, w, 3), [base_r, base_g, base_b], dtype=np.uint8)
            noise = np.random.normal(0, 8.0, size=(h, w, 3))
            bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # 25% - градиент (светлый в темный или наоборот)
        else:
            top_val = np.random.randint(180, 255)
            bottom_val = np.random.randint(120, 220)
            if np.random.rand() > 0.5:
                top_val, bottom_val = bottom_val, top_val
            gradient = np.linspace(top_val / 255.0, bottom_val / 255.0, h).reshape(-1, 1, 1)
            bg = (np.full((h, w, 3), 200, dtype=np.float32) * gradient).astype(np.uint8)
        
        bg = apply_synthetic_shadow(bg, alpha_float)
        bg_mean = float(bg.mean())
        fg = apply_white_on_white_fg(image, bg_mean)
        
        img_float = fg.astype(np.float32)
        bg_float = bg.astype(np.float32)
        out = img_float * alpha_float + bg_float * (1.0 - alpha_float)
        return np.clip(out, 0, 255).astype(np.uint8)
    
    def __getitem__(self, idx):
        import pickle
        self._open_lmdb()
        
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(str(idx).encode()))
        
        image = data['image']
        alpha = data['alpha']
        
        if self.use_bg_augmentation:
            image = self.composite_with_white_bg(image, alpha)
        
        if self.transform:
            augmented = self.transform(image=image, mask=alpha)
            image = augmented['image']
            alpha = augmented['mask']
        
        alpha = alpha.float() / 255.0
        return {'image': image, 'mask': alpha.unsqueeze(0)}

# =====================
# LOSS
# =====================
class MattingLoss(nn.Module):
    def __init__(self, use_edge_weighted=True, edge_weight=2.0):
        super().__init__()
        self.use_edge_weighted = use_edge_weighted
        self.edge_weight = edge_weight
    
    def compute_edge_weight(self, mask):
        kernel = torch.ones(1, 1, 3, 3, device=mask.device)
        edges = F.conv2d(mask, kernel, padding=1)
        edges = ((edges > 0) & (edges < 9)).float()
        weight_map = torch.ones_like(mask)
        weight_map = weight_map + edges * (self.edge_weight - 1.0)
        return weight_map
    
    def forward(self, pred, target):
        if self.use_edge_weighted:
            edge_weight_map = self.compute_edge_weight(target)
            mse_loss = ((pred - target) ** 2 * edge_weight_map).mean()
        else:
            mse_loss = F.mse_loss(pred, target)
        
        # Более мягкий штраф на детали, чтобы не усиливать текстуры/тени
        l1_loss = F.l1_loss(pred, target)
        
        total_loss = 1.0 * mse_loss + 0.1 * l1_loss
        
        return total_loss, {'mse': mse_loss.item(), 'l1': l1_loss.item()}

# =====================
# METRICS
# =====================
def compute_sad(pred, target):
    return torch.abs(pred - target).sum() / pred.numel()

def compute_gradient_loss(pred, target):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    
    pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
    target_grad_x = F.conv2d(target, sobel_x, padding=1)
    target_grad_y = F.conv2d(target, sobel_y, padding=1)
    
    grad_loss = torch.abs(pred_grad_x - target_grad_x).mean() + torch.abs(pred_grad_y - target_grad_y).mean()
    return grad_loss

def extract_prediction(output):
    def find_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj
        elif isinstance(obj, (list, tuple)) and len(obj) > 0:
            result = find_tensor(obj[-1])
            if result is not None:
                return result
            result = find_tensor(obj[0])
            if result is not None:
                return result
            for item in obj:
                result = find_tensor(item)
                if result is not None:
                    return result
        return None
    
    tensor = find_tensor(output)
    if tensor is None:
        raise ValueError(f"Не удалось найти тензор в выходе модели! Тип: {type(output)}")
    return tensor

# =====================
# TRAINING FUNCTIONS
# =====================
def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    accumulation_steps,
    save_every_iters=None,
    save_callback=None,
    epoch_idx=None,
    start_batch_idx: int = 0,
):
    model.train()
    model.apply(freeze_batchnorm)
    total_loss = 0
    total_mse = 0
    total_sad = 0
    total_grad = 0
    n_batches = 0
    optimizer.zero_grad()

    window_loss = 0.0
    window_sad = 0.0
    window_grad = 0.0
    window_n = 0

    pbar = tqdm(loader, desc="Training", disable=False, file=sys.stdout, dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar):
        if start_batch_idx and batch_idx < start_batch_idx:
            continue
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Size check
        if tuple(images.shape[-2:]) != (EXPECTED_SIZE, EXPECTED_SIZE):
            raise ValueError(f"Unexpected image size: {tuple(images.shape[-2:])}, expected {(EXPECTED_SIZE, EXPECTED_SIZE)}")
        
        with autocast():
            output = model(images)
            pred_logits = extract_prediction(output)
            pred = pred_logits.sigmoid()
            
            loss, loss_dict = criterion(pred, masks)
            loss = loss / accumulation_steps
            
            with torch.no_grad():
                sad = compute_sad(pred, masks)
                grad_loss = compute_gradient_loss(pred, masks)
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        total_mse += loss_dict['mse']
        total_sad += sad.item()
        total_grad += grad_loss.item()
        n_batches += 1
        
        pbar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}", 'sad': f"{sad.item():.4f}"})

        window_loss += loss.item() * accumulation_steps
        window_sad += sad.item()
        window_grad += grad_loss.item()
        window_n += 1
        
        if TRAIN_LOG_EVERY_ITERS and (window_n >= TRAIN_LOG_EVERY_ITERS):
            msg = (
                f"[train] epoch={epoch_idx+1 if epoch_idx is not None else '?'} "
                f"iter={batch_idx+1} "
                f"avg_loss={window_loss / window_n:.4f} "
                f"avg_sad={window_sad / window_n:.4f} "
                f"avg_grad={window_grad / window_n:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
            pbar.close()
            print("\n" + msg, flush=True)
            sys.stdout.flush()
            pbar = tqdm(loader, desc="Training", initial=batch_idx+1, total=len(loader), disable=False, file=sys.stdout, dynamic_ncols=True)
            window_loss = 0.0
            window_sad = 0.0
            window_grad = 0.0
            window_n = 0

        if save_every_iters is not None and save_callback is not None:
            if (batch_idx + 1) % save_every_iters == 0:
                try:
                    save_callback(batch_idx=batch_idx + 1, epoch=epoch_idx)
                except Exception as e:
                    print(f"Warning: Failed to save iter checkpoint: {e}")
    
    if (batch_idx + 1) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
    
    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'sad': total_sad / n_batches,
        'grad': total_grad / n_batches,
        'lr': optimizer.param_groups[0]['lr']
    }

@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_mse = 0
    total_sad = 0
    total_grad = 0
    n_batches = 0
    
    for batch in tqdm(loader, desc="Validation"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        if tuple(images.shape[-2:]) != (EXPECTED_SIZE, EXPECTED_SIZE):
            raise ValueError(f"Unexpected val image size: {tuple(images.shape[-2:])}")
        
        output = model(images)
        pred_logits = extract_prediction(output)
        pred = pred_logits.sigmoid()
        
        loss, loss_dict = criterion(pred, masks)
        sad = compute_sad(pred, masks)
        grad_loss = compute_gradient_loss(pred, masks)
        
        total_loss += loss.item()
        total_mse += loss_dict['mse']
        total_sad += sad.item()
        total_grad += grad_loss.item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'sad': total_sad / n_batches,
        'grad': total_grad / n_batches
    }

def freeze_encoder(model):
    for name, param in model.named_parameters():
        if 'bb' in name:
            param.requires_grad = False

def unfreeze_encoder(model):
    for name, param in model.named_parameters():
        if 'bb' in name:
            param.requires_grad = True

def freeze_batchnorm(module):
    """Заморозить все BatchNorm слои в eval режим (для batch_size=1)"""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

def make_loaders(train_dataset, val_dataset, batch_size, num_workers=0):
    """Создать train/val DataLoader под нужный batch_size"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )
    return train_loader, val_loader

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Phase 1 (epoch 0-{PHASE1_EPOCHS-1}): batch={PHASE1_BATCH_SIZE}, accum={PHASE1_ACCUM_STEPS}")
    print(f"Phase 2 (epoch {PHASE1_EPOCHS}+): batch={PHASE2_BATCH_SIZE}, accum={PHASE2_ACCUM_STEPS}")
    print(f"NUM_WORKERS: {NUM_WORKERS}")
    print(f"Checkpoints dir: {CHECKPOINT_DIR}")
    
    # Transforms (REDUCED intensity 2-3x)
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.03, rotate_limit=5, p=0.2),
        A.ColorJitter(brightness=0.06, contrast=0.06, saturation=0.06, hue=0.02, p=0.15),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Datasets (REDUCED p_white_bg 2.5x)
    if USE_LMDB:
        train_dataset = UnifiedLMDBDataset(TRAIN_LMDB, transform=train_transform, size=1024, use_bg_augmentation=True, p_white_bg=0.10)
        val_dataset = UnifiedLMDBDataset(VAL_LMDB, transform=val_transform, size=1024, use_bg_augmentation=False)
    else:
        train_dataset = UnifiedMattingDataset(TRAIN_IMG_DIR, TRAIN_ALPHA_DIR, transform=train_transform, size=1024, use_bg_augmentation=True, p_white_bg=0.10)
        val_dataset = UnifiedMattingDataset(VAL_IMG_DIR, VAL_ALPHA_DIR, transform=val_transform, size=1024, use_bg_augmentation=False)
    
    # DataLoaders - Фаза 1 (batch=2)
    current_batch = PHASE1_BATCH_SIZE
    current_accum = PHASE1_ACCUM_STEPS
    
    train_loader, val_loader = make_loaders(train_dataset, val_dataset, current_batch, NUM_WORKERS)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model
    model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
    model = model.to(DEVICE)
    freeze_encoder(model)
    
    # Loss, Optimizer, Scheduler
    criterion = MattingLoss(use_edge_weighted=USE_EDGE_WEIGHTED_LOSS, edge_weight=EDGE_WEIGHT_FACTOR)
    # LR = 1e-5: VERY conservative for frozen encoder fine-tuning
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=0.0)
    
    # Рассчитываем total_steps с учётом обеих фаз
    phase1_steps_per_epoch = max(1, len(train_loader) // PHASE1_ACCUM_STEPS)
    # После переключения будет в 2 раза меньше батчей, но в 2 раза больше accum
    phase2_loader_len = len(train_loader) // 2 if current_batch == 2 else len(train_loader)
    phase2_steps_per_epoch = max(1, phase2_loader_len // PHASE2_ACCUM_STEPS)
    
    total_steps = phase1_steps_per_epoch * PHASE1_EPOCHS + phase2_steps_per_epoch * (EPOCHS - PHASE1_EPOCHS)
    WARMUP_STEPS = 500
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)
    scaler = GradScaler()
    
    print(f"Total optimizer steps: {total_steps}")
    print(f"Phase1 steps/epoch: {phase1_steps_per_epoch}, Phase2 steps/epoch: {phase2_steps_per_epoch}")
    
    # Define checkpoint paths before resume
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_NAME = f"birefnet_{datetime.now().strftime('%Y%m%d_%H%M')}"
    BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / f"{RUN_NAME}_best.pth"
    LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / f"{RUN_NAME}_last.pth"
    ITER_CHECKPOINT_PATH = CHECKPOINT_DIR / f"{RUN_NAME}_iter.pth"
    
    # Initialize history and best_metric before resume
    history = {'train_loss': [], 'train_sad': [], 'val_loss': [], 'val_sad': []}
    best_metric = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    # Resume (optional)
    start_epoch = 0
    start_batch_idx = 0
    if RESUME_CHECKPOINT_PATH is not None:
        resume_path = Path(RESUME_CHECKPOINT_PATH)
        if resume_path.exists():
            print(f"\nResuming from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location='cpu')
            
            # Model/optim/scheduler/scaler
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'], strict=False)
                if 'optimizer_state_dict' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if 'scheduler_state_dict' in ckpt:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                if 'scaler_state_dict' in ckpt:
                    scaler.load_state_dict(ckpt['scaler_state_dict'])
            
            # Training state
            if 'history' in ckpt and isinstance(ckpt['history'], dict):
                history = ckpt['history']
            else:
                history = {'train_loss': [], 'train_sad': [], 'val_loss': [], 'val_sad': []}
            
            if 'best_metric' in ckpt:
                best_metric = ckpt['best_metric']
            else:
                best_metric = float('inf')
            
            # Restore phase params if present
            if 'current_batch' in ckpt:
                current_batch = ckpt['current_batch']
            if 'current_accum' in ckpt:
                current_accum = ckpt['current_accum']

            # Recreate loaders to match resumed batch size
            train_loader, val_loader = make_loaders(train_dataset, val_dataset, current_batch, NUM_WORKERS)
            
            # Restore start_epoch and start_batch_idx
            if 'epoch' in ckpt and ckpt['epoch'] is not None:
                start_epoch = max(0, int(ckpt['epoch']) - 1)
            if 'batch_idx' in ckpt and ckpt['batch_idx'] is not None:
                start_batch_idx = int(ckpt['batch_idx'])
            
            print(f"   Loaded: start_epoch={start_epoch+1}, start_batch_idx={start_batch_idx}")
            print(f"   Loaded: current_batch={current_batch}, current_accum={current_accum}")
        else:
            print(f"Warning: RESUME_CHECKPOINT_PATH not found: {resume_path}")
    
    print(f"Early stopping metric: {EARLY_STOPPING_METRIC}, Patience: {PATIENCE}")
    
    for epoch in range(start_epoch, EPOCHS):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{EPOCHS}\n{'='*60}")
        # ENCODER UNFREEZING DISABLED - keep frozen for stable fine-tuning
        if False and epoch == PHASE1_EPOCHS:  # Disabled
            print("\nUnfreezing encoder...")
            unfreeze_encoder(model)
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
            model.apply(freeze_batchnorm)
        
        def save_iter_checkpoint(batch_idx, epoch):
            atomic_torch_save({
                'epoch': (epoch + 1) if epoch is not None else None,
                'batch_idx': batch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'current_batch': current_batch,
                'current_accum': current_accum,
                'best_metric': best_metric,
                'history': history
            }, ITER_CHECKPOINT_PATH)

        epoch_start_batch = start_batch_idx if epoch == start_epoch else 0
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            DEVICE,
            current_accum,
            save_every_iters=SAVE_EVERY_ITERS,
            save_callback=save_iter_checkpoint,
            epoch_idx=epoch,
            start_batch_idx=epoch_start_batch,
        )
        val_metrics = val_epoch(model, val_loader, criterion, DEVICE)
        
        # КРИТИЧНО: Агрессивная очистка памяти после КАЖДОЙ эпохи
        # Пересоздаем DataLoaders чтобы освободить накопленную память
        del train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        # Пересоздаем loaders для следующей эпохи
        train_loader, val_loader = make_loaders(train_dataset, val_dataset, current_batch, NUM_WORKERS)
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_sad'].append(train_metrics['sad'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_sad'].append(val_metrics['sad'])
        
        print(f"\nTrain: loss={train_metrics['loss']:.4f}, sad={train_metrics['sad']:.4f}, mse={train_metrics['mse']:.4f}")
        print(f"Val:   loss={val_metrics['loss']:.4f}, sad={val_metrics['sad']:.4f}, mse={val_metrics['mse']:.4f}")
        print(f"LR: {train_metrics['lr']:.2e}")
        
        current_metric = val_metrics[EARLY_STOPPING_METRIC.replace('val_', '')]
        
        # Сохранение last checkpoint после каждой эпохи (atomic save)
        try:
            atomic_torch_save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'current_batch': current_batch,
                'current_accum': current_accum,
                'best_metric': best_metric,
                'history': history
            }, LAST_CHECKPOINT_PATH)
        except Exception as e:
            print(f"Warning: Failed to save last checkpoint: {e}")
        
        if current_metric < best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            patience_counter = 0
            
            try:
                atomic_torch_save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_metric': best_metric,
                    'history': history
                }, BEST_CHECKPOINT_PATH)
            except Exception as e:
                print(f"Warning: Failed to save best checkpoint: {e}")
            else:
                print(f"Best model saved! {EARLY_STOPPING_METRIC}={best_metric:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*60}\nTraining completed!\n   Best epoch: {best_epoch}\n   Best {EARLY_STOPPING_METRIC}: {best_metric:.4f}\n{'='*60}")
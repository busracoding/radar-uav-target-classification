
import os
import glob
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt



def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#the pad crop operations
def pad_or_crop_2d(x: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = x.shape

    if h > target_h:
        s = (h - target_h) // 2
        x = x[s:s + target_h, :]
    if w > target_w:
        s = (w - target_w) // 2
        x = x[:, s:s + target_w]

    h, w = x.shape
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)

    if pad_h > 0 or pad_w > 0:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        x = np.pad(x, ((top, bottom), (left, right)), mode="constant", constant_values=0.0)

    return x.astype(np.float32)


#the light aıgmentions
def augment_radar_map(x: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    if random.random() > p:
        return x

    if random.random() < 0.5:
        scale = 1.0 + (random.random() - 0.5) * 0.15
        x = x * scale

    if random.random() < 0.7:
        std = 0.02 * x.std().clamp_min(1e-6)
        x = x + torch.randn_like(x) * std

    if random.random() < 0.5:
        shift_w = random.randint(-3, 3)
        x = torch.roll(x, shifts=shift_w, dims=2)
    if random.random() < 0.3:
        shift_h = random.randint(-2, 2)
        x = torch.roll(x, shifts=shift_h, dims=1)

    return x


#the dataset
class RadarCSVDataset(Dataset):
    def __init__(self, file_paths: List[str], labels: List[int],
                 target_hw: Tuple[int, int], augment: bool = False):
        self.file_paths = file_paths
        self.labels = labels
        self.th, self.tw = target_hw
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        y = int(self.labels[idx])

        arr = pd.read_csv(path, header=None).values.astype(np.float32)
        arr = pad_or_crop_2d(arr, self.th, self.tw)

        # per-sample z-score
        m = float(arr.mean())
        s = float(arr.std())
        if s < 1e-6:
            s = 1e-6
        arr = (arr - m) / s

        x = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
        if self.augment:
            x = augment_radar_map(x)

        return x, torch.tensor(y, dtype=torch.long)

#the dataset scanning
def scan_dataset(data_root: str,
                 allowed_classes: Tuple[str, ...] = ("Cars", "Drones", "People")) -> Tuple[List[str], List[int], Dict[int, str]]:
    
    data_root = os.path.abspath(data_root)
    subs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    lower_map = {d.lower(): d for d in subs}

    chosen = []
    for c in allowed_classes:
        if c.lower() in lower_map:
            chosen.append(lower_map[c.lower()])

    if len(chosen) < 2:
        raise ValueError(f"Could not find enough class folders inside {data_root}. Found subdirs: {subs}")

    chosen = sorted(chosen)
    class_to_id = {c: i for i, c in enumerate(chosen)}
    id_to_class = {i: c for c, i in class_to_id.items()}

    file_paths, labels = [], []
    for c in chosen:
        dpath = os.path.join(data_root, c)
        csvs = glob.glob(os.path.join(dpath, "**", "*.csv"), recursive=True)
        csvs = sorted(csvs)
        if len(csvs) == 0:
            raise ValueError(f"Class folder '{c}' contains no CSVs.")
        file_paths.extend(csvs)
        labels.extend([class_to_id[c]] * len(csvs))

    return file_paths, labels, id_to_class


def select_balanced_subset(file_paths: List[str], labels: List[int],
                           limit_total: int, seed: int) -> Tuple[List[str], List[int]]:
    if limit_total <= 0 or limit_total >= len(file_paths):
        return file_paths, labels

    rng = np.random.default_rng(seed)
    file_paths = np.array(file_paths)
    labels = np.array(labels)

    classes = np.unique(labels)
    k = len(classes)
    per_class = limit_total // k

    chosen_idx = []
    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        take = min(per_class, len(idx_c))
        chosen_idx.extend(idx_c[:take].tolist())

    remaining = limit_total - len(chosen_idx)
    if remaining > 0:
        all_idx = np.arange(len(labels))
        mask = np.ones(len(labels), dtype=bool)
        mask[chosen_idx] = False
        rest = all_idx[mask]
        rng.shuffle(rest)
        chosen_idx.extend(rest[:remaining].tolist())

    rng.shuffle(chosen_idx)
    chosen_idx = chosen_idx[:limit_total]
    return file_paths[chosen_idx].tolist(), labels[chosen_idx].tolist()


#CNN +Transformer
class PatchEmbedCNN(nn.Module):
    """
    Input:  [B,1,H,W]
    Output: tokens [B, N, D]
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        # Downsample a bit to reduce token count
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/2 W/2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/4 W/4
            nn.Conv2d(64, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True),
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        f = self.conv(x)            # [B,D,H',W']
        B, D, H, W = f.shape
        tokens = f.flatten(2).transpose(1, 2).contiguous()  # [B, N=H'*W', D]
        return tokens


class RadarTransformerClassifier(nn.Module):
    def __init__(self, num_classes: int, embed_dim=256, nhead=8, depth=4, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.patch = PatchEmbedCNN(embed_dim=embed_dim)

        # CLS token
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = None  

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=mlp_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def _init_pos(self, n_tokens: int, device):
        # pos embedding for (CLS + tokens)
        self.pos = nn.Parameter(torch.zeros(1, 1 + n_tokens, self.cls.shape[-1], device=device))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        tok = self.patch(x)                     # [B,N,D]
        B, N, D = tok.shape
        if self.pos is None or self.pos.shape[1] != (1 + N):
            self._init_pos(N, x.device)

        cls = self.cls.expand(B, -1, -1)        # [B,1,D]
        z = torch.cat([cls, tok], dim=1)        # [B,1+N,D]
        z = z + self.pos
        z = self.encoder(z)
        z = self.norm(z[:, 0, :])               # CLS
        z = self.dropout(z)
        return self.head(z)


#the early stopping
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0

    def step(self, val_loss: float) -> bool:
        if self.best is None or val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


#train and evaluation
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def train_one_epoch(model, loader, optimizer, device, label_smoothing: float, grad_clip: float):
    model.train()
    losses, accs = [], []
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())
        accs.append(accuracy_from_logits(logits, y))
    return float(np.mean(losses)), float(np.mean(accs))


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    losses, accs = [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        losses.append(loss.item())
        accs.append(accuracy_from_logits(logits, y))
    return float(np.mean(losses)), float(np.mean(accs))


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    all_preds, all_true = [], []
    for x, y in tqdm(loader, desc="test", leave=False):
        x = x.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_true.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_true)


def plot_curves(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_curve.png"), dpi=200)
    plt.close()


def plot_confusion(cm, class_names, out_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    thresh = cm.max() * 0.6 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@dataclass
class Config:
    data_root: str
    img_h: int
    img_w: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    seed: int
    augment: bool
    limit_total: int
    num_workers: int
    out_dir: str
    patience: int
    min_delta: float
    label_smoothing: float
    grad_clip: float

    # transformer params
    embed_dim: int
    nhead: int
    depth: int
    mlp_dim: int
    dropout: float


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--img_h", type=int, default=128)
    parser.add_argument("--img_w", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", type=int, default=1)
    parser.add_argument("--limit_total", type=int, default=10000, help="0 = all")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="runs/task2_cnn_transformer")

    # convergence
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-3)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # transformer
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    cfg = Config(
        data_root=args.data_root,
        img_h=args.img_h,
        img_w=args.img_w,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        augment=bool(args.augment),
        limit_total=args.limit_total,
        num_workers=args.num_workers,
        out_dir=args.out_dir,
        patience=args.patience,
        min_delta=args.min_delta,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        depth=args.depth,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout
    )

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    file_paths, labels, id_to_class = scan_dataset(cfg.data_root, allowed_classes=("Cars", "Drones", "People"))
    print(f"[INFO] total CSV found (3-class only) = {len(file_paths)}")
    print(f"[INFO] classes = {id_to_class}")

    file_paths, labels = select_balanced_subset(file_paths, labels, cfg.limit_total, cfg.seed)
    print(f"[INFO] using subset = {len(file_paths)} (unique, balanced)")

    X = np.array(file_paths)
    y = np.array(labels)
    num_classes = len(id_to_class)
    class_names = [id_to_class[i] for i in range(num_classes)]

    # split: 70/15/15
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=cfg.seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=cfg.seed, stratify=y_tmp)

    train_ds = RadarCSVDataset(list(X_train), list(y_train), (cfg.img_h, cfg.img_w), augment=cfg.augment)
    val_ds   = RadarCSVDataset(list(X_val), list(y_val), (cfg.img_h, cfg.img_w), augment=False)
    test_ds  = RadarCSVDataset(list(X_test), list(y_test), (cfg.img_h, cfg.img_w), augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = RadarTransformerClassifier(
        num_classes=num_classes,
        embed_dim=cfg.embed_dim,
        nhead=cfg.nhead,
        depth=cfg.depth,
        mlp_dim=cfg.mlp_dim,
        dropout=cfg.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, threshold=1e-3, verbose=True
    )
    early = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)

    os.makedirs(cfg.out_dir, exist_ok=True)
    best_val_loss = float("inf")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    for ep in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, cfg.label_smoothing, cfg.grad_clip)
        va_loss, va_acc = eval_one_epoch(model, val_loader, device)

        scheduler.step(va_loss)
        lr_now = float(optimizer.param_groups[0]["lr"])

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(lr_now)

        print(f"Epoch {ep:02d}/{cfg.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | lr {lr_now:.2e}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save({"model_state": model.state_dict(), "id_to_class": id_to_class},
                       os.path.join(cfg.out_dir, "best.pt"))

        if early.step(va_loss):
            print(f"[INFO] Early stopping at epoch {ep}. Best val_loss={early.best:.4f}")
            break

    plot_curves(history, cfg.out_dir)

    # test best
    ckpt = torch.load(os.path.join(cfg.out_dir, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    preds, ytrue = predict_all(model, test_loader, device)
    print("\n[TEST] Classification report:")
    print(classification_report(ytrue, preds, target_names=class_names, digits=4, zero_division=0))

    cm = confusion_matrix(ytrue, preds)
    print("[TEST] Confusion matrix:\n", cm)
    plot_confusion(cm, class_names, os.path.join(cfg.out_dir, "confusion_matrix.png"))

    print(f"\n[INFO] Saved to: {cfg.out_dir}")
    print(" - best.pt")
    print(" - loss_curve.png")
    print(" - acc_curve.png")
    print(" - confusion_matrix.png")


if __name__ == "__main__":
    main()
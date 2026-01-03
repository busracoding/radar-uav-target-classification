
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

        m = float(arr.mean())
        s = float(arr.std())
        if s < 1e-6:
            s = 1e-6
        arr = (arr - m) / s

        x = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
        if self.augment:
            x = augment_radar_map(x)

        return x, torch.tensor(y, dtype=torch.long)



def scan_dataset(data_root: str,
                 allowed_classes: Tuple[str, ...] = ("Cars", "Drones", "People")) -> Tuple[List[str], List[int], Dict[int, str]]:

    data_root = os.path.abspath(data_root)
    subs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

    # map lower->real folder name
    lower_map = {d.lower(): d for d in subs}

    chosen = []
    for c in allowed_classes:
        if c.lower() in lower_map:
            chosen.append(lower_map[c.lower()])

    if len(chosen) < 2:
        raise ValueError(f"Could not find enough class folders inside {data_root}. Found subdirs: {subs}")

    chosen = sorted(chosen)  # stable label ordering

    class_to_id = {c: i for i, c in enumerate(chosen)}
    id_to_class = {i: c for c, i in class_to_id.items()}

    file_paths, labels = [], []
    for c in chosen:
        dpath = os.path.join(data_root, c)
        csvs = glob.glob(os.path.join(dpath, "**", "*.csv"), recursive=True)
        csvs = sorted(csvs)
        if len(csvs) == 0:
            raise ValueError(f"Class folder '{c}' contains no CSVs (recursive).")
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



class CNNBackbone(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block3(self.block2(self.block1(x)))


class CNNBiLSTMClassifier(nn.Module):
    def __init__(self, num_classes: int, base=32, hidden=256, layers=1, dropout=0.25):
        super().__init__()
        self.backbone = CNNBackbone(in_ch=1, base=base)
        self.hidden = hidden
        self.layers = layers
        self.lstm = None
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes)
        )
        self._inited = False

    def _init_lstm(self, input_size: int, device: torch.device):
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden,
            num_layers=self.layers,
            batch_first=True,
            bidirectional=True
        ).to(device)
        self._inited = True

    def forward(self, x):
        feat = self.backbone(x)  # [B,C,H',W']
        B, C, Hp, Wp = feat.shape
        seq = feat.permute(0, 3, 1, 2).contiguous()  # [B,W',C,H']
        seq = seq.view(B, Wp, C * Hp)                # [B,W',C*H']
        if not self._inited:
            self._init_lstm(input_size=C * Hp, device=x.device)
        out, _ = self.lstm(seq)
        out_last = out[:, -1, :]
        return self.head(out_last)



class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0

    def step(self, metric_value: float) -> bool:
        """
        Returns True if should stop.
        We assume we minimize metric_value (e.g., val_loss).
        """
        if self.best is None or metric_value < self.best - self.min_delta:
            self.best = metric_value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience



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
    parser.add_argument("--out_dir", type=str, default="runs/task1_cnn_bilstm")

    # convergence / stability knobs
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience on val_loss")
    parser.add_argument("--min_delta", type=float, default=1e-3, help="minimum val_loss improvement to reset patience")
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)

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
        grad_clip=args.grad_clip
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

    # split: 70/15/15 stratified
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=cfg.seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=cfg.seed, stratify=y_tmp)

    train_ds = RadarCSVDataset(list(X_train), list(y_train), (cfg.img_h, cfg.img_w), augment=cfg.augment)
    val_ds   = RadarCSVDataset(list(X_val), list(y_val), (cfg.img_h, cfg.img_w), augment=False)
    test_ds  = RadarCSVDataset(list(X_test), list(y_test), (cfg.img_h, cfg.img_w), augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = CNNBiLSTMClassifier(num_classes=num_classes, base=32, hidden=256, layers=1, dropout=0.25).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # LR scheduler for stable convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, threshold=1e-3, verbose=True
    )

    os.makedirs(cfg.out_dir, exist_ok=True)

    best_val_loss = float("inf")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    early = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)

    for ep in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            label_smoothing=cfg.label_smoothing, grad_clip=cfg.grad_clip
        )
        va_loss, va_acc = eval_one_epoch(model, val_loader, device)

        # scheduler step on val loss
        scheduler.step(va_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(current_lr)

        print(f"Epoch {ep:02d}/{cfg.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | lr {current_lr:.2e}")

        # save best by val loss
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save({"model_state": model.state_dict(), "id_to_class": id_to_class},
                       os.path.join(cfg.out_dir, "best.pt"))

        # early stopping check
        if early.step(va_loss):
            print(f"[INFO] Early stopping triggered at epoch {ep}. Best val_loss={early.best:.4f}")
            break

    plot_curves(history, cfg.out_dir)

    # test best model
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



import os, glob, random, argparse, time
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

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
    # amplitude jitter
    if random.random() < 0.5:
        scale = 1.0 + (random.random() - 0.5) * 0.20
        x = x * scale
    # gaussian noise
    if random.random() < 0.8:
        std = 0.03 * x.std().clamp_min(1e-6)
        x = x + torch.randn_like(x) * std
    # small shifts
    if random.random() < 0.6:
        x = torch.roll(x, shifts=random.randint(-4, 4), dims=2)
    if random.random() < 0.4:
        x = torch.roll(x, shifts=random.randint(-3, 3), dims=1)
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
    lower_map = {d.lower(): d for d in subs}

    chosen = []
    for c in allowed_classes:
        if c.lower() in lower_map:
            chosen.append(lower_map[c.lower()])
    if len(chosen) < 2:
        raise ValueError(f"Could not find enough class folders in {data_root}. Subdirs: {subs}")

    chosen = sorted(chosen)
    class_to_id = {c: i for i, c in enumerate(chosen)}
    id_to_class = {i: c for c, i in class_to_id.items()}

    file_paths, labels = [], []
    for c in chosen:
        csvs = glob.glob(os.path.join(data_root, c, "**", "*.csv"), recursive=True)
        csvs = sorted(csvs)
        file_paths.extend(csvs)
        labels.extend([class_to_id[c]] * len(csvs))

    return file_paths, labels, id_to_class


def select_balanced_subset(file_paths: List[str], labels: List[int], limit_total: int, seed: int):
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



class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, k, padding=p),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop),
        )
    def forward(self, x): return self.net(x)


class StrongCNNBackbone(nn.Module):
    def __init__(self, base=32, drop=0.05):
        super().__init__()
        self.b1 = ConvBlock(1, base, drop=drop)
        self.p1 = nn.MaxPool2d(2)
        self.b2 = ConvBlock(base, base*2, drop=drop)
        self.p2 = nn.MaxPool2d(2)
        self.b3 = ConvBlock(base*2, base*4, drop=drop)
        self.p3 = nn.MaxPool2d(2)
        self.b4 = ConvBlock(base*4, base*4, drop=drop)  # extra depth
    def forward(self, x):
        x = self.p1(self.b1(x))
        x = self.p2(self.b2(x))
        x = self.p3(self.b3(x))
        x = self.b4(x)
        return x  # [B,C,H',W']



class xLSTMBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.10):
        super().__init__()
        self.input_ln = nn.LayerNorm(input_size)
        self.h_ln = nn.LayerNorm(hidden_size)
        self.lstm = nn.LSTMCell(input_size, hidden_size)

        self.res_gate = nn.Sequential(nn.Linear(input_size + hidden_size, hidden_size), nn.Sigmoid())
        self.mem_gate = nn.Sequential(nn.Linear(input_size + hidden_size, hidden_size), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_t, state):
        h, c = state
        x = self.input_ln(x_t)
        h_n = self.h_ln(h)

        h_new, c_new = self.lstm(x, (h_n, c))

        g = self.res_gate(torch.cat([x, h_n], dim=1))
        h_out = g * h_new + (1.0 - g) * h
        h_out = self.dropout(h_out)

        m = self.mem_gate(torch.cat([x, h_n], dim=1))
        c_hat = torch.tanh(c_new)
        c_out = m * c_new + (1.0 - m) * c_hat
        return h_out, c_out


class xLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.10):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList([
            xLSTMBlock(input_size if i == 0 else hidden_size, hidden_size, dropout=dropout)
            for i in range(num_layers)
        ])

    def forward(self, x):  # x: [B,T,F]
        B, T, _ = x.shape
        device = x.device
        hs = [torch.zeros(B, self.hidden_size, device=device) for _ in self.layers]
        cs = [torch.zeros(B, self.hidden_size, device=device) for _ in self.layers]

        for t in range(T):
            inp = x[:, t, :]
            for i, layer in enumerate(self.layers):
                hs[i], cs[i] = layer(inp, (hs[i], cs[i]))
                inp = hs[i]
        return hs[-1]


class CNNxLSTMClassifier(nn.Module):
    def __init__(self, num_classes: int, base=32, hidden=256, layers=2, dropout=0.20):
        super().__init__()
        self.backbone = StrongCNNBackbone(base=base, drop=0.05)
        self.hidden = hidden
        self.layers = layers
        self.seq_proj = None
        self.rnn = None
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
        self._inited = False

    def _init_modules(self, input_size: int, device):
        self.seq_proj = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, self.hidden),
            nn.GELU(),
            nn.Dropout(0.10),
        ).to(device)
        self.rnn = xLSTM(self.hidden, self.hidden, num_layers=self.layers, dropout=0.10).to(device)
        self._inited = True

    def forward(self, x):
        feat = self.backbone(x)         
        B, C, Hp, Wp = feat.shape
        seq = feat.permute(0, 3, 1, 2).contiguous().view(B, Wp, C * Hp) 
        if not self._inited:
            self._init_modules(C * Hp, x.device)
        seq = self.seq_proj(seq)
        h_last = self.rnn(seq)
        return self.head(h_last)


class EarlyStoppingF1:
    def __init__(self, patience: int = 12, min_delta: float = 1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad = 0

    def step(self, val_f1: float) -> bool:
        if self.best is None or val_f1 > self.best + self.min_delta:
            self.best = val_f1
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience


@torch.no_grad()
def eval_metrics(model, loader, device):
    model.eval()
    all_preds, all_true, losses = [], [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_true.append(y.cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return float(np.mean(losses)), float(acc), float(macro_f1)


def train_one_epoch(model, loader, optimizer, device, label_smoothing, grad_clip):
    model.train()
    losses = []
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
    return float(np.mean(losses))


def plot_curves(hist, out_dir):
    e = np.arange(1, len(hist["train_loss"]) + 1)

    plt.figure()
    plt.plot(e, hist["train_loss"], label="train_loss")
    plt.plot(e, hist["val_loss"], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=220); plt.close()

    plt.figure()
    plt.plot(e, hist["val_acc"], label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_curve.png"), dpi=220); plt.close()

    plt.figure()
    plt.plot(e, hist["val_f1"], label="val_macro_f1")
    plt.xlabel("epoch"); plt.ylabel("macro-F1"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "f1_curve.png"), dpi=220); plt.close()


def plot_confusion(cm, names, out_path, normalize=False):
    if normalize:
        cm = cm.astype(np.float32)
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar()
    ticks = np.arange(len(names))
    plt.xticks(ticks, names, rotation=45, ha="right")
    plt.yticks(ticks, names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            plt.text(j, i, txt, ha="center", va="center", color="white" if val > (cm.max()*0.6) else "black")
    plt.ylabel("True"); plt.xlabel("Pred"); plt.tight_layout()
    plt.savefig(out_path, dpi=220); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--limit_total", type=int, default=12000)
    ap.add_argument("--img_h", type=int, default=128)
    ap.add_argument("--img_w", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augment", type=int, default=1)

    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.20)

    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--out_dir", type=str, default="runs/task3_cnn_xlstm_v2")

    ap.add_argument("--warmup_epochs", type=int, default=5)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    file_paths, labels, id_to_class = scan_dataset(args.data_root, ("Cars", "Drones", "People"))
    print(f"[INFO] total CSV found (3-class only) = {len(file_paths)}")
    print(f"[INFO] classes = {id_to_class}")

    file_paths, labels = select_balanced_subset(file_paths, labels, args.limit_total, args.seed)
    print(f"[INFO] using subset = {len(file_paths)} (unique, balanced)")

    X = np.array(file_paths)
    y = np.array(labels)
    class_names = [id_to_class[i] for i in range(len(id_to_class))]

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=args.seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=args.seed, stratify=y_tmp)

    train_ds = RadarCSVDataset(list(X_train), list(y_train), (args.img_h, args.img_w), augment=bool(args.augment))
    val_ds   = RadarCSVDataset(list(X_val), list(y_val), (args.img_h, args.img_w), augment=False)
    test_ds  = RadarCSVDataset(list(X_test), list(y_test), (args.img_h, args.img_w), augment=False)

 
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / (class_counts + 1e-9)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.double),
                                    num_samples=len(sample_weights),
                                    replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = CNNxLSTMClassifier(num_classes=len(id_to_class),
                              base=args.base, hidden=args.hidden,
                              layers=args.layers, dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

  
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs))

    def set_lr(lr):
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    os.makedirs(args.out_dir, exist_ok=True)
    best_f1 = -1.0
    stopper = EarlyStoppingF1(patience=args.patience, min_delta=1e-3)

    hist = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "lr": []}

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        # warmup
        if ep <= args.warmup_epochs:
            warm_lr = args.lr * (ep / max(1, args.warmup_epochs))
            set_lr(warm_lr)
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.label_smoothing, args.grad_clip)

        val_loss, val_acc, val_f1 = eval_metrics(model, val_loader, device)

        # cosine step after warmup
        if ep > args.warmup_epochs:
            cosine.step()

        lr_now = optimizer.param_groups[0]["lr"]
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)
        hist["val_f1"].append(val_f1)
        hist["lr"].append(lr_now)

        print(f"Epoch {ep:03d}/{args.epochs} | "
              f"train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f} | lr {lr_now:.2e}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"model_state": model.state_dict(), "id_to_class": id_to_class},
                       os.path.join(args.out_dir, "best.pt"))

        if stopper.step(val_f1):
            print(f"[INFO] Early stop (val macro-F1 not improving). Best F1={stopper.best:.4f}")
            break

    train_time = time.time() - t0
    plot_curves(hist, args.out_dir)

    ckpt = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    # Test
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for x, yb in tqdm(test_loader, desc="test", leave=False):
            x = x.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.append(pred)
            y_true.append(yb.numpy())
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    rep = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    plot_confusion(cm, class_names, os.path.join(args.out_dir, "confusion_matrix.png"), normalize=False)
    plot_confusion(cm, class_names, os.path.join(args.out_dir, "confusion_matrix_norm.png"), normalize=True)

    with open(os.path.join(args.out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"train_time_seconds: {train_time:.2f}\n")
        f.write(f"test_accuracy: {acc:.4f}\n")
        f.write(f"test_macro_f1: {macro_f1:.4f}\n\n")
        f.write(rep + "\n")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm) + "\n")

    print(f"\n[TEST] acc={acc:.4f} macroF1={macro_f1:.4f}")
    print("[INFO] Saved to:", args.out_dir)
    print(" - best.pt, metrics.txt")
    print(" - loss_curve.png, acc_curve.png, f1_curve.png")
    print(" - confusion_matrix.png, confusion_matrix_norm.png")


if __name__ == "__main__":
    main()

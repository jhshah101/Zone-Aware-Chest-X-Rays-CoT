import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .dataset import build_records, encode_and_split, LobeDataset, collate_fn
from .yolo_lobes import load_yolo
from .model import ZoneCoTClassifier

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_epoch(model, loader, criterion, optimizer=None, rationale_weight: float = 0.0, train: bool = True):
    model.train(train)
    device = next(model.parameters()).device
    total_loss, total_acc, n = 0.0, 0.0, 0

    for batch in tqdm(loader, desc=("train" if train else "val")):
        labels = torch.tensor([b["label"] for b in batch], device=device, dtype=torch.long)

        out = model(batch, num_lobes=5, rationale_targets=None)
        logits = out["logits"]

        loss = criterion(logits, labels)
        if out["rationale_loss"] is not None and rationale_weight > 0:
            loss = loss + rationale_weight * out["rationale_loss"]

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)
        total_acc += (preds == labels).sum().item()
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)

    return total_loss / max(n,1), total_acc / max(n,1)

def main(cfg_path: str = "configs/config.yaml"):
    cfg = load_config(cfg_path)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo = load_yolo(cfg.yolo_model_path)
    records = build_records(cfg.csv_path, cfg.dataset_dir, cfg.folder_map)
    train_recs, val_recs, le = encode_and_split(records, test_size=0.2, seed=cfg.seed)

    train_ds = LobeDataset(train_recs, yolo, cfg.conf_th, cfg.num_lobes)
    val_ds   = LobeDataset(val_recs, yolo, cfg.conf_th, cfg.num_lobes)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    num_classes = len(le.classes_)
    model = ZoneCoTClassifier(num_classes=num_classes, img_size=cfg.img_size,
                              hidden_dim=cfg.hidden_dim, use_rationale=cfg.use_rationale).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    os.makedirs(os.path.dirname(cfg.ckpt_path), exist_ok=True)

    best_val = 0.0
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, cfg.rationale_weight, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer=None, rationale_weight=cfg.rationale_weight, train=False)

        print(f"Epoch {epoch:02d} | Train loss {tr_loss:.4f} acc {tr_acc:.4f} | Val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val:
            best_val = va_acc
            torch.save({"state": model.state_dict(),
                        "classes": list(le.classes_)},
                       cfg.ckpt_path)
            print(f"✅ saved best -> {cfg.ckpt_path}")

    print("Best Val Acc:", best_val)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)

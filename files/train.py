"""
Train KeyGenius on the PIG dataset.

Run from the repo root:
    python train.py

Watch for:
  * match_rate climbing toward ~0.65-0.72 (real SOTA territory). If it pins at
    ~0.99, your split is leaking — check piece_level_split.
  * finger_dist staying spread across 1-5. If it collapses onto one finger,
    raise crf_weight or lower lr.
"""
from __future__ import annotations

import math
import os

import torch
from torch.utils.data import DataLoader

from config import CFG
from data.pig_parser import load_pieces
from data.dataset import (
    FingeringDataset, build_hand_sequences, collate, piece_level_split,
)
from model.architecture import KeyGenius
from evaluate import evaluate


def make_loaders():
    pieces = load_pieces(CFG.paths.pig_root, CFG.paths.fingering_glob)
    train_pieces, val_pieces = piece_level_split(
        pieces, CFG.train.val_fraction, CFG.train.seed)
    train_seqs = build_hand_sequences(train_pieces, CFG.train.max_seq_len)
    val_seqs = build_hand_sequences(val_pieces, CFG.train.max_seq_len)
    print(f"{len(pieces)} pieces -> {len(train_pieces)} train / {len(val_pieces)} val")
    print(f"{len(train_seqs)} train sequences / {len(val_seqs)} val sequences")

    train_ds = FingeringDataset(train_seqs, augment=True)
    val_ds = FingeringDataset(val_seqs, augment=False)
    train_loader = DataLoader(train_ds, batch_size=CFG.train.batch_size,
                              shuffle=True, collate_fn=collate, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=CFG.train.batch_size,
                            shuffle=False, collate_fn=collate)
    return train_loader, val_loader


def lr_lambda(epoch):
    warm = CFG.train.warmup_epochs
    if epoch < warm:
        return (epoch + 1) / warm
    progress = (epoch - warm) / max(1, CFG.train.epochs - warm)
    return 0.5 * (1 + math.cos(math.pi * progress))


def main():
    torch.manual_seed(CFG.train.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_loader, val_loader = make_loaders()
    model = KeyGenius().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=CFG.train.lr,
                            weight_decay=CFG.train.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    os.makedirs(CFG.paths.checkpoint_dir, exist_ok=True)
    best = 0.0
    stale = 0

    for epoch in range(CFG.train.epochs):
        model.train()
        running = {"loss": 0.0, "crf": 0.0, "focal": 0.0}
        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            opt.zero_grad()
            loss, parts = model.loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.train.grad_clip)
            opt.step()
            running["loss"] += loss.item()
            running["crf"] += parts["crf"]
            running["focal"] += parts["focal"]
        sched.step()

        nb = len(train_loader)
        metrics = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch:3d} | loss {running['loss']/nb:6.3f} "
            f"(crf {running['crf']/nb:5.2f} focal {running['focal']/nb:5.3f}) "
            f"| val match {metrics['match_rate']*100:5.2f}% "
            f"| dist {metrics['finger_dist_pct']}"
        )

        if metrics["match_rate"] > best:
            best = metrics["match_rate"]
            stale = 0
            torch.save({"model": model.state_dict(),
                        "match_rate": best, "epoch": epoch},
                       CFG.paths.best_ckpt)
            print(f"  -> saved best ({best*100:.2f}%) to {CFG.paths.best_ckpt}")
        else:
            stale += 1
            if stale >= CFG.train.early_stop_patience:
                print(f"early stop at epoch {epoch} (best {best*100:.2f}%)")
                break

    print(f"done. best val match rate: {best*100:.2f}%")


if __name__ == "__main__":
    main()

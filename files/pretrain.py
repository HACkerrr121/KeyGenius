"""
Phase 1: PRETRAIN on ThumbSet (large, noisy, partially labeled).

Trains the encoder / emission head with a MASKED focal loss — only the notes
that have a real 1..5 finger contribute; unlabeled (finger 0) notes pass through
for context but are ignored by the loss. The CRF transitions are intentionally
left for Phase 2 (fine-tuning on clean, fully labeled PIG).

Run:
    python pretrain.py --thumb thumb_files/FingeringFiles
Then fine-tune:
    python train.py --init checkpoints/pretrained.pt --lr 1e-4

Saves: checkpoints/pretrained.pt
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from config import CFG
from data.thumbset_parser import load_thumbset
from data.dataset import (
    FingeringDataset, build_hand_sequences, collate, piece_level_split,
)
from model.architecture import KeyGenius


@torch.no_grad()
def labeled_match_rate(model, loader, device):
    """Match rate computed ONLY over labeled notes (label >= 0)."""
    model.eval()
    correct = total = 0
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)
        paths = model.predict(batch["cont"], batch["pc"], batch["hand"], batch["mask"])
        labels, mask = batch["labels"], batch["mask"]
        for b, path in enumerate(paths):
            gold = labels[b][mask[b]].tolist()
            for pred, g in zip(path, gold):
                if g >= 0:                      # labeled only
                    correct += int(pred == g)
                    total += 1
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thumb", default="thumb_files/FingeringFiles",
                    help="ThumbSet FingeringFiles folder")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="checkpoints/pretrained.pt")
    args = ap.parse_args()

    torch.manual_seed(CFG.train.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    pieces = load_thumbset(args.thumb)
    total = sum(len(p) for p in pieces)
    labeled = sum(1 for p in pieces for n in p if n.finger >= 1)
    print(f"ThumbSet: {len(pieces)} pieces, {total} notes, "
          f"{labeled} labeled ({100*labeled/max(total,1):.1f}%)")

    train_pieces, val_pieces = piece_level_split(pieces, 0.1, CFG.train.seed)
    train_seqs = build_hand_sequences(train_pieces, CFG.train.max_seq_len)
    val_seqs = build_hand_sequences(val_pieces, CFG.train.max_seq_len)
    train_loader = DataLoader(FingeringDataset(train_seqs, augment=True),
                              batch_size=CFG.train.batch_size, shuffle=True,
                              collate_fn=collate)
    val_loader = DataLoader(FingeringDataset(val_seqs), batch_size=CFG.train.batch_size,
                            collate_fn=collate)

    model = KeyGenius().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=CFG.train.weight_decay)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            opt.zero_grad()
            loss, _ = model.pretrain_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.train.grad_clip)
            opt.step()
            running += loss.item()
        mr = labeled_match_rate(model, val_loader, device)
        print(f"epoch {epoch:3d} | focal {running/len(train_loader):.4f} "
              f"| val labeled-match {mr*100:.2f}%")
        if mr > best:
            best = mr
            torch.save({"model": model.state_dict(), "match_rate": best,
                        "epoch": epoch, "phase": "pretrain"}, args.out)
            print(f"  -> saved pretrained ({best*100:.2f}%) to {args.out}")

    print(f"done. best pretrain labeled-match: {best*100:.2f}%")
    print(f"now fine-tune:  python train.py --init {args.out} --lr 1e-4")


if __name__ == "__main__":
    main()

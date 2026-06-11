"""
Evaluation metrics for piano fingering.

The field-standard number on PIG is the MATCH RATE: fraction of notes whose
predicted finger equals the ground-truth finger. State-of-the-art on PIG sits
around 65-71% — so a clean ~70% here is a *real* result, not the bogus 99%
the old overfit pipeline reported.

`general_match_rate` implements M_gen for pieces that have several annotators:
a prediction counts as correct if it matches ANY annotator. Use it for your
final reported number when a piece has multiple fingering files.
"""
from __future__ import annotations

from collections import Counter

import torch


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    dist = Counter()
    for batch in loader:
        for k in ("cont", "pc", "hand", "labels", "mask"):
            batch[k] = batch[k].to(device)
        paths = model.predict(batch["cont"], batch["pc"], batch["hand"], batch["mask"])
        labels = batch["labels"]
        mask = batch["mask"]
        for b, path in enumerate(paths):
            gold = labels[b][mask[b]].tolist()
            for pred, g in zip(path, gold):
                correct += int(pred == g)
                total += 1
                dist[pred + 1] += 1
    match_rate = correct / max(total, 1)
    # normalize finger distribution for readability
    s = sum(dist.values()) or 1
    dist_pct = {f: round(100 * dist.get(f, 0) / s, 1) for f in range(1, 6)}
    return {"match_rate": match_rate, "n_notes": total, "finger_dist_pct": dist_pct}


def general_match_rate(predictions, annotations_per_note):
    """M_gen for multi-annotator pieces.

    predictions:           list[int] length N (predicted finger 1..5)
    annotations_per_note:  list[set[int]] length N (all annotators' fingers)
    """
    correct = sum(1 for p, gt in zip(predictions, annotations_per_note) if p in gt)
    return correct / max(len(predictions), 1)

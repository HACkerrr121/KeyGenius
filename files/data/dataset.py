"""
Dataset, collation, and splitting for KeyGenius.

Key correctness choices:
  * Split is BY PIECE, not by sequence — the same piece never appears in both
    train and val (prevents the inflated accuracy the old pipeline reported).
  * One training example = one hand's note sequence (its own CRF chain).
  * Pitch transposition augmentation (fingering is ~invariant to transposition);
    applied on-the-fly to the train split only.
  * Long sequences are windowed to max_seq_len.
"""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from config import CFG, MIDI_MIN, MIDI_MAX
from data.features import build_features
from data.pig_parser import split_by_hand


def _window(seq, max_len):
    if len(seq) <= max_len:
        return [seq]
    return [seq[i:i + max_len] for i in range(0, len(seq), max_len)]


def build_hand_sequences(pieces, max_len):
    """Flatten pieces -> list of (hand_sequence_of_Notes) windows.

    Returns a list where each item is a list[Note] all from one hand.
    """
    sequences = []
    for notes in pieces:
        for _, hand_notes in split_by_hand(notes).items():
            if len(hand_notes) < 2:
                continue
            sequences.extend(_window(hand_notes, max_len))
    return sequences


def piece_level_split(pieces, val_fraction, seed):
    rng = random.Random(seed)
    idx = list(range(len(pieces)))
    rng.shuffle(idx)
    n_val = max(1, int(len(pieces) * val_fraction))
    val_ids = set(idx[:n_val])
    train = [p for i, p in enumerate(pieces) if i not in val_ids]
    val = [p for i, p in enumerate(pieces) if i in val_ids]
    return train, val


class FingeringDataset(Dataset):
    def __init__(self, hand_sequences, augment=False):
        self.seqs = hand_sequences
        self.augment = augment
        self.tcfg = CFG.train

    def __len__(self):
        return len(self.seqs)

    def _maybe_transpose(self, notes):
        if not self.augment:
            return notes
        lo, hi = self.tcfg.transpose_range
        shift = random.randint(lo, hi)
        if shift == 0:
            return notes
        midis = [n.midi for n in notes]
        if min(midis) + shift < MIDI_MIN or max(midis) + shift > MIDI_MAX:
            return notes  # would leave the keyboard; skip
        out = copy.deepcopy(notes)
        for n in out:
            n.midi += shift
        return out

    def __getitem__(self, i):
        notes = self._maybe_transpose(self.seqs[i])
        cont, pc, hand = build_features(notes)
        labels = np.array([n.finger - 1 for n in notes], dtype=np.int64)  # 0..4
        return {
            "cont": torch.from_numpy(cont),
            "pc": torch.from_numpy(pc),
            "hand": torch.from_numpy(hand),
            "labels": torch.from_numpy(labels),
            "length": len(notes),
        }


def collate(batch):
    B = len(batch)
    T = max(b["length"] for b in batch)
    ncont = batch[0]["cont"].shape[1]

    cont = torch.zeros(B, T, ncont)
    pc = torch.zeros(B, T, dtype=torch.long)
    hand = torch.zeros(B, T, dtype=torch.long)
    labels = torch.zeros(B, T, dtype=torch.long)
    mask = torch.zeros(B, T, dtype=torch.bool)

    for i, b in enumerate(batch):
        L = b["length"]
        cont[i, :L] = b["cont"]
        pc[i, :L] = b["pc"]
        hand[i, :L] = b["hand"]
        labels[i, :L] = b["labels"]
        mask[i, :L] = True
    return {"cont": cont, "pc": pc, "hand": hand, "labels": labels, "mask": mask}

"""
Feature extraction for a single per-hand note sequence.

Used by BOTH training (on PIG Notes) and inference (on notes parsed from
MusicXML/MIDI). Any object with .onset, .offset, .midi, .hand works, so the
exact same feature code runs in both paths. Output widths are pinned in
config.FeatureConfig.

Continuous features (8), all designed to be hand-symmetric where it matters:
  0  pitch        normalized into the 88-key range
  1  duration     log-scaled note length
  2  ioi_prev     log inter-onset interval to previous note
  3  ioi_next     log inter-onset interval to next note
  4  int_prev     signed pitch interval to previous note (/12)  <- ascending/descending
  5  int_next     signed pitch interval to next note (/12)
  6  chord_prev   1 if previous note shares this onset (chord)
  7  chord_next   1 if next note shares this onset (chord)
"""
from __future__ import annotations

import math

import numpy as np

from config import MIDI_MIN, MIDI_MAX, CFG

_CHORD_EPS = 1e-3


def _log(x: float) -> float:
    return math.log1p(max(x, 0.0))


def build_features(notes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """notes: ordered list (one hand) of objects with onset/offset/midi/hand.

    Returns (continuous[T,8] float32, pitch_class[T] int64, hand[T] int64).
    """
    T = len(notes)
    cont = np.zeros((T, CFG.feat.n_continuous), dtype=np.float32)
    pc = np.zeros(T, dtype=np.int64)
    hand = np.zeros(T, dtype=np.int64)

    span = float(MIDI_MAX - MIDI_MIN)
    for i, n in enumerate(notes):
        prev = notes[i - 1] if i > 0 else None
        nxt = notes[i + 1] if i < T - 1 else None

        cont[i, 0] = (n.midi - MIDI_MIN) / span
        cont[i, 1] = _log(n.offset - n.onset)
        if prev is not None:
            cont[i, 2] = _log(n.onset - prev.onset)
            cont[i, 4] = (n.midi - prev.midi) / 12.0
            cont[i, 6] = 1.0 if abs(n.onset - prev.onset) < _CHORD_EPS else 0.0
        if nxt is not None:
            cont[i, 3] = _log(nxt.onset - n.onset)
            cont[i, 5] = (nxt.midi - n.midi) / 12.0
            cont[i, 7] = 1.0 if abs(nxt.onset - n.onset) < _CHORD_EPS else 0.0

        pc[i] = n.midi % 12
        hand[i] = n.hand

    return cont, pc, hand

"""
Parser for the PIG (PIano fingernG) dataset.

PIG fingering files are tab-separated with a version header line:

    //Version: PianoFingering_v...
    id  onset  offset  pitch  onset_vel  offset_vel  channel  finger

  - pitch  : spelled pitch, e.g. "C4", "F#5", "Bb3", "C##2"
  - channel: 0 = right hand, 1 = left hand
  - finger : 1..5 (right hand) or -1..-5 (left hand). Thumb = |1|, pinky = |5|.
             Substitutions look like "1_5" / "-2_-1" -> we keep the FIRST finger.

Download: https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/
(Academic / non-profit use only — cite Nakamura, Saito, Yoshii 2020.)
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

from config import MIDI_MIN, MIDI_MAX

_LETTER_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def spelled_pitch_to_midi(token: str) -> int:
    """'C4' -> 60, 'A0' -> 21, 'F#5' -> 78, 'Bb3' -> 58, 'C##2' -> 38."""
    token = token.strip()
    i = 0
    letter = token[i].upper()
    i += 1
    accidental = 0
    while i < len(token) and token[i] in "#b":
        accidental += 1 if token[i] == "#" else -1
        i += 1
    octave = int(token[i:])
    semitone = _LETTER_SEMITONE[letter] + accidental
    # MIDI: C-1 = 0, so C4 = 60 => 12*(octave+1) + semitone
    return 12 * (octave + 1) + semitone


def _parse_finger(field: str) -> int:
    """Return signed finger 1..5 / -1..-5, taking first of a substitution."""
    first = field.split("_")[0]
    return int(first)


@dataclass
class Note:
    onset: float
    offset: float
    midi: int
    finger: int       # 1..5 (absolute; thumb=1, pinky=5)
    hand: int         # 0 = right, 1 = left

    @property
    def duration(self) -> float:
        return max(self.offset - self.onset, 1e-4)


def parse_pig_file(path: str) -> list[Note]:
    notes: list[Note] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                parts = line.split()          # some files are space-separated
            if len(parts) < 8:
                continue
            try:
                onset = float(parts[1])
                offset = float(parts[2])
                midi = spelled_pitch_to_midi(parts[3])
                signed_finger = _parse_finger(parts[7])
            except (ValueError, IndexError):
                continue

            if not (MIDI_MIN <= midi <= MIDI_MAX):
                continue
            hand = 0 if signed_finger > 0 else 1
            finger = abs(signed_finger)
            if not (1 <= finger <= 5):
                continue
            notes.append(Note(onset, offset, midi, finger, hand))
    return notes


def split_by_hand(notes: list[Note]) -> dict[int, list[Note]]:
    """Split a piece into per-hand sequences sorted by (onset, pitch).

    Each hand is its own CRF chain — this is what makes finger-transition
    modeling meaningful (you never transition between hands).
    """
    out: dict[int, list[Note]] = {0: [], 1: []}
    for n in notes:
        out[n.hand].append(n)
    for h in out:
        out[h].sort(key=lambda x: (x.onset, x.midi))
    return out


def load_pieces(pig_root: str, pattern: str) -> list[list[Note]]:
    """Load every fingering file under pig_root as one list[Note] per file."""
    paths = sorted(glob.glob(os.path.join(pig_root, pattern), recursive=True))
    pieces = []
    for p in paths:
        notes = parse_pig_file(p)
        if len(notes) >= 2:
            pieces.append(notes)
    if not pieces:
        raise FileNotFoundError(
            f"No PIG fingering files found under {pig_root!r} matching {pattern!r}. "
            "Download PIG and point Paths.pig_root at the unzipped folder."
        )
    return pieces


if __name__ == "__main__":
    from config import CFG
    pieces = load_pieces(CFG.paths.pig_root, CFG.paths.fingering_glob)
    total = sum(len(p) for p in pieces)
    print(f"Loaded {len(pieces)} pieces, {total} notes total.")
    hands = split_by_hand(pieces[0])
    print(f"Piece 0: {len(hands[0])} RH notes, {len(hands[1])} LH notes")

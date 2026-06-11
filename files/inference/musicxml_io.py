"""
Symbolic music IO for inference.

Parsing:  MusicXML / .mxl / MIDI  ->  per-hand note sequences (music21).
Writing:  predicted fingerings are written BACK into the score as real
          `articulations.Fingering` notation elements, then exported as
          MusicXML and (optionally) rendered to PNG/PDF by MuseScore.

Why this matters: the old pipeline tried to extract pixel (x, y) boxes from the
sheet image and stamp red numbers on top — which constantly misaligned (210
boxes for 395 notes, etc.). Writing fingerings as notation lets the engraver
place them correctly by construction. No coordinate extraction at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from music21 import converter, articulations, chord as m21chord, note as m21note


@dataclass
class MNote:
    onset: float
    offset: float
    midi: int
    hand: int
    ref: object = field(repr=False, default=None)   # music21 Note or Chord
    pitch_index: int = 0                             # which pitch within a chord


def _hand_of_part(part_index: int, total_parts: int) -> int:
    """RH = 0, LH = 1. Standard piano export: part 0 treble, part 1 bass."""
    if total_parts >= 2:
        return 0 if part_index == 0 else 1
    return 0  # single staff -> treat as right hand


def parse_score(path: str) -> tuple[object, list[MNote]]:
    """Return (music21 score, flat list of MNote) for a MusicXML/MIDI file."""
    score = converter.parse(path)
    parts = list(score.parts) if score.parts else [score]
    total = len(parts)
    mnotes: list[MNote] = []

    for pi, part in enumerate(parts):
        hand = _hand_of_part(pi, total)
        for el in part.flatten().notes:
            onset = float(el.offset)
            dur = float(el.quarterLength) or 0.25
            if isinstance(el, m21chord.Chord):
                for ci, p in enumerate(el.pitches):
                    mnotes.append(MNote(onset, onset + dur, p.midi, hand,
                                        ref=el, pitch_index=ci))
            elif isinstance(el, m21note.Note):
                mnotes.append(MNote(onset, onset + dur, el.pitch.midi, hand,
                                    ref=el, pitch_index=0))
    return score, mnotes


def hand_sequences(mnotes: list[MNote]) -> dict[int, list[MNote]]:
    """Split + sort exactly as training did: per hand, by (onset, midi)."""
    out = {0: [], 1: []}
    for n in mnotes:
        out[n.hand].append(n)
    for h in out:
        out[h].sort(key=lambda x: (x.onset, x.midi))
    return out


def write_fingerings(score, mnote_to_finger: dict[int, int], out_path: str,
                     render: bool = False):
    """Attach predicted fingerings to notes and export.

    mnote_to_finger: maps id(MNote.ref-tagged note) -> we instead attach per
    MNote below; here we expect the caller to have set .pred on each MNote.
    """
    score.write("musicxml", fp=out_path)
    if render:
        # Requires MuseScore configured in music21 environment.
        try:
            png = out_path.rsplit(".", 1)[0] + ".png"
            score.write("musicxml.png", fp=png)
            return out_path, png
        except Exception as e:  # noqa: BLE001
            print(f"[warn] PNG render failed (is MuseScore installed?): {e}")
    return out_path, None


def attach_fingerings(mnotes: list[MNote], predictions: dict[int, int]):
    """predictions: maps the index in `mnotes` (a flat list) -> finger 1..5.

    Attaches an articulations.Fingering to each referenced music21 element.
    """
    for idx, mn in enumerate(mnotes):
        finger = predictions.get(idx)
        if finger is None or mn.ref is None:
            continue
        mn.ref.articulations.append(articulations.Fingering(finger))


def flag_low_confidence(mnotes: list[MNote], flagged_indices, color: str = "red"):
    """Color the noteheads of low-confidence notes so they stand out in MuseScore.

    flagged_indices: iterable of indices into `mnotes` to mark for review.
    """
    flagged = set(flagged_indices)
    for idx, mn in enumerate(mnotes):
        if idx in flagged and mn.ref is not None:
            try:
                mn.ref.style.color = color
            except Exception:  # noqa: BLE001
                pass

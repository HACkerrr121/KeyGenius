"""
End-to-end fingering prediction.

Usage:
    python predict.py input.musicxml -o output_fingered.musicxml
    python predict.py input.mid      -o out.musicxml
    python predict.py photo.jpg      -o out.musicxml --render   # runs OMR first

Pipeline:
    input -> [OMR if image] -> MusicXML -> per-hand note sequences
          -> features -> KeyGenius (+CRF) -> fingerings
          -> written back into the score as notation -> MusicXML (+optional PNG)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from config import CFG
from data.features import build_features
from model.architecture import KeyGenius
from inference.musicxml_io import (
    parse_score, hand_sequences, attach_fingerings, write_fingerings,
    flag_low_confidence,
)
from inference.omr import image_to_musicxml

_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pdf")
_SYMBOLIC_EXT = (".musicxml", ".mxl", ".xml", ".mid", ".midi")


def load_model(ckpt_path: str, device: str) -> KeyGenius:
    model = KeyGenius().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    mr = state.get("match_rate")
    if mr is not None:
        print(f"loaded model (val match rate {mr*100:.1f}%)")
    return model


@torch.no_grad()
def predict_hand(model, seq, device):
    """seq: ordered list[MNote] for one hand.

    Returns (fingers 1..5, confidence in [0,1]) per note.
    """
    cont, pc, hand = build_features(seq)
    cont = torch.from_numpy(cont).unsqueeze(0).to(device)
    pc = torch.from_numpy(pc).unsqueeze(0).to(device)
    hand = torch.from_numpy(hand).unsqueeze(0).to(device)
    mask = torch.ones(1, len(seq), dtype=torch.bool, device=device)
    path, conf = model.predict_with_confidence(cont, pc, hand, mask)
    return [c + 1 for c in path], conf


def run(input_path: str, out_path: str, ckpt: str, render: bool,
        flag_threshold: float = 0.6):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt, device)

    ext = os.path.splitext(input_path)[1].lower()
    if ext in _IMAGE_EXT:
        print("running OMR on image...")
        xml_path = image_to_musicxml(input_path)
        print(f"OMR -> {xml_path}")
    elif ext in _SYMBOLIC_EXT:
        xml_path = input_path
    else:
        raise ValueError(f"Unsupported input type: {ext}")

    score, mnotes = parse_score(xml_path)
    print(f"parsed {len(mnotes)} notes")

    # keep global indices so we can write predictions back to the right notes
    indexed = list(enumerate(mnotes))
    per_hand = {0: [], 1: []}
    for idx, mn in indexed:
        per_hand[mn.hand].append((idx, mn))
    for h in per_hand:
        per_hand[h].sort(key=lambda pair: (pair[1].onset, pair[1].midi))

    predictions: dict[int, int] = {}
    confidence: dict[int, float] = {}
    for h, pairs in per_hand.items():
        if not pairs:
            continue
        seq = [mn for _, mn in pairs]
        fingers, conf = predict_hand(model, seq, device)
        for (idx, _), f, c in zip(pairs, fingers, conf):
            predictions[idx] = f
            confidence[idx] = c

    # flag the notes the model is least sure about
    flagged = [idx for idx, c in confidence.items() if c < flag_threshold]
    attach_fingerings(mnotes, predictions)
    flag_low_confidence(mnotes, flagged)

    out, png = write_fingerings(score, predictions, out_path, render=render)
    print(f"wrote {out}")
    if png:
        print(f"rendered {png}")

    # review report
    if confidence:
        avg = sum(confidence.values()) / len(confidence)
        print(f"\nmean confidence: {avg*100:.1f}%")
        print(f"flagged for review (< {flag_threshold:.0%}): "
              f"{len(flagged)}/{len(confidence)} notes (shown in red)")
        if flagged:
            HAND = {0: "RH", 1: "LH"}
            shown = sorted(flagged, key=lambda i: confidence[i])[:10]
            for idx in shown:
                mn = mnotes[idx]
                print(f"  {HAND[mn.hand]} midi {mn.midi} @ beat {mn.onset:.2f} "
                      f"-> finger {predictions[idx]} ({confidence[idx]*100:.0f}%)")
            if len(flagged) > 10:
                print(f"  ... and {len(flagged) - 10} more")


def main():
    ap = argparse.ArgumentParser(description="KeyGenius fingering prediction")
    ap.add_argument("input", help="MusicXML / MIDI / image of sheet music")
    ap.add_argument("-o", "--out", default="fingered.musicxml")
    ap.add_argument("--ckpt", default=CFG.paths.best_ckpt)
    ap.add_argument("--render", action="store_true",
                    help="also render a PNG (needs MuseScore configured)")
    ap.add_argument("--flag-threshold", type=float, default=0.6,
                    help="flag (color red) notes with confidence below this")
    args = ap.parse_args()
    run(args.input, args.out, args.ckpt, args.render, args.flag_threshold)


if __name__ == "__main__":
    main()

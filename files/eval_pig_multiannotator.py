"""
Evaluate KeyGenius against PIG's multi-annotator subsets.

PIG fingering files are named  {piece}-{annotator}_fingering.txt
(e.g. 001-1_fingering.txt, 001-2_fingering.txt). The Bach/Mozart/Chopin test
pieces have 4-6 annotators each. This script:

  1. groups files by piece,
  2. for pieces with >=2 annotators, builds the SET of human fingers per note,
  3. runs the model and reports:
       - M_single : match rate vs each single annotator (averaged)
       - M_gen    : general match rate (prediction matches ANY annotator)
       - human    : inter-annotator agreement (the ceiling)

If M_gen lands near `human`, the model is at the human ceiling — which is the
honest, strong way to report results on a subjective task.

Run:  python eval_pig_multiannotator.py --ckpt checkpoints/keygenius_best.pt
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict

import torch

from config import CFG
from data.pig_parser import parse_pig_file, split_by_hand
from data.features import build_features
from model.architecture import KeyGenius


def group_by_piece(pig_root: str):
    """Return {piece_id: [file_path, ...]} for files with annotator suffixes."""
    files = glob.glob(os.path.join(pig_root, "**", "*_fingering*.txt"),
                      recursive=True)
    groups = defaultdict(list)
    for f in files:
        stem = os.path.basename(f).split("_fingering")[0]   # e.g. "001-2"
        piece_id = stem.split("-")[0]                        # e.g. "001"
        groups[piece_id].append(f)
    return groups


def _key(note):
    return (note.hand, round(note.onset, 3), note.midi)


@torch.no_grad()
def predict_piece(model, notes, device):
    """Predict fingers for one annotator's note list. Returns {key: finger}."""
    preds = {}
    for h, seq in split_by_hand(notes).items():
        if len(seq) < 1:
            continue
        cont, pc, hand = build_features(seq)
        cont = torch.from_numpy(cont).unsqueeze(0).to(device)
        pc = torch.from_numpy(pc).unsqueeze(0).to(device)
        hand = torch.from_numpy(hand).unsqueeze(0).to(device)
        mask = torch.ones(1, len(seq), dtype=torch.bool, device=device)
        path = model.predict(cont, pc, hand, mask)[0]
        for n, c in zip(seq, path):
            preds[_key(n)] = c + 1
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CFG.paths.best_ckpt)
    ap.add_argument("--pig_root", default=CFG.paths.pig_root)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KeyGenius().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.eval()

    groups = group_by_piece(args.pig_root)
    multi = {pid: fs for pid, fs in groups.items() if len(fs) >= 2}
    if not multi:
        print("No multi-annotator pieces found. (PIG's Bach/Mozart/Chopin "
              "subsets have several annotators each — make sure they're present.)")
        return
    print(f"{len(multi)} multi-annotator pieces")

    tot_single = tot_gen = tot_human = 0.0
    n_pieces = 0
    for pid, files in sorted(multi.items()):
        annotations = [parse_pig_file(f) for f in files]
        annotations = [a for a in annotations if a]
        if len(annotations) < 2:
            continue

        # per-note set of human fingers, keyed by (hand, onset, midi)
        finger_sets = defaultdict(set)
        for ann in annotations:
            for n in ann:
                finger_sets[_key(n)].add(n.finger)

        # model predictions on the first annotator's note layout
        preds = predict_piece(model, annotations[0], device)

        keys = [k for k in preds if k in finger_sets]
        if not keys:
            continue

        gen = sum(preds[k] in finger_sets[k] for k in keys) / len(keys)

        # average single-annotator match rate
        singles = []
        for ann in annotations:
            ann_map = {_key(n): n.finger for n in ann}
            shared = [k for k in keys if k in ann_map]
            if shared:
                singles.append(sum(preds[k] == ann_map[k] for k in shared) / len(shared))
        single = sum(singles) / len(singles) if singles else 0.0

        # inter-annotator agreement (ceiling): avg pairwise match between humans
        pair_scores = []
        maps = [{_key(n): n.finger for n in ann} for ann in annotations]
        for i in range(len(maps)):
            for j in range(i + 1, len(maps)):
                shared = [k for k in maps[i] if k in maps[j]]
                if shared:
                    pair_scores.append(
                        sum(maps[i][k] == maps[j][k] for k in shared) / len(shared))
        human = sum(pair_scores) / len(pair_scores) if pair_scores else float("nan")

        tot_single += single
        tot_gen += gen
        tot_human += human
        n_pieces += 1
        print(f"  piece {pid} ({len(annotations)} annot): "
              f"M_single {single*100:5.1f}%  M_gen {gen*100:5.1f}%  "
              f"human {human*100:5.1f}%")

    if n_pieces:
        print("\n=== averages over {} pieces ===".format(n_pieces))
        print(f"M_single (vs one annotator) : {tot_single/n_pieces*100:.1f}%")
        print(f"M_gen    (vs any annotator) : {tot_gen/n_pieces*100:.1f}%")
        print(f"human ceiling (inter-annot) : {tot_human/n_pieces*100:.1f}%")
        print("\nReport M_gen as your headline number, with the human ceiling "
              "for context.")


if __name__ == "__main__":
    main()

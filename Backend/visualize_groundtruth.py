"""
Visualize ground truth fingerings from annotation files on the score image.
Uses Audiveris for pixel-accurate coordinates.
"""
import sys
import os
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "UI" / "python"))

from Datasets import parse_fingering_file
from inference import extract_coords_from_omr, _build_audiveris_args
from fast_oemer_extract import extract_from_musicxml

import subprocess, tempfile, zipfile


NUMBERS_DIR = Path(__file__).parent.parent / "UI" / "python" / "numbers"


def run_audiveris_and_get_coords(img_path):
    img_path = str(Path(img_path).resolve())
    tmp_dir = tempfile.mkdtemp(prefix="audiveris_gt_")
    subprocess.run(
        _build_audiveris_args(tmp_dir, img_path),
        capture_output=True, text=True, timeout=300
    )
    stem = Path(img_path).stem
    omr_path = os.path.join(tmp_dir, f"{stem}.omr")
    mxl_path = os.path.join(tmp_dir, f"{stem}.mxl")
    coords = extract_coords_from_omr(omr_path)
    notes_data = extract_from_musicxml(mxl_path)
    return notes_data, coords


def overlay_number(bg, x, y, finger, scale):
    img_h, img_w = bg.shape[:2]
    overlay_path = NUMBERS_DIR / f"{finger}_small.png"
    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_UNCHANGED)
    if overlay is None:
        return bg

    h, w = overlay.shape[:2]
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    overlay = cv2.resize(overlay, (new_w, new_h))

    x = x - new_w // 2
    x = max(0, min(x, img_w - new_w))
    y = max(0, min(y - new_h - 4, img_h - new_h))

    if x + new_w > img_w or y + new_h > img_h or y < 0:
        return bg

    b, g, r, a = cv2.split(overlay)
    mask = a.astype(float) / 255.0
    roi = bg[y:y+new_h, x:x+new_w]
    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1 - mask) + [b, g, r][c] * mask
    bg[y:y+new_h, x:x+new_w] = roi
    return bg


def visualize(img_path, fingering_path, output_path):
    print(f"Running Audiveris on {img_path}...")
    notes_data, coords = run_audiveris_and_get_coords(img_path)
    print(f"  {len(notes_data)} notes, {len(coords)} coords from .omr")

    print(f"Loading ground truth from {fingering_path}...")
    gt_notes = parse_fingering_file(fingering_path)
    print(f"  {len(gt_notes)} ground truth notes")

    # Match Audiveris notes to ground truth by note name + hand in sequence order
    # Both are sorted in reading/time order — match greedily
    gt_right = [n for n in gt_notes if n['hand'] == 0]
    gt_left  = [n for n in gt_notes if n['hand'] == 1]

    ri, li = 0, 0
    matched_fingers = []
    for note in notes_data:
        hand = note['hand']
        name = note['note'].replace('b', '-')  # music21 uses b for flat

        if hand == 'RIGHT' and ri < len(gt_right):
            matched_fingers.append(gt_right[ri]['finger'])
            ri += 1
        elif hand == 'LEFT' and li < len(gt_left):
            matched_fingers.append(gt_left[li]['finger'])
            li += 1
        else:
            matched_fingers.append(0)

    print(f"  Matched {sum(1 for f in matched_fingers if f > 0)} notes")

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {img_path}")

    img_h, img_w = img.shape[:2]
    scale = img_w / 1080 * 0.7

    placed = 0
    for i, (note, (cx, cy), finger) in enumerate(zip(notes_data, coords, matched_fingers)):
        if finger < 1 or finger > 5:
            continue
        img = overlay_number(img, cx, cy, finger, scale)
        placed += 1

    print(f"  Placed {placed} fingering overlays")
    cv2.imwrite(output_path, img)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    img   = "Music_Data/Scores/001_Bach_Invention_No1_C_page_0.jpg"
    fing  = "Music_Data/FingeringFiles/001-1_fingering.txt"
    out   = "/tmp/groundtruth_001.jpg"

    if len(sys.argv) == 4:
        img, fing, out = sys.argv[1], sys.argv[2], sys.argv[3]

    visualize(img, fing, out)

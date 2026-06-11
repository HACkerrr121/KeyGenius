"""
Quick test: run new weights on teaser image, show fingerings.
Uses existing MusicXML (skips slow Audiveris step).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

from fast_oemer_extract import extract_from_musicxml
from inference import load_model, notes_to_features, predict_batch

BACKEND = Path(__file__).parent
CHECKPOINT = BACKEND / "best_model_right.pth"
XML_PATH   = BACKEND / "001_Bach_Invention_No1_C_page_0.musicxml"
IMG_PATH   = BACKEND / "001_Bach_Invention_No1_C_page_0_teaser.png"
OUT_PATH   = BACKEND / "test_fingering_output.png"

FINGER_COLORS = {
    1: "#e74c3c",  # red   - thumb
    2: "#e67e22",  # orange
    3: "#f1c40f",  # yellow
    4: "#2ecc71",  # green
    5: "#3498db",  # blue
    0: "#95a5a6",  # gray  - unknown
}

def main():
    print(f"Loading MusicXML: {XML_PATH}")
    notes_data = extract_from_musicxml(str(XML_PATH))
    print(f"  {len(notes_data)} notes extracted")

    right = [n for n in notes_data if n['hand'] == 'RIGHT']
    left  = [n for n in notes_data if n['hand'] == 'LEFT']
    print(f"  RIGHT: {len(right)}, LEFT: {len(left)}")

    print(f"Loading model: {CHECKPOINT}")
    model = load_model(str(CHECKPOINT))

    features = notes_to_features(notes_data)
    print("Running inference...")
    fingers   = predict_batch(model, features)
    print(f"  Finger distribution: {Counter(fingers)}")

    # --- Visualize ---
    img = Image.open(IMG_PATH).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    try:
        font_big  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font_big  = ImageFont.load_default()
        font_small = font_big

    # Scale beat offsets → pixel x
    offsets = [n['start_time'] for n in notes_data]
    max_offset = max(offsets) if offsets else 1.0
    LEFT_MARGIN  = 80
    RIGHT_MARGIN = 80
    usable_w = W - LEFT_MARGIN - RIGHT_MARGIN

    # Right hand stays in top half, left hand bottom half
    # (matches treble/bass stave layout)
    for note, finger in zip(notes_data, fingers):
        offset = note['start_time']
        x = int(LEFT_MARGIN + (offset / max_offset) * usable_w)

        if note['hand'] == 'RIGHT':
            y = int(H * 0.28)
        else:
            y = int(H * 0.72)

        color = FINGER_COLORS.get(finger, "#95a5a6")
        r = 18
        draw.ellipse([x - r, y - r, x + r, y + r],
                     fill=color + "CC", outline="white", width=2)
        txt = str(finger)
        bbox = draw.textbbox((0, 0), txt, font=font_big)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x - tw // 2, y - th // 2), txt, fill="white", font=font_big)

    # Legend
    legend_x, legend_y = 20, 20
    draw.rectangle([legend_x - 5, legend_y - 5,
                    legend_x + 210, legend_y + 185],
                   fill=(255, 255, 255, 200), outline="gray")
    draw.text((legend_x + 5, legend_y), "Finger legend:", fill="black", font=font_small)
    labels = {1: "1 = Thumb", 2: "2 = Index", 3: "3 = Middle",
              4: "4 = Ring", 5: "5 = Pinky"}
    for i, (f, label) in enumerate(labels.items()):
        cy = legend_y + 30 + i * 28
        draw.ellipse([legend_x + 5, cy - 10, legend_x + 25, cy + 10],
                     fill=FINGER_COLORS[f])
        draw.text((legend_x + 32, cy - 8), label, fill="black", font=font_small)

    img.save(OUT_PATH)
    print(f"\nSaved visualization → {OUT_PATH}")

    # Print first 20 fingerings
    print(f"\n{'Note':8s} {'Hand':6s} {'Beat':6s} {'Finger'}")
    print("-" * 35)
    for note, finger in list(zip(notes_data, fingers))[:25]:
        print(f"{note['note']:8s} {note['hand']:6s} {note['start_time']:6.2f}  {finger}")

    # Open the image
    import subprocess
    subprocess.Popen(["open", str(OUT_PATH)])
    print("\nImage opened.")

if __name__ == "__main__":
    main()

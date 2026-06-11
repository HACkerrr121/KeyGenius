"""
Full Audiveris pipeline: image → .omr pixel coords → model → viz on note heads.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
from inference import infer

BACKEND    = Path(__file__).parent
CHECKPOINT = BACKEND / "best_model_right.pth"
IMG_PATH   = BACKEND / "001_Bach_Invention_No1_C_page_0_teaser.png"
OUT_PATH   = BACKEND / "test_audiveris_output.png"

FINGER_COLORS = {
    1: (231, 76,  60,  210),   # red   - thumb
    2: (230, 126, 34,  210),   # orange
    3: (241, 196, 15,  210),   # yellow
    4: (46,  204, 113, 210),   # green
    5: (52,  152, 219, 210),   # blue
    0: (149, 165, 166, 210),   # gray
}

def main():
    print(f"Running Audiveris + model on: {IMG_PATH}")
    notes, coords, fingers = infer(str(IMG_PATH), str(CHECKPOINT))

    if not notes:
        print("No notes returned!")
        return

    print(f"\n{len(notes)} notes with pixel coords")
    print(f"Finger distribution: {Counter(fingers)}")

    img = Image.open(IMG_PATH).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
    except Exception:
        font = ImageFont.load_default()

    for note, (x, y, page), finger in zip(notes, coords, fingers):
        color = FINGER_COLORS.get(finger, FINGER_COLORS[0])
        # Draw label above the note head so it's not hidden by the note
        lx, ly = x, y - 38
        r = 20
        # Shadow for contrast
        draw.ellipse([lx - r + 2, ly - r + 2, lx + r + 2, ly + r + 2],
                     fill=(0, 0, 0, 120))
        draw.ellipse([lx - r, ly - r, lx + r, ly + r],
                     fill=color, outline=(255, 255, 255, 255), width=3)
        # Thin connecting line from label to note
        draw.line([(lx, ly + r), (x, y)], fill=(0, 0, 0, 160), width=1)
        txt = str(finger)
        bbox = draw.textbbox((0, 0), txt, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((lx - tw // 2, ly - th // 2), txt, fill=(255, 255, 255, 255), font=font)

    # Legend
    try:
        lfont = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except Exception:
        lfont = font
    lx, ly = 15, 15
    draw.rectangle([lx, ly, lx + 175, ly + 175], fill=(255,255,255,200), outline=(180,180,180,255))
    draw.text((lx+8, ly+5), "Fingering", fill=(0,0,0,255), font=lfont)
    labels = {1:"Thumb",2:"Index",3:"Middle",4:"Ring",5:"Pinky"}
    for i, (f, name) in enumerate(labels.items()):
        cy = ly + 35 + i * 27
        c = FINGER_COLORS[f]
        draw.ellipse([lx+8, cy-10, lx+28, cy+10], fill=c)
        draw.text((lx+36, cy-9), f"{f} = {name}", fill=(0,0,0,255), font=lfont)

    img.save(OUT_PATH)
    print(f"\nSaved → {OUT_PATH}")

    print(f"\n{'#':>3}  {'Note':8} {'x':>5} {'y':>5}  Finger")
    print("-" * 35)
    for i, (note, (x, y, _), f) in enumerate(zip(notes, coords, fingers)):
        print(f"{i+1:>3}  {note:8} {x:>5} {y:>5}  {f}")

    import subprocess
    subprocess.Popen(["open", str(OUT_PATH)])

if __name__ == "__main__":
    main()

"""Re-render from cached inference results, no Audiveris re-run."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BACKEND  = Path(__file__).parent
IMG_PATH = BACKEND / "001_Bach_Invention_No1_C_page_0_teaser.png"
OUT_PATH = BACKEND / "test_audiveris_output.png"

FINGER_COLORS = {
    1: (231, 76,  60,  220),
    2: (230, 126, 34,  220),
    3: (200, 180, 0,   220),
    4: (46,  204, 113, 220),
    5: (52,  152, 219, 220),
    0: (149, 165, 166, 220),
}

# Cached results from last Audiveris run
notes  = ['C3','E3','G3','B3','E4','F4','F4','F#4','A4','A4','A4','C5','C5','C5','D5','D5','G5','G4','B4','B4','E3','D4','D4','F#4','G4','Ab4','A4','B4','C5','F5','G5','A4','C#5','F3','B3','C4','D4','F4','G4','G4','G4','G4','G#4','B4','C5','C5','E5','G5','G3','C5','C#5','F3','D4','D4','F4','A4','A4','C5','F5','G5','E4','C4','D4','D4','A4','A4']
coords = [(357,271,0),(357,284,0),(630,291,0),(942,263,0),(1043,278,0),(1045,242,0),(1078,270,0),(1110,285,0),(1145,277,0),(1370,258,0),(392,315,0),(429,311,0),(460,305,0),(529,311,0),(562,305,0),(596,316,0),(629,405,0),(665,419,0),(700,412,0),(942,392,0),(1211,405,0),(1211,419,0),(1246,426,0),(1247,392,0),(1279,385,0),(313,570,0),(351,577,0),(388,583,0),(466,577,0),(542,570,0),(1185,598,0),(313,703,0),(466,688,0),(1224,605,0),(1263,612,0),(1302,618,0),(1341,624,0),(1380,609,0),(1417,615,0),(1457,603,0),(959,891,0),(238,934,0),(313,962,0),(351,1048,0),(390,920,0),(390,1043,0),(390,1043,0),(551,1046,0),(553,921,0),(594,934,0),(627,941,0),(672,947,0),(705,956,0),(782,948,0),(783,910,0),(884,935,0),(920,926,0),(959,934,0),(994,922,0),(1031,929,0),(1177,914,0),(1215,927,0),(1338,928,0),(1363,921,0),(1388,927,0),(1424,934,0)]
fingers = [1,2,2,1,4,3,2,1,2,1,1,3,2,1,4,3,5,1,3,5,1,2,1,2,1,2,1,1,2,3,5,1,5,1,2,3,2,1,4,3,2,1,3,2,1,4,5,1,4,5,1,2,2,3,2,1,4,5,1,2,1,2,2,3,4]

img  = Image.open(IMG_PATH).convert("RGB")
draw = ImageDraw.Draw(img, "RGBA")

try:
    font  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
    lfont = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
except Exception:
    font  = ImageFont.load_default()
    lfont = font

for note, (x, y, page), finger in zip(notes, coords, fingers):
    color = FINGER_COLORS.get(finger, FINGER_COLORS[0])
    lx, ly = x, y - 38
    r = 20
    draw.ellipse([lx-r+2, ly-r+2, lx+r+2, ly+r+2], fill=(0,0,0,120))
    draw.ellipse([lx-r, ly-r, lx+r, ly+r], fill=color, outline=(255,255,255,255), width=3)
    draw.line([(lx, ly+r), (x, y)], fill=(0,0,0,160), width=1)
    txt  = str(finger)
    bbox = draw.textbbox((0,0), txt, font=font)
    tw   = bbox[2]-bbox[0]
    th   = bbox[3]-bbox[1]
    draw.text((lx-tw//2, ly-th//2), txt, fill=(255,255,255,255), font=font)

# Legend
lx2, ly2 = 15, 15
draw.rectangle([lx2, ly2, lx2+185, ly2+185], fill=(255,255,255,210), outline=(180,180,180,255))
draw.text((lx2+8, ly2+5), "Fingering", fill=(0,0,0,255), font=lfont)
labels = {1:"Thumb",2:"Index",3:"Middle",4:"Ring",5:"Pinky"}
for i,(f,name) in enumerate(labels.items()):
    cy = ly2+38+i*28
    c  = FINGER_COLORS[f]
    draw.ellipse([lx2+8, cy-11, lx2+30, cy+11], fill=c)
    bx = draw.textbbox((0,0), str(f), font=lfont)
    tw = bx[2]-bx[0]; th = bx[3]-bx[1]
    draw.text((lx2+19-tw//2, cy-th//2), str(f), fill=(255,255,255,255), font=lfont)
    draw.text((lx2+38, cy-10), name, fill=(0,0,0,255), font=lfont)

img.save(OUT_PATH)
print(f"Saved → {OUT_PATH}")
import subprocess; subprocess.Popen(["open", str(OUT_PATH)])

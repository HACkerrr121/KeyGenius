# check_fingering.py
from pathlib import Path

f = Path("Music_Data/FingeringFiles/001-1_fingering.txt")
lines = f.read_text().strip().split("\n")

print(f"Total lines: {len(lines)}")
print("\nFirst 10 lines:")
for line in lines[:10]:
    print(f"  {line}")

print("\nMiddle lines (around 230):")
for line in lines[225:235]:
    print(f"  {line}")
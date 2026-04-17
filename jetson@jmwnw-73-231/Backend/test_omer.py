# test_oemer.py
from oemer import MODULE_PATH
from oemer.inference import inference
import os
import numpy as np
from scipy import ndimage

img_path = 'Music_Data/Scores/001_Bach_Invention_No1_C_page_0.jpg'

print("Running oemer inference...")
staff_pred, _ = inference(os.path.join(MODULE_PATH, 'checkpoints/unet_big'), img_path, use_tf=False)
seg_pred, _ = inference(os.path.join(MODULE_PATH, 'checkpoints/seg_net'), img_path, manual_th=None, use_tf=False)

# Find staff lines
staff_map = (staff_pred == 1).astype(np.uint8)
row_sums = staff_map.sum(axis=1)

print(f"Image height: {staff_pred.shape[0]}")
print(f"Max row sum: {row_sums.max()}")

# Find rows with staff lines
threshold = row_sums.max() * 0.1 if row_sums.max() > 0 else 0
line_rows = np.where(row_sums > threshold)[0]

print(f"Found {len(line_rows)} rows with staff lines")

if len(line_rows) > 0:
    # Group into staves
    gaps = np.diff(line_rows)
    big_gaps = np.where(gaps > 20)[0]
    
    staff_groups = np.split(line_rows, big_gaps + 1)
    print(f"Found {len(staff_groups)} staff line groups")
    
    # Find staff centers - lower threshold
    staff_centers = []
    for i, group in enumerate(staff_groups):
        if len(group) >= 3:  # Changed from 10 to 3
            center = np.mean(group)
            staff_centers.append(center)
            print(f"  Staff {i}: y={group[0]} to {group[-1]}, center={center:.0f}, rows={len(group)}")
    
    print(f"\nFound {len(staff_centers)} valid staves")
    
    # Now get noteheads and assign to staves
    notehead_map = (seg_pred == 2).astype(np.uint8)
    labeled, n_notes = ndimage.label(notehead_map)
    
    print(f"Found {n_notes} noteheads")
    
    notes_data = []
    for i in range(1, n_notes + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < 15 or len(ys) > 3000:
            continue
        
        cx, cy = int(xs.mean()), int(ys.mean())
        
        # Find closest staff
        if staff_centers:
            distances = [abs(cy - sc) for sc in staff_centers]
            closest_staff = np.argmin(distances)
            min_dist = min(distances)
            # In piano music: even staff index = treble (right), odd = bass (left)
            hand = "RIGHT" if closest_staff % 2 == 0 else "LEFT"
        else:
            closest_staff = -1
            min_dist = -1
            hand = "UNKNOWN"
        
        notes_data.append((cx, cy, closest_staff, hand, min_dist))
    
    # Sort by x position (reading order)
    notes_data.sort(key=lambda x: x[0])
    
    print("\nFirst 20 notes with staff assignment:")
    for i, (cx, cy, staff, hand, dist) in enumerate(notes_data[:20]):
        print(f"  Note {i+1}: pos=({cx}, {cy}), staff={staff}, hand={hand}, dist={dist:.0f}")
    
    # Count hands
    right_count = sum(1 for n in notes_data if n[3] == "RIGHT")
    left_count = sum(1 for n in notes_data if n[3] == "LEFT")
    print(f"\nTotal: {right_count} RIGHT hand, {left_count} LEFT hand")
else:
    print("No staff lines found!")
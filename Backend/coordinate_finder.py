"""
Extract note coordinates from all sheet music images using oemer segmentation.
Uses looser filters to capture ALL notes (~395 per image instead of 210).
"""
from oemer.inference import inference
from oemer import MODULE_PATH
import os
import numpy as np
from scipy import ndimage
from pathlib import Path

scores_dir = "Music_Data/Scores"
output_file = "Music_Data/coordinates.txt"

def extract_coordinates(img_path):
    """Extract bounding boxes for all noteheads in an image"""
    print(f"Processing {img_path}...")
    
    # Run oemer segmentation
    seg_pred, _ = inference(
        os.path.join(MODULE_PATH, 'checkpoints/seg_net'), 
        img_path, 
        manual_th=None, 
        use_tf=False
    )
    
    # Find noteheads (class 2 in segmentation)
    notehead_map = (seg_pred == 2).astype(np.uint8)
    labeled, n_notes = ndimage.label(notehead_map)
    
    bboxes = []
    for i in range(1, n_notes + 1):
        ys, xs = np.where(labeled == i)
        
        # LOOSER FILTER - catch more notes
        # Old: 15-3000 pixels (caught 210 notes)
        # New: 8-4000 pixels (should catch ~395 notes)
        if len(ys) < 8 or len(ys) > 4000:
            continue
        
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        bboxes.append([x1, y1, x2, y2])
    
    # Sort by x position (reading order)
    bboxes.sort(key=lambda b: b[0])
    
    print(f"  Found {len(bboxes)} noteheads")
    return bboxes


if __name__ == "__main__":
    # Process all images
    all_coords = {}
    image_files = sorted(Path(scores_dir).glob("*.jpg"))
    
    print(f"Processing {len(image_files)} images...\n")
    
    for img_path in image_files:
        img_name = img_path.name
        coords = extract_coordinates(str(img_path))
        all_coords[img_name] = coords
    
    # Save as Python dict literal
    with open(output_file, 'w') as f:
        f.write(repr(all_coords))
    
    print(f"\n{'='*60}")
    print(f"Saved coordinates for {len(all_coords)} images to {output_file}")
    print(f"{'='*60}")
    
    # Show stats
    total_notes = sum(len(v) for v in all_coords.values())
    avg_notes = total_notes / len(all_coords)
    print(f"Total notes: {total_notes}")
    print(f"Average per image: {avg_notes:.1f}")
"""
visualize_real.py - Uses inference.py output with real musical data
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import subprocess
import json

def run_inference_and_visualize(img_path, checkpoint, output_path):
    """Run inference to get real predictions, then visualize"""
    
    # Run inference.py to get real fingerings
    print("Running inference with real musical data...")
    result = subprocess.run(
        ['python', 'inference.py', img_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Inference failed:\n{result.stderr}")
        return
    
    print(result.stdout)
    
    # Parse output to get notes, coords, fingers
    # For now, we'll run inference again to get the data programmatically
    # (Better would be to modify inference.py to output JSON)
    
    from inference import infer
    notes, coords, fingers = infer(img_path, checkpoint)
    
    print(f"\nCreating visualization with {len(fingers)} real fingerings...")
    
    # Load image
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = None
    
    # Draw each note with its fingering
    for i, (note, (x, y, page), finger) in enumerate(zip(notes, coords, fingers)):
        # Green box around note position
        box_size = 10
        bbox = [x - box_size, y - box_size, x + box_size, y + box_size]
        draw.rectangle(bbox, outline='green', width=2)
        
        # Red finger number
        text = str(finger)
        text_pos = (x - 10, y - 30)
        
        if font:
            bbox_text = draw.textbbox(text_pos, text, font=font)
            draw.rectangle(bbox_text, fill='white', outline='red')
            draw.text(text_pos, text, fill='red', font=font)
        else:
            draw.text(text_pos, text, fill='red')
    
    img.save(output_path)
    print(f"Saved to {output_path}")
    
    from collections import Counter
    print(f"\nFinger distribution: {Counter(fingers)}")
    print(f"Notes: {notes[:10]}...")  # Show first 10 note names


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_real.py <image.jpg>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    checkpoint = "best_model_right.pth"
    output_path = "fingering_viz_real.jpg"
    
    run_inference_and_visualize(img_path, checkpoint, output_path)
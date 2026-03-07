from PIL import Image, ImageOps
import cv2
import numpy as np
import sys
from pathlib import Path


def pil_to_opencv(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def opencv_to_pil(cv_image):
    return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))


def overlay_transparent_image(background_img, overlay_img_rgba, x_offset, y_offset):
    bg_h, bg_w = background_img.shape[:2]

    scale = bg_w / 1080 * 0.3
    h, w = overlay_img_rgba.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    overlay = cv2.resize(overlay_img_rgba, (new_w, new_h))

    if x_offset + new_w > bg_w or y_offset + new_h > bg_h:
        return background_img

    b, g, r, a = cv2.split(overlay)
    overlay_rgb = cv2.merge((b, g, r))

    mask = a.astype(float) / 255.0
    inv_mask = 1.0 - mask

    roi = background_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * inv_mask + overlay_rgb[:, :, c] * mask

    background_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = roi
    return background_img


def process_image(input_path, output_path, notes, coords, fingers):
    try:
        with Image.open(input_path) as img:
            bg = pil_to_opencv(img)
        
        for i in range(len(coords)):
            finger = fingers[i]
            if finger < 1 or finger > 5:
                continue
                
            overlay = cv2.imread(f"./numbers/{finger}_small.png", cv2.IMREAD_UNCHANGED)
            if overlay is None:
                print(f"Number image not found: ./numbers/{finger}_small.png", file=sys.stderr)
                continue
            
            x = coords[i][0]
            y = max(0, coords[i][1] - 40)
            
            bg = overlay_transparent_image(bg, overlay, x, y)

        output_image = opencv_to_pil(bg)
        output_image.save(output_path)
        print(f"Saved to {output_path}")
        
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


def main():
    if len(sys.argv) < 3:
        print("Usage: python process_image.py input.jpg output.jpg")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Add Backend to path
    backend_path = Path(__file__).parent.parent.parent / "Backend"
    sys.path.insert(0, str(backend_path))
    
    from inference import infer
    
    checkpoint = backend_path / "best_model_right.pth"
    if not checkpoint.exists():
        print(f"Model not found: {checkpoint}")
        sys.exit(1)
    
    notes, coords, fingers = infer(input_path, str(checkpoint))
    
    if len(notes) == 0:
        print("No notes found")
        sys.exit(1)
    
    process_image(input_path, output_path, notes, coords, fingers)


if __name__ == "__main__":
    main()
"""
KeyGenius Inference - predict fingerings from sheet music.
Handles multiple pages.
"""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from model import FingeringModel
from fast_oemer_extract import extract_from_image, extract_from_pages, extract_from_folder, NOTE_TO_MIDI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path):
    checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

    
    model = FingeringModel(
        input_dim=5,
        embed_dim=128,
        hidden_dim=256,
        num_layers=3,
        num_fingers=6,
        dropout=0.0
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def notes_to_features(notes):
    """Convert note names to model features."""
    features = []
    prev_midi = None
    
    for note_name in notes:
        midi = NOTE_TO_MIDI.get(note_name, 60)
        
        midi_norm = (midi - 21) / 87.0
        duration = 1.0
        delta = 0.1
        
        if prev_midi is not None:
            interval = midi - prev_midi
            if midi > prev_midi:
                direction = 1.0
            elif midi < prev_midi:
                direction = -1.0
            else:
                direction = 0.0
        else:
            interval = 0
            direction = 0.0
        
        interval_norm = interval / 24.0
        
        features.append([midi_norm, duration, delta, interval_norm, direction])
        prev_midi = midi
    
    return np.array(features, dtype=np.float32)


def predict_batch(model, features, batch_size=200):
    """Predict fingerings for any number of notes."""
    n = len(features)
    all_fingers = []
    all_conf = []
    
    for i in range(0, n, batch_size):
        batch = features[i:i+batch_size]
        seq_len = len(batch)
        
        # Pad to batch_size
        if seq_len < batch_size:
            batch = np.pad(batch, ((0, batch_size - seq_len), (0, 0)), constant_values=0)
        
        mask = np.zeros(batch_size, dtype=np.float32)
        mask[:seq_len] = 1.0
        
        with torch.no_grad():
            feat_t = torch.from_numpy(batch).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
            
            logits = model(feat_t, mask_t)
            probs = F.softmax(logits, dim=-1)
            conf, preds = torch.max(probs, dim=-1)
            
            all_fingers.extend(preds[0, :seq_len].cpu().numpy().tolist())
            all_conf.extend(conf[0, :seq_len].cpu().numpy().tolist())
    
    return all_fingers, all_conf


def infer(img_input, checkpoint_path):
    """
    Main inference function.
    
    Args:
        img_input: single image path, list of paths, or folder path
        checkpoint_path: path to model checkpoint
    
    Returns:
        notes: ['C4', 'D4', ...] 
        coords: [(x, y, page), ...]
        fingers: [1, 2, 3, ...]
        confidences: [0.9, 0.85, ...]
    """
    # Extract notes
    print("Extracting notes...")
    
    if isinstance(img_input, list):
        notes, coords, bboxes = extract_from_pages(img_input)
    elif Path(img_input).is_dir():
        notes, coords, bboxes = extract_from_folder(img_input)
    else:
        notes, coords, bboxes = extract_from_image(img_input)
        coords = [(x, y, 0) for x, y in coords]
    
    if len(notes) == 0:
        return [], [], [], []
    
    print(f"Found {len(notes)} notes")
    
    # Load model
    print("Loading model...")
    model = load_model(checkpoint_path)
    
    # Convert to features
    features = notes_to_features(notes)
    
    # Predict
    print("Predicting fingerings...")
    fingers, confidences = predict_batch(model, features)
    
    return notes, coords, fingers, confidences


if __name__ == "__main__":
    import sys
    
    checkpoint = "checkpoints/best_model.pt"
    
    if not Path(checkpoint).exists():
        print(f"ERROR: Model not found at {checkpoint}")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python infer.py image.jpg")
        print("  python infer.py page1.jpg page2.jpg page3.jpg")
        print("  python infer.py ./folder_with_pages/")
        sys.exit(1)
    
    # Handle input
    if len(sys.argv) == 2:
        img_input = sys.argv[1]
    else:
        img_input = sys.argv[1:]
    
    # Run
    notes, coords, fingers, confidences = infer(img_input, checkpoint)
    
    print(f"\n{'='*50}")
    print(f"Results: {len(notes)} notes")
    print(f"{'='*50}\n")
    
    for i in range(min(20, len(notes))):
        x, y, page = coords[i]
        print(f"  {notes[i]:4s} @ ({x:4d}, {y:4d}) page {page} -> finger {fingers[i]} ({confidences[i]:.2f})")
    
    if len(notes) > 20:
        print(f"  ... and {len(notes) - 20} more")

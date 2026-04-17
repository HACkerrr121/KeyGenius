"""
KeyGenius Inference - predict fingerings from sheet music.
Uses oemer full pipeline for proper note extraction.
"""
import torch
import numpy as np
from pathlib import Path
from model import FingeringTransformer
from fast_oemer_extract import (
    extract_from_image, extract_from_pages, extract_from_folder,
    extract_from_musicxml, NOTE_TO_MIDI
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = FingeringTransformer(
        input_dim=17,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        num_fingers=6,
        class_weights=None
    )
    
    state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k != 'class_weights'}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    if 'val_acc' in checkpoint:
        print(f"  Val accuracy: {checkpoint['val_acc']*100:.1f}%")
    
    return model


def notes_to_features(notes_data):
    """
    Convert extracted notes to 17-dim model features.
    Features 12-16 (prev_finger) are left as zeros initially,
    then filled in during two-pass inference.
    """
    features = []
    n = len(notes_data)
    
    if n == 0:
        return np.array([], dtype=np.float32).reshape(0, 17)
    
    midis = [note['midi'] for note in notes_data]
    
    for i, note in enumerate(notes_data):
        midi = note['midi']
        
        # Basic features - now from proper OMR/MusicXML data
        midi_norm = (midi - 21) / 87.0
        duration = min(note.get('duration', 0.5), 2.0)
        delta_time = min(note.get('delta_time', 0.2), 2.0)
        
        # Intervals
        interval_prev = (midi - midis[i-1]) / 24.0 if i > 0 else 0.0
        interval_next = (midis[i+1] - midi) / 24.0 if i < n - 1 else 0.0
        interval_prev = np.clip(interval_prev, -1, 1)
        interval_next = np.clip(interval_next, -1, 1)
        
        # Direction
        if i > 0:
            direction = 1.0 if midi > midis[i-1] else (-1.0 if midi < midis[i-1] else 0.0)
        else:
            direction = 0.0
        
        # Chord features - from proper detection
        is_chord = 1.0 if note.get('is_chord', False) else 0.0
        chord_size_norm = min(note.get('chord_size', 1), 5) / 5.0
        chord_position = note.get('chord_position', 0.5)
        
        # Pattern detection
        if i >= 2:
            recent = [midis[j] - midis[j-1] for j in range(max(1, i-3), i+1)]
            steps = sum(1 for iv in recent if abs(iv) in [1, 2])
            arps = sum(1 for iv in recent if abs(iv) in [3, 4, 5])
            repeats = sum(1 for iv in recent if iv == 0)
            pattern_scale = steps / len(recent)
            pattern_arpeggio = arps / len(recent)
            pattern_repeat = repeats / len(recent)
        else:
            pattern_scale = 0.0
            pattern_arpeggio = 0.0
            pattern_repeat = 0.0
        
        # Previous finger - zeros for now, filled in during two-pass
        prev_finger_onehot = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        feature_vec = [
            midi_norm,
            duration,
            delta_time,
            interval_prev,
            interval_next,
            direction,
            is_chord,
            chord_size_norm,
            chord_position,
            pattern_scale,
            pattern_arpeggio,
            pattern_repeat,
            *prev_finger_onehot
        ]
        
        features.append(feature_vec)
    
    return np.array(features, dtype=np.float32)


def predict_batch(model, features, max_seq=200):
    """
    Two-pass prediction with prev_finger feedback.
    
    Pass 1: predict with prev_finger = zeros
    Pass 2: fill in prev_finger from pass 1 predictions, re-predict
    
    Only 2 forward passes per chunk instead of N.
    """
    n = len(features)
    all_fingers = []
    
    for i in range(0, n, max_seq):
        batch = features[i:i+max_seq].copy()
        seq_len = min(max_seq, n - i)
        
        if seq_len < max_seq:
            batch = np.pad(batch, ((0, max_seq - seq_len), (0, 0)), constant_values=0)
        
        mask = np.zeros(max_seq, dtype=np.float32)
        mask[:seq_len] = 1.0
        
        with torch.no_grad():
            feat_t = torch.from_numpy(batch).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
            pad_mask = (mask_t == 0)
            
            # Pass 1: predict with no prev_finger info
            preds = model.generate(feat_t, src_key_padding_mask=pad_mask, mask=mask_t)
            
            # Fill in prev_finger one-hot from pass 1 predictions
            for j in range(1, seq_len):
                prev_f = preds[0, j-1].item()
                onehot = [0.0] * 5
                if 1 <= prev_f <= 5:
                    onehot[prev_f - 1] = 1.0
                batch[j, 12:17] = onehot
            
            # Pass 2: re-predict with prev_finger context
            feat_t = torch.from_numpy(batch).unsqueeze(0).to(device)
            preds = model.generate(feat_t, src_key_padding_mask=pad_mask, mask=mask_t)
        
        all_fingers.extend(preds[0, :seq_len].cpu().numpy().tolist())
    
    return all_fingers


def infer(img_input, checkpoint_path):
    """
    Main inference function.
    
    Args:
        img_input: single image path, list of paths, folder path,
                   or .musicxml/.xml/.mxl file
        checkpoint_path: path to model checkpoint
    
    Returns:
        notes: ['C4', 'D4', ...] 
        coords: [(x, y, page), ...]
        fingers: [1, 2, 3, ...]
    """
    print("Extracting notes...")
    
    input_path = img_input if isinstance(img_input, str) else img_input[0]
    
    # Support MusicXML input directly (skip oemer)
    if isinstance(img_input, str) and input_path.endswith(('.musicxml', '.xml', '.mxl')):
        print("  (parsing MusicXML directly)")
        notes_data = extract_from_musicxml(img_input)
    elif isinstance(img_input, list):
        notes_data = extract_from_pages(img_input)
    elif Path(img_input).is_dir():
        notes_data = extract_from_folder(img_input)
    else:
        notes_data = extract_from_image(img_input)
    
    if len(notes_data) == 0:
        print("No notes found!")
        return [], [], []
    
    print(f"Found {len(notes_data)} notes")
    
    # Stats
    right_count = sum(1 for n in notes_data if n['hand'] == 'RIGHT')
    left_count = sum(1 for n in notes_data if n['hand'] == 'LEFT')
    chord_count = sum(1 for n in notes_data if n.get('is_chord', False))
    midis = [n['midi'] for n in notes_data]
    print(f"  {right_count} RIGHT hand, {left_count} LEFT hand, {chord_count} in chords")
    print(f"  Pitch range: MIDI {min(midis)}-{max(midis)}")
    
    print("Loading model...")
    model = load_model(checkpoint_path)
    
    print("Converting to features...")
    features = notes_to_features(notes_data)
    
    print("Predicting fingerings (two-pass)...")
    fingers = predict_batch(model, features)
    
    # Debug: show finger distribution
    from collections import Counter
    print(f"Finger distribution: {Counter(fingers)}")
    
    # Convert to old format for compatibility
    notes = [n['note'] for n in notes_data]
    coords = [(n['x'], n['y'], n.get('page', 0)) for n in notes_data]
    
    return notes, coords, fingers


if __name__ == "__main__":
    import sys
    
    checkpoint = "best_model_right.pth"
    
    if not Path(checkpoint).exists():
        print(f"ERROR: Model not found at {checkpoint}")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python inference.py image.jpg")
        print("  python inference.py score.musicxml")
        sys.exit(1)
    
    img_input = sys.argv[1]
    notes, coords, fingers = infer(img_input, checkpoint)
    
    print(f"\n{'='*50}")
    print(f"Results: {len(notes)} notes")
    print(f"{'='*50}\n")
    
    for i in range(min(20, len(notes))):
        x, y, page = coords[i]
        print(f"  {notes[i]:6s} @ ({x:4d}, {y:4d}) page {page} -> finger {fingers[i]}")
    
    if len(notes) > 20:
        print(f"  ... and {len(notes) - 20} more")
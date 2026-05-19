"""
KeyGenius Inference - predict fingerings from sheet music.
Uses Audiveris .omr file for pixel-accurate note head coordinates.
"""
import torch
import numpy as np
from pathlib import Path
from model import FingeringTransformer
from fast_oemer_extract import extract_from_musicxml
import zipfile
import subprocess
import tempfile
import os
from xml.etree import ElementTree as ET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_coords_from_omr(omr_path):
    """
    Extract pixel-accurate note head center coordinates from Audiveris .omr file.
    The .omr is a ZIP containing sheet#1/sheet#1.xml with <head> elements and bounds.
    Returns list of (cx, cy) sorted in reading order (top-to-bottom, left-to-right).
    """
    with zipfile.ZipFile(omr_path) as z:
        xml = z.read('sheet#1/sheet#1.xml').decode()

    root = ET.fromstring(xml)
    heads = []
    for el in root.iter('head'):
        b = el.find('bounds')
        if b is None:
            continue
        x = int(b.attrib['x'])
        y = int(b.attrib['y'])
        w = int(b.attrib['w'])
        h = int(b.attrib['h'])
        heads.append((x + w // 2, y + h // 2))

    # Sort in reading order: row band then x
    heads.sort(key=lambda p: (p[1] // 150, p[0]))
    return heads


def _find_audiveris_cmd():
    # Allow override via env var
    env = os.environ.get("AUDIVERIS_CMD")
    if env:
        return env
    # Linux wrapper script (Jetson)
    linux = os.path.expanduser("~/audiveris/audiveris/bin/Audiveris")
    if os.path.exists(linux):
        return linux
    # Mac app bundle via bundled Java
    mac_java = "/Applications/Audiveris.app/Contents/runtime/Contents/Home/bin/java"
    if os.path.exists(mac_java):
        return mac_java  # handled specially below
    raise RuntimeError("Audiveris not found. Set AUDIVERIS_CMD env var.")

_AUDIVERIS_CMD = _find_audiveris_cmd()
_MAC_JAVA = "/Applications/Audiveris.app/Contents/runtime/Contents/Home/bin/java"
_MAC_CP = "/Applications/Audiveris.app/Contents/app/*"

def _build_audiveris_args(output_dir, img_path):
    if _AUDIVERIS_CMD == _MAC_JAVA:
        return [
            _MAC_JAVA, "--enable-native-access=ALL-UNNAMED",
            "-cp", _MAC_CP, "Audiveris",
            "-batch", "-transcribe", "-export",
            "-output", output_dir, img_path,
        ]
    return [
        _AUDIVERIS_CMD,
        "-batch", "-transcribe", "-export",
        "-output", output_dir, img_path,
    ]


def extract_from_image_with_coords(img_path):
    """Run Audiveris to get MusicXML, then use pre-extracted coordinates"""
    img_path = str(Path(img_path).resolve())
    print(f"Running Audiveris to get musical data from {img_path}...")

    tmp_dir = tempfile.mkdtemp(prefix="audiveris_")
    result = subprocess.run(
        _build_audiveris_args(tmp_dir, img_path),
        capture_output=True,
        text=True,
        timeout=300,
    )

    # Show Audiveris output for debugging
    if result.stdout:
        print(result.stdout[-2000:])
    if result.stderr:
        print(result.stderr[-2000:])

    img_name = Path(img_path).stem
    xml_path = os.path.join(tmp_dir, f"{img_name}.mxl")

    if not os.path.exists(xml_path):
        files = os.listdir(tmp_dir)
        print(f"Files in tmp_dir: {files}")
        raise RuntimeError(f"Audiveris failed to generate MusicXML. tmp_dir contents: {files}")
    
    print(f"  MusicXML generated: {xml_path}")
    
    # Get musical data from MusicXML
    notes_data = extract_from_musicxml(xml_path)
    
    # Load pixel-accurate coordinates from Audiveris .omr
    omr_path = os.path.join(tmp_dir, f"{img_name}.omr")
    try:
        coords = extract_coords_from_omr(omr_path)
        print(f"  Loaded {len(coords)} coordinates from .omr")

        for i in range(min(len(coords), len(notes_data))):
            notes_data[i]['x'], notes_data[i]['y'] = coords[i]
            notes_data[i]['has_coord'] = True
        if len(coords) < len(notes_data):
            print(f"  WARNING: Only {len(coords)} coordinates for {len(notes_data)} notes")

    except Exception as e:
        print(f"  WARNING: Could not load .omr coordinates: {e}")
    
    return notes_data


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = FingeringTransformer(
        input_dim=12,
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
    """Convert extracted notes to 12-dim model features."""
    features = []
    n = len(notes_data)

    if n == 0:
        return np.array([], dtype=np.float32).reshape(0, 12)

    midis = [note['midi'] for note in notes_data]

    for i, note in enumerate(notes_data):
        midi = note['midi']

        midi_norm = (midi - 21) / 87.0
        duration = min(note.get('duration', 0.5), 2.0)
        delta_time = min(note.get('delta_time', 0.2), 2.0)

        interval_prev = (midi - midis[i-1]) / 24.0 if i > 0 else 0.0
        interval_next = (midis[i+1] - midi) / 24.0 if i < n - 1 else 0.0
        interval_prev = np.clip(interval_prev, -1, 1)
        interval_next = np.clip(interval_next, -1, 1)

        if i > 0:
            direction = 1.0 if midi > midis[i-1] else (-1.0 if midi < midis[i-1] else 0.0)
        else:
            direction = 0.0

        is_chord = 1.0 if note.get('is_chord', False) else 0.0
        chord_size_norm = min(note.get('chord_size', 1), 5) / 5.0
        chord_position = note.get('chord_position', 0.5)

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

        features.append([
            midi_norm, duration, delta_time,
            interval_prev, interval_next, direction,
            is_chord, chord_size_norm, chord_position,
            pattern_scale, pattern_arpeggio, pattern_repeat,
        ])

    return np.array(features, dtype=np.float32)


def predict_batch(model, features, max_seq=200):
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
            preds = model.generate(feat_t, src_key_padding_mask=pad_mask, mask=mask_t)

        all_fingers.extend(preds[0, :seq_len].cpu().numpy().tolist())

    return all_fingers


def infer(img_input, checkpoint_path):
    """Main inference function"""
    print("Extracting notes...")
    
    # Use pre-extracted coordinates
    notes_data = extract_from_image_with_coords(img_input)
    
    if len(notes_data) == 0:
        print("No notes found!")
        return [], [], []
    
    print(f"Found {len(notes_data)} notes")
    
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
    
    from collections import Counter
    print(f"Finger distribution: {Counter(fingers)}")
    
    # Only return notes with real pixel coordinates
    valid = [(n, f) for n, f in zip(notes_data, fingers) if n.get('has_coord', False)]
    if valid:
        notes_data, fingers = zip(*valid)
        print(f"  {len(notes_data)} notes with real coordinates")

    notes = [n['note'] for n in notes_data]
    coords = [(n['x'], n['y'], n.get('page', 0)) for n in notes_data]

    return notes, coords, list(fingers)


if __name__ == "__main__":
    import sys
    
    checkpoint = "best_model_right.pth"
    
    if not Path(checkpoint).exists():
        print(f"ERROR: Model not found at {checkpoint}")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py image.jpg")
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
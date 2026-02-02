import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

# Note name to MIDI number mapping
NOTE_TO_MIDI = {}
for octave in range(0, 9):
    for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
        midi_num = octave * 12 + i + 12
        NOTE_TO_MIDI[f"{note}{octave}"] = midi_num
        flat_map = {'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb'}
        if note in flat_map:
            NOTE_TO_MIDI[f"{flat_map[note]}{octave}"] = midi_num


def parse_fingering_file(filepath):
    lines = Path(filepath).read_text().strip().split("\n")
    
    notes = []
    for line in lines:
        if not line.strip() or line.startswith("//"):
            continue
        
        parts = line.split()
        if len(parts) < 8:
            continue
        
        try:
            idx = int(parts[0])
            start_time = float(parts[1])
            end_time = float(parts[2])
            note_name = parts[3]
            hand = int(parts[6])
            finger_raw = parts[7]
            
            if '_' in finger_raw:
                finger = int(finger_raw.split('_')[0])
            else:
                finger = int(finger_raw)
            
            finger = abs(finger)
            
            if finger < 1 or finger > 5:
                continue
            
            if note_name not in NOTE_TO_MIDI:
                continue
            midi_num = NOTE_TO_MIDI[note_name]
            
            notes.append({
                'idx': idx,
                'start': start_time,
                'end': end_time,
                'note': note_name,
                'midi': midi_num,
                'hand': hand,
                'finger': finger,
                'duration': end_time - start_time
            })
        except (ValueError, IndexError):
            continue
    
    return notes


def extract_sequences(notes, hand, max_seq_len=200):
    hand_notes = [n for n in notes if n['hand'] == hand]
    
    if len(hand_notes) == 0:
        return []
    
    hand_notes = sorted(hand_notes, key=lambda x: x['start'])
    
    sequences = []
    for i in range(0, len(hand_notes), max_seq_len):
        seq = hand_notes[i:i + max_seq_len]
        if len(seq) >= 10:
            sequences.append(seq)
    
    return sequences


def encode_sequence(seq):
    features = []
    fingers = []
    
    prev_midi = None
    prev_time = None
    
    for note in seq:
        midi = note['midi']
        start = note['start']
        dur = note['duration']
        finger = note['finger']
        
        midi_norm = (midi - 21) / 87.0
        
        if prev_time is not None:
            delta = start - prev_time
        else:
            delta = 0.0
        delta = min(delta, 2.0)
        
        if prev_midi is not None:
            interval = midi - prev_midi
        else:
            interval = 0
        interval_norm = interval / 24.0
        
        if prev_midi is not None:
            if midi > prev_midi:
                direction = 1.0
            elif midi < prev_midi:
                direction = -1.0
            else:
                direction = 0.0
        else:
            direction = 0.0
        
        features.append([
            midi_norm,
            min(dur, 2.0),
            delta,
            interval_norm,
            direction
        ])
        
        fingers.append(finger)
        
        prev_midi = midi
        prev_time = start
    
    return np.array(features, dtype=np.float32), np.array(fingers, dtype=np.int64)


class FingeringDataset(Dataset):
    def __init__(self, data_dir, hand=0, max_seq_len=200, split='train', val_ratio=0.2):
        self.max_seq_len = max_seq_len
        self.hand = hand
        
        finger_files = sorted(Path(data_dir).glob("*.txt"))
        print(f"Found {len(finger_files)} fingering files")
        
        all_sequences = []
        for f in finger_files:
            notes = parse_fingering_file(f)
            sequences = extract_sequences(notes, hand, max_seq_len)
            all_sequences.extend(sequences)
        
        print(f"Extracted {len(all_sequences)} sequences for {'right' if hand == 0 else 'left'} hand")
        
        np.random.seed(42)
        indices = np.random.permutation(len(all_sequences))
        split_idx = int(len(indices) * (1 - val_ratio))
        
        if split == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
        
        self.sequences = all_sequences
        print(f"{split.capitalize()} set: {len(self.indices)} sequences")
        
        all_fingers = []
        for i in self.indices:
            seq = self.sequences[i]
            all_fingers.extend([n['finger'] for n in seq])
        
        from collections import Counter
        dist = Counter(all_fingers)
        print(f"Finger distribution: {dict(sorted(dist.items()))}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        seq = self.sequences[self.indices[idx]]
        features, fingers = encode_sequence(seq)
        
        seq_len = len(features)
        
        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
            fingers = np.pad(fingers, (0, pad_len), mode='constant', constant_values=0)
        
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:seq_len] = 1.0
        
        return (
            torch.from_numpy(features),
            torch.from_numpy(fingers),
            torch.from_numpy(mask),
        )


if __name__ == "__main__":
    data_dir = "Music_Data/FingeringFiles"
    
    print("=" * 60)
    print("RIGHT HAND")
    print("=" * 60)
    ds_right = FingeringDataset(data_dir, hand=0, split='train')
    
    print("\n" + "=" * 60)
    print("LEFT HAND")
    print("=" * 60)
    ds_left = FingeringDataset(data_dir, hand=1, split='train')
    
    print("\n" + "=" * 60)
    print("SAMPLE")
    print("=" * 60)
    features, fingers, mask = ds_right[0]
    print(f"Features shape: {features.shape}")
    print(f"Fingers shape: {fingers.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Actual sequence length: {int(mask.sum().item())}")
    print(f"First 10 fingers: {fingers[:10].tolist()}")
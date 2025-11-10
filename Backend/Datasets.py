from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
import torch

MAX_NOTES = 350

# -------------------------
# Step 1: Load and parse text files
# -------------------------

folder = Path("Music_Data/FingeringFiles")
txt_files = list(folder.glob("*.txt"))
if not txt_files:
    raise FileNotFoundError(f"No .txt files found in {folder.absolute()}")

y_labels_cnn_notes = []
y_labels_cnn_hand = []
y_labels_cnn_time_stamp = []
y_labels_rnn = []

for file_path in txt_files:
    lines = file_path.read_text().strip().split("\n")

    notes, hands, times, rnns = [], [], [], []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) > 6:
            notes.append(parts[3])
            hands.append(parts[6])
            times.append(parts[2])
            rnns.append(parts[-1])
    y_labels_cnn_notes.append(notes)
    y_labels_cnn_hand.append(hands)
    y_labels_cnn_time_stamp.append(times)
    y_labels_rnn.append(rnns)

# -------------------------
# Step 2: Padding with 999 placeholders
# -------------------------
for i in range(len(y_labels_cnn_hand)):
    n = MAX_NOTES - len(y_labels_cnn_hand[i])
    if n > 0:
        pad = [999] * n
        y_labels_cnn_hand[i] += pad
        y_labels_cnn_notes[i] += pad
        y_labels_cnn_time_stamp[i] += pad
        y_labels_rnn[i] += pad

# -------------------------
# Step 3: Build mappings
# -------------------------

all_notes_flat = [note for seq in y_labels_cnn_notes for note in seq if note != 999]
unique_notes = np.unique(all_notes_flat)
note_map = {note: idx for idx, note in enumerate(unique_notes)}

rnn_switch_unique = [f"{i}_{z}" for i in range(-10, 10) for z in range(-10, 10)]
rnn_map = {rnn_switch_unique[i]: i for i in range(len(rnn_switch_unique))}

# -------------------------
# Step 4: Convert data to numeric format
# -------------------------
for i in range(len(y_labels_cnn_notes)):
    for z in range(len(y_labels_cnn_hand[i])):
        # Convert hand
        if y_labels_cnn_hand[i][z] != 999:
            y_labels_cnn_hand[i][z] = abs(int(y_labels_cnn_hand[i][z]))

        # Convert timestamp
        if y_labels_cnn_time_stamp[i][z] != 999:
            try:
                y_labels_cnn_time_stamp[i][z] = float(y_labels_cnn_time_stamp[i][z])
            except ValueError:
                y_labels_cnn_time_stamp[i][z] = 999.0

        # Convert RNN label
        val = y_labels_rnn[i][z]
        if val == 999:
            y_labels_rnn[i][z] = 999
        elif val in rnn_map:
            y_labels_rnn[i][z] = rnn_map[val]
        elif isinstance(val, str) and val.endswith('_'):
            try:
                y_labels_rnn[i][z] = abs(int(val.rstrip('_')))
            except ValueError:
                y_labels_rnn[i][z] = 999
        else:
            try:
                y_labels_rnn[i][z] = abs(int(val))
            except ValueError:
                y_labels_rnn[i][z] = 999

        # Convert note
        val_note = y_labels_cnn_notes[i][z]
        y_labels_cnn_notes[i][z] = note_map.get(val_note, 999)

# -------------------------
# Step 5: Define Datasets
# -------------------------

class ImageDataset(Dataset):
    def __init__(self, transforms=None):
        self.images = list(Path("Music_Data/Scores").glob("*.jpg"))
        if not self.images:
            raise FileNotFoundError("No images found in Music_Data/Scores/")
        self.hand = y_labels_cnn_hand
        self.notes = y_labels_cnn_notes
        self.time = y_labels_cnn_time_stamp
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        idx = index % len(self.hand)

        labels = [self.hand[idx], self.notes[idx], self.time[idx]]

        # Truncate or pad each label sequence
        padded_labels = []
        for seq in labels:
            seq = seq[:MAX_NOTES]
            seq += [999] * max(0, MAX_NOTES - len(seq))
            padded_labels.append(seq)

        img = read_image(str(self.images[index])).float() / 255.0
        if self.transforms:
            img = self.transforms(img)
        return img, torch.tensor(padded_labels)


class NoteDataset(Dataset):
    def __init__(self):
        self.hand = y_labels_cnn_hand
        self.notes = y_labels_cnn_notes
        self.time = y_labels_cnn_time_stamp
        self.labels = y_labels_rnn

    def __len__(self):
        return len(self.hand)

    def __getitem__(self, index):
        def pad(seq):
            seq = seq[:MAX_NOTES]
            seq += [999] * max(0, MAX_NOTES - len(seq))
            return torch.tensor(seq)

        hand = pad(self.hand[index])
        notes = pad(self.notes[index])
        time = pad(self.time[index])
        labels = pad(self.labels[index])

        x = torch.stack([hand, notes, time], dim=1)
        return x, labels


# -------------------------
# Debug check
# -------------------------
if __name__ == "__main__":
    print(f"✅ Loaded {len(txt_files)} fingering files")
    print(f"✅ {len(y_labels_cnn_notes)} label groups prepared")

    try:
        img_ds = ImageDataset()
        print(f"✅ {len(img_ds)} images loaded")
    except FileNotFoundError as e:
        print(f"⚠️ {e}")

    note_ds = NoteDataset()
    sample_x, sample_y = note_ds[0]
    print(f"Sample input shape: {sample_x.shape}, labels shape: {sample_y.shape}")

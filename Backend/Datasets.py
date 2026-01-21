from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
import json

MAX_NOTES = 350
NUM_NOTE_CLASSES = 128


folder = Path("Music_Data/FingeringFiles")
txt_files = sorted(list(folder.glob("*.txt")))
if not txt_files:
    raise FileNotFoundError(f"No .txt files found in {folder.absolute()}")

# Load coordinates
coords_path = Path("Music_Data/coordinates.txt")
if coords_path.exists():
    with open(coords_path) as f:
        content = f.read()
    
    try:
        all_coords = json.loads(content)
    except json.JSONDecodeError:
        import ast
        all_coords = ast.literal_eval(content)
    print(f"✅ Loaded coordinates for {len(all_coords)} images")
else:
    print("⚠️ WARNING: coordinates.txt not found, using empty coords")
    all_coords = {}

y_labels_cnn_notes = []
y_labels_cnn_hand = []
y_labels_cnn_time_stamp = []
y_labels_rnn = []
y_labels_coords = []
file_names = []

for file_path in txt_files:
    lines = file_path.read_text().strip().split("\n")
    
    img_name = file_path.stem + ".jpg"
    coords = all_coords.get(img_name, [])
    file_names.append(img_name)

    notes, hands, times, rnns = [], [], [], []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) > 6:
            notes.append(parts[3])
            hands.append(parts[6])
            times.append(parts[2])
            rnns.append(parts[3])
    
    y_labels_cnn_notes.append(notes)
    y_labels_cnn_hand.append(hands)
    y_labels_cnn_time_stamp.append(times)
    y_labels_rnn.append(rnns)
    y_labels_coords.append(coords)

if not y_labels_cnn_notes:
    raise ValueError("No valid data loaded from fingering files")


for i in range(len(y_labels_cnn_hand)):
    n = MAX_NOTES - len(y_labels_cnn_hand[i])
    if n > 0:
        y_labels_cnn_hand[i] += [999] * n
        y_labels_cnn_notes[i] += [999] * n
        y_labels_cnn_time_stamp[i] += [0] * n
        y_labels_rnn[i] += [999] * n
    
    coord_pad = MAX_NOTES - len(y_labels_coords[i])
    if coord_pad > 0:
        y_labels_coords[i] += [[0, 0, 0, 0]] * coord_pad
    y_labels_coords[i] = y_labels_coords[i][:MAX_NOTES]

all_notes_flat = [note for seq in y_labels_cnn_notes for note in seq if note != 999]
unique_notes = sorted(list(set(all_notes_flat)))
note_map = {note: idx for idx, note in enumerate(unique_notes)}

print(f"Found {len(unique_notes)} unique notes")

for i in range(len(y_labels_cnn_notes)):
    for z in range(len(y_labels_cnn_hand[i])):
        if y_labels_cnn_hand[i][z] != 999:
            try:
                hand_val = abs(int(y_labels_cnn_hand[i][z]))
                y_labels_cnn_hand[i][z] = min(hand_val, 10)
            except:
                y_labels_cnn_hand[i][z] = 999

        if y_labels_cnn_time_stamp[i][z] != 0:
            try:
                y_labels_cnn_time_stamp[i][z] = float(y_labels_cnn_time_stamp[i][z])
            except:
                y_labels_cnn_time_stamp[i][z] = 0.0

        val_note = y_labels_cnn_notes[i][z]
        if val_note == 999:
            y_labels_cnn_notes[i][z] = 999
        else:
            note_idx = note_map.get(val_note, 0)
            y_labels_cnn_notes[i][z] = min(note_idx, NUM_NOTE_CLASSES - 1)

        val_rnn = y_labels_rnn[i][z]
        if val_rnn == 999:
            y_labels_rnn[i][z] = 999
        else:
            rnn_idx = note_map.get(val_rnn, 0)
            y_labels_rnn[i][z] = min(rnn_idx, NUM_NOTE_CLASSES - 1)


def verify_data_ranges():
    for i in range(len(y_labels_cnn_hand)):
        for j in range(len(y_labels_cnn_hand[i])):
            hand_val = y_labels_cnn_hand[i][j]
            if hand_val != 999 and (hand_val < 0 or hand_val > 10):
                y_labels_cnn_hand[i][j] = 999
            
            note_val = y_labels_cnn_notes[i][j]
            if note_val != 999 and (note_val < 0 or note_val >= NUM_NOTE_CLASSES):
                y_labels_cnn_notes[i][j] = 999
            
            rnn_val = y_labels_rnn[i][j]
            if rnn_val != 999 and (rnn_val < 0 or rnn_val >= NUM_NOTE_CLASSES):
                y_labels_rnn[i][j] = 999

verify_data_ranges()



image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=2, fill=255),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



def create_coord_mask(img_size, bboxes, target_size=(224, 224)):
    h, w = img_size
    mask = np.zeros((h, w), dtype=np.float32)
    
    if not bboxes:
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(target_size, Image.NEAREST)
        return torch.tensor(np.array(mask_pil).astype(np.float32) / 255.0).unsqueeze(0)
    
    for bbox in bboxes:
        if not bbox or bbox == [0, 0, 0, 0]:
            continue
        x1, y1, x2, y2 = bbox
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
    
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    mask_pil = mask_pil.resize(target_size, Image.NEAREST)
    mask = np.array(mask_pil).astype(np.float32) / 255.0
    
    return torch.tensor(mask).unsqueeze(0)


class ImageDataset(Dataset):
    def __init__(self, transforms=None):
        self.images = sorted(list(Path("Music_Data/Scores").glob("*.jpg")))
        if not self.images:
            raise FileNotFoundError("No images found in Music_Data/Scores/")
        self.hand = y_labels_cnn_hand
        self.notes = y_labels_cnn_notes
        self.time = y_labels_cnn_time_stamp
        self.coords = y_labels_coords
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        idx = index % len(self.hand)
        
        labels = torch.zeros((3, MAX_NOTES), dtype=torch.float32)
        labels[0, :] = torch.tensor(self.hand[idx][:MAX_NOTES], dtype=torch.float32)
        labels[1, :] = torch.tensor(self.notes[idx][:MAX_NOTES], dtype=torch.float32)
        labels[2, :] = torch.tensor(self.time[idx][:MAX_NOTES], dtype=torch.float32)
        
        coords_target = torch.tensor(self.coords[idx][:MAX_NOTES], dtype=torch.float32)

        img = Image.open(self.images[index]).convert("RGB")
        orig_size = img.size[::-1]
        
        coord_mask = create_coord_mask(orig_size, self.coords[idx])
        
        if self.transforms:
            img = self.transforms(img)
        else:
            img = val_transforms(img)
        
        coord_mask = torch.nn.functional.interpolate(
            coord_mask.unsqueeze(0), size=(img.shape[1], img.shape[2]), mode='nearest'
        ).squeeze(0)
        
        img = torch.cat([img, coord_mask], dim=0)

        return img, labels, coords_target


class NoteDataset(Dataset):
    def __init__(self):
        self.hand = y_labels_cnn_hand
        self.notes = y_labels_cnn_notes
        self.time = y_labels_cnn_time_stamp
        self.labels = y_labels_rnn

    def __len__(self):
        return len(self.hand)

    def __getitem__(self, index):
        x = torch.zeros((MAX_NOTES, 3), dtype=torch.float32)
        x[:, 0] = torch.tensor(self.hand[index][:MAX_NOTES], dtype=torch.float32)
        x[:, 1] = torch.tensor(self.notes[index][:MAX_NOTES], dtype=torch.float32)
        x[:, 2] = torch.tensor(self.time[index][:MAX_NOTES], dtype=torch.float32)
        
        labels = torch.tensor(self.labels[index][:MAX_NOTES], dtype=torch.float32)
        
        return x, labels


# -------------------------
# Debug check
# -------------------------
if __name__ == "__main__":
    print(f"✅ Loaded {len(txt_files)} fingering files")
    print(f"✅ {len(y_labels_cnn_notes)} label groups prepared")

    try:
        img_ds = ImageDataset(transforms=image_transforms)
        print(f"✅ {len(img_ds)} images loaded")
        
        sample_img, sample_labels, sample_coords = img_ds[0]
        print(f"Sample image shape: {sample_img.shape}")
        print(f"Sample labels shape: {sample_labels.shape}")
        print(f"Sample coords shape: {sample_coords.shape}")
    except Exception as e:
        print(f"⚠️ {e}")

    note_ds = NoteDataset()
    sample_x, sample_y = note_ds[0]
    print(f"RNN input shape: {sample_x.shape}, labels shape: {sample_y.shape}")
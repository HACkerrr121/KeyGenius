import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from CNN_RNN import CNN_RNN
from Datasets import ImageDataset, NoteDataset

# ------------------------------
# DEVICE
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------------------
# HYPERPARAMETERS
# ------------------------------
MAX_NOTES = 350
NUM_HAND_CLASSES = 11
NUM_NOTE_CLASSES = 128
IMG_WIDTH = 1653
IMG_HEIGHT = 2339
LR = 1e-3
EPOCHS = 6000
BATCH_SIZE = 16

# RNN parameters
RNN_HIDDEN_SIZE = 64
RNN_LAYERS = 3

# ------------------------------
# IMAGE TRANSFORMS
# ------------------------------
image_transforms = transforms.Compose([
    transforms.RandomCrop((224, 224), padding=10),
    transforms.RandomRotation(degrees=2, fill=255),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.002, 0.01), value=0.5)
])

# ------------------------------
# DATASETS & LOADERS
# ------------------------------
dataset_CNN = DataLoader(
    ImageDataset(transforms=image_transforms),
    batch_size=BATCH_SIZE,
    shuffle=True
)

dataset_RNN = DataLoader(
    NoteDataset(),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ------------------------------
# MODEL & OPTIMIZER
# ------------------------------
cnn_params = {
    "input_channels": 3,
    "max_sequence": MAX_NOTES,
    "num_hand_classes": NUM_HAND_CLASSES,
    "num_note_classes": NUM_NOTE_CLASSES,
    "note_weight": 1.0,
    "time_stamp_weight": 0.01,
    "hand_weight": 1.0,
    "rnn_weight": 1.0
}

rnn_params = {
    "hidden_size": RNN_HIDDEN_SIZE,
    "num_layers": RNN_LAYERS,
    "output_size": NUM_NOTE_CLASSES
}

model = CNN_RNN(cnn_params=cnn_params, rnn_params=rnn_params).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ------------------------------
# TRAINING LOOP
# ------------------------------
for epoch in range(EPOCHS):
    cnn_total_loss = 0.0
    rnn_total_loss = 0.0
    model.train()

    # --- Train on images (CNN) ---
    for batch_idx, (images, labels) in enumerate(dataset_CNN):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss, _ = model(images, labels)
        loss.backward()
        optimizer.step()

        cnn_total_loss += loss.item()

    # --- Train on sequences (RNN) ---
    for batch_idx, (seq_inputs, seq_labels) in enumerate(dataset_RNN):
        seq_inputs = seq_inputs.to(device)
        seq_labels = seq_labels.to(device)

        optimizer.zero_grad()
        loss, _ = model(seq_inputs, seq_labels)
        loss.backward()
        optimizer.step()

        rnn_total_loss += loss.item()

    # --- Logging ---
    if epoch % 100 == 0:
        avg_cnn_loss = cnn_total_loss / len(dataset_CNN)
        avg_rnn_loss = rnn_total_loss / len(dataset_RNN)

        print("=" * 80)
        print(f"🎵 EPOCH {epoch:4d}/{EPOCHS} 🎵")
        print(f"📊 CNN Loss: {avg_cnn_loss:.4f} | RNN Loss: {avg_rnn_loss:.4f}")
        print(f"⚡ LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("=" * 80)

# ------------------------------
# SAVE CHECKPOINT
# ------------------------------
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'cnn_rnn_checkpoint.pth')

print("✅ Joint CNN+RNN checkpoint saved successfully!")

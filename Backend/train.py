import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from RNN import RNN
from CNN import CNN
from Datasets import ImageDataset, NoteDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

### HYPERPARAMETERS ###
MAX_NOTES = 350
h_layer = 64
input_size = (MAX_NOTES, 3)
output_size = 10 
img_width = 1653
img_height = 2339
out_cnn = (MAX_NOTES, 3)
lr = 1e-3
epochs = 1

image_transforms = transforms.Compose([                
  transforms.RandomCrop((224, 224), padding=10),   
  transforms.RandomRotation(degrees=2, fill=255), 
  transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),  
  transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)), 
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  transforms.RandomErasing(p=0.2, scale=(0.002, 0.01), value=0.5)
])

# Model parameters
input_size = (MAX_NOTES, 3)
output_size = 400
h_layer = 64
num_hand_classes = 11
num_note_classes = 128

# Initialize models
model_RNN = RNN(input_size, output_size, h_layer).to(device)
optim_RNN = torch.optim.AdamW(model_RNN.parameters(), lr=lr)

model_CNN = CNN(
   input_channels=3,
   max_sequence=MAX_NOTES,
   num_hand_classes=num_hand_classes,
   num_note_classes=num_note_classes,
   note_weight=1.0,
   time_stamp_weight=0.01,
   hand_weight=1.0
).to(device)
optim_CNN = torch.optim.AdamW(model_CNN.parameters(), lr=lr)

# Load checkpoint if it exists
checkpoint_path = "model_weights.pth"
start_epoch = 0

if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
    try:
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model_CNN.load_state_dict(checkpoint['model_CNN_state_dict'])
        model_RNN.load_state_dict(checkpoint['model_RNN_state_dict'])
        optim_CNN.load_state_dict(checkpoint['optim_CNN_state_dict'])
        optim_RNN.load_state_dict(checkpoint['optim_RNN_state_dict'])
        
        # If you want to track epochs across runs, save/load epoch count too
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        
        print(f"✅ Checkpoint loaded successfully! Resuming from epoch {start_epoch}")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("Starting training from scratch...")
        start_epoch = 0
else:
    print("No checkpoint found. Starting training from scratch...")

# Create data loaders
dataset_CNN = DataLoader(
   ImageDataset(transforms=image_transforms),
   batch_size=16,
   shuffle=True
)

dataset_RNN = DataLoader(
   NoteDataset(),
   batch_size=16,
   shuffle=True
)

# Training loop
for epoch in range(start_epoch, start_epoch + epochs):
    cnn_total_loss = 0
    rnn_total_loss = 0

    # Train CNN
    model_CNN.train()
    for batch_idx, (images, labels) in enumerate(dataset_CNN):
        images = images.to(device)
        labels = labels.to(device)
        
        optim_CNN.zero_grad()
        loss, logits = model_CNN(images, labels)
        loss.backward()
        optim_CNN.step()

        cnn_total_loss += loss.item()
    
    # Train RNN
    model_RNN.train()
    for batch_idx, (x_train, labels) in enumerate(dataset_RNN):
        x_train = x_train.to(device)
        labels = labels.to(device)

        optim_RNN.zero_grad()
        loss, logits = model_RNN(x_train, labels)
        loss.backward()
        optim_RNN.step()

        rnn_total_loss += loss.item()

    # Print progress
    if epoch % 1 == 0:
        avg_cnn_loss = cnn_total_loss / len(dataset_CNN)
        avg_rnn_loss = rnn_total_loss / len(dataset_RNN)
        
        print("=" * 80)
        print(f"🎵 EPOCH {epoch:4d}/{start_epoch + epochs - 1} 🎵")
        print(f"📊 Losses → CNN: {avg_cnn_loss:.4f} | RNN: {avg_rnn_loss:.4f}")
        print(f"⚡ Learning Rate: {optim_CNN.param_groups[0]['lr']:.6f}")
        print("=" * 80)

print('Training completed!')

# Save checkpoint
print('Saving checkpoint...')
torch.save({
    'epoch': start_epoch + epochs - 1,  # Save the last epoch number
    'model_CNN_state_dict': model_CNN.state_dict(),
    'model_RNN_state_dict': model_RNN.state_dict(),
    'optim_CNN_state_dict': optim_CNN.state_dict(),
    'optim_RNN_state_dict': optim_RNN.state_dict()
}, checkpoint_path)

print('✅ Checkpoint saved successfully!')
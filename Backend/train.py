import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from RNN import RNN
from CNN import CNN
from Datasets import ImageDataset, NoteDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

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
epochs = 6000


image_transforms = transforms.Compose([                
  transforms.RandomCrop((224, 224), padding=10),   
  transforms.RandomRotation(degrees=2, fill=255), 
  transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),  
  transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)), 
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  transforms.RandomErasing(p=0.2, scale=(0.002, 0.01), value=0.5)
])


input_size = (MAX_NOTES, 3)
output_size = 400
h_layer = 64

model_RNN = RNN(input_size, output_size, h_layer).to(device)
optim_RNN = torch.optim.AdamW(model_RNN.named_parameters(), lr)

num_hand_classes = 11  # or however many hand positions (0-9 from your data)
num_note_classes = 128  # or len(unique) from your dataset preprocessing

model_CNN = CNN(
   input_channels=3,  # RGB images
   max_sequence=MAX_NOTES,  # 350
   num_hand_classes=num_hand_classes,
   num_note_classes=num_note_classes,
   note_weight=1.0,
   time_stamp_weight=0.01,
   hand_weight=1.0
).to(device)
optim_CNN = torch.optim.AdamW(model_CNN.named_parameters(), lr)


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


for epoch in range(epochs):


   cnn_total_loss = 0
   rnn_total_loss = 0


   model_CNN.train()

   for batch_idx, (images, labels) in enumerate(dataset_CNN):
       images = images.to(device)
       labels = labels.to(device)
       
       optim_CNN.zero_grad()
       loss,logits = model_CNN(images, labels)
       loss.backward()
       optim_CNN.step()

       cnn_total_loss += loss.item()
   
   model_RNN.train()

   for batch_idx, (x_train, labels) in enumerate(dataset_RNN):
       x_train = x_train.to(device)
       labels = labels.to(device)

       optim_RNN.zero_grad()
       loss, logits = model_RNN(x_train, labels)
       loss.backward()
       optim_RNN.step()

       rnn_total_loss += loss.item()


   if epoch % 100 == 0:
       # Calculate average losses
       avg_cnn_loss = cnn_total_loss / len(dataset_CNN)
       avg_rnn_loss = rnn_total_loss / len(dataset_RNN)
       
       print("=" * 80)
       print(f"🎵 EPOCH {epoch:4d}/{epochs} 🎵")
       print(f"📊 Losses → CNN: {avg_cnn_loss:.4f} | RNN: {avg_rnn_loss:.4f}")
       print(f"⚡ Learning Rate: {optim_CNN.param_groups[0]['lr']:.6f}")
       print("=" * 80)

   


checkpoint = torch.load('model_checkpoint.pth')
model_CNN.load_state_dict(checkpoint['model_CNN_state_dict'])
model_RNN.load_state_dict(checkpoint['model_RNN_state_dict'])
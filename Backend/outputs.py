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
from PIL import Image
from torchvision.io import read_image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


MAX_NOTES = 350
h_layer = 64
input_size = (MAX_NOTES, 3)
output_size = 10 
img_width = 1653
img_height = 2339
out_cnn = (MAX_NOTES, 3)
lr = 1e-3
epochs = 1


input_size = (MAX_NOTES, 3)
output_size = 400
h_layer = 64

model_RNN = RNN(input_size, output_size, h_layer).to(device)

num_hand_classes = 11 
num_note_classes = 128  
model_CNN = CNN(
   input_channels=3,  # RGB images
   max_sequence=MAX_NOTES,  # 350
   num_hand_classes=num_hand_classes,
   num_note_classes=num_note_classes,
   note_weight=1.0,
   time_stamp_weight=0.01,
   hand_weight=1.0
).to(device)


checkpoint = torch.load("model_weights.pth", map_location=device)
model_CNN.load_state_dict(checkpoint['model_CNN_state_dict'])
model_RNN.load_state_dict('model_RNN_state_dict')

model_CNN.eval()
model_RNN.eval()



def call(img):
    img = Image.new(img)
    img.resize(1653, 2339)
    img = read_image(img).float() / 255.0
    logits_cnn = model_CNN(img)

    logits_rnn = model_RNN(logits_cnn)

    return logits_rnn
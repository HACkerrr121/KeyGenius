from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision.io import read_image
import torch

MAX_NOTES = 350

folder = Path("Music_Data/FingeringFiles")
txt_files = list(folder.glob("*.txt"))
x = []
y_labels_cnn_notes = [[] for i in range(len(txt_files))]
y_labels_cnn_hand = [[] for i in range(len(txt_files))]
y_labels_cnn_time_stamp = [[] for i in range(len(txt_files))]
y_labels_rnn = [[] for i in range(len(txt_files))]


for file_idx, file_path in enumerate(txt_files):
  txt = file_path.read_text()
  lines = txt.split("\n")
  for line_num, line in enumerate(lines):
      if line.strip(): 
          line_parts = line.split("\t")
          
          if len(line_parts) > 3: 
              y_labels_cnn_notes[file_idx].append(line_parts[3])
              y_labels_cnn_hand[file_idx].append(line_parts[6])
              y_labels_cnn_time_stamp[file_idx].append(line_parts[2])
              y_labels_rnn[file_idx].append(line_parts[-1])


for i in range(len(y_labels_cnn_hand)):
 randoms = [999 for z in range(i)]
 y_labels_cnn_hand[i] += randoms
 y_labels_cnn_notes[i] += randoms
 y_labels_cnn_time_stamp[i] += randoms
 y_labels_rnn[i] += randoms


all_notes_flat = [note for sublist in y_labels_cnn_notes for note in sublist]
unique = np.unique(all_notes_flat)
map = [(unique[i], i) for i in range(len(unique))]
map = dict(map)
#print(y_labels_cnn_notes[:2])
rnn_switch_unique = [str(i)+"_"+str(z) for i in range(-10,10) for z in range(-10, 10)]
rnn_map = dict([(rnn_switch_unique[i], i) for i in range(len(rnn_switch_unique))])

for i in range(len(y_labels_cnn_notes)):
   for z in range(len(y_labels_cnn_hand[i])):
       y_labels_cnn_hand[i][z] = abs(int(y_labels_cnn_hand[i][z]))



       y_labels_cnn_time_stamp[i][z] = float(y_labels_cnn_time_stamp[i][z])




       if y_labels_rnn[i][z] in rnn_map.keys():
          y_labels_rnn[i][z] = rnn_map.get(y_labels_rnn[i][z])
       elif isinstance(y_labels_rnn[i][z], str) and y_labels_rnn[i][z].endswith('_'):
           y_labels_rnn[i][z] = int(y_labels_rnn[i][z].rstrip('_'))
       else:
           y_labels_rnn[i][z] = int(y_labels_rnn[i][z])
       y_labels_rnn[i][z] = abs(y_labels_rnn[i][z])



       y_labels_cnn_notes[i][z] = map.get(y_labels_cnn_notes[i][z])
       if y_labels_cnn_notes[i][z] == None:
           y_labels_cnn_notes[i][z] = 999


print(y_labels_rnn)



class ImageDataset(Dataset):
  
  def __init__(self, transforms=None):
      super().__init__()

      self.images = list(Path("Music_Data/Scores").glob("*.jpg"))
      print(len(self.images))

      self.labels = [y_labels_cnn_hand, y_labels_cnn_notes, y_labels_cnn_time_stamp]
      
      self.transforms = transforms
  
  def __len__(self):
      return len(self.images)

  def __getitem__(self, index):

      labels = [self.labels[i][index % len(self.labels[i])] for i in range(len(self.labels))]
      
      padded_labels = []
      for label_list in labels:
          if len(label_list) < MAX_NOTES:
              padded = label_list + [999] * (MAX_NOTES - len(label_list))
          else:
              padded = label_list[:MAX_NOTES]
          padded_labels.append(padded)
      
      img = self.get_image(index)
      return img, torch.tensor(padded_labels)
  
  def get_image(self, idx):
      image = read_image(str(self.images[idx])).float() / 255.0
      if self.transforms:
          image = self.transforms(image)
      return image


class NoteDataset(Dataset):

  def __init__(self):
      super().__init__()

      self.hand = y_labels_cnn_hand
      self.notes = y_labels_cnn_notes
      self.time = y_labels_cnn_time_stamp
      self.labels = y_labels_rnn
  
  def __len__(self):
      return len(self.hand)

  def __getitem__(self, index):
      hand_padded = self.hand[index][:MAX_NOTES] + [999] * max(0, MAX_NOTES - len(self.hand[index]))
      notes_padded = self.notes[index][:MAX_NOTES] + [999] * max(0, MAX_NOTES - len(self.notes[index]))
      time_padded = self.time[index][:MAX_NOTES] + [999] * max(0, MAX_NOTES - len(self.time[index]))
      labels_padded = self.labels[index][:MAX_NOTES] + [999] * max(0, MAX_NOTES - len(self.labels[index]))
      
      return torch.stack([torch.tensor(hand_padded), torch.tensor(notes_padded), torch.tensor(time_padded)], dim=1), torch.tensor(labels_padded)
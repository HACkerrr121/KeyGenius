# debug_inference.py
import torch
import numpy as np
from model import FingeringTransformer
from Datasets import FingeringDataset, encode_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
checkpoint = torch.load('best_model_right.pth', map_location=device, weights_only=False)
model = FingeringTransformer(
    input_dim=17, d_model=256, nhead=8, num_layers=6,
    dim_feedforward=1024, dropout=0.0, num_fingers=6, class_weights=None
)
state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k != 'class_weights'}
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

# Load a validation sequence (model hasn't seen this piece)
val_ds = FingeringDataset("Music_Data/FingeringFiles", hand=0, split='val')
features, true_fingers, mask = val_ds[0]

print("True fingers (first 50):")
print(true_fingers[:50].tolist())

# Predict with REAL training features
with torch.no_grad():
    feat_t = features.unsqueeze(0).to(device)
    mask_t = mask.unsqueeze(0).to(device)
    pad_mask = (mask_t == 0)
    
    preds = model.generate(feat_t, src_key_padding_mask=pad_mask, mask=mask_t)
    pred_fingers = preds[0].cpu().numpy()

print("\nPredicted fingers (first 50):")
print(pred_fingers[:50].tolist())

# Check accuracy
valid = mask.numpy() == 1
correct = (pred_fingers == true_fingers.numpy()) & valid
acc = correct.sum() / valid.sum()
print(f"\nAccuracy on this sequence: {acc*100:.1f}%")

# Count finger distribution
from collections import Counter
print(f"\nTrue distribution: {Counter(true_fingers[valid].tolist())}")
print(f"Pred distribution: {Counter(pred_fingers[valid].tolist())}")
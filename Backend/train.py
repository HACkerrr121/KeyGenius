import torch
from torch.utils.data import DataLoader
import numpy as np

from Datasets import FingeringDataset
from model import FingeringModel, FingeringLoss, compute_accuracy, compute_per_finger_accuracy

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "Music_Data/FingeringFiles"
HAND = 0  # 0 = right, 1 = left
HAND_NAME = "right" if HAND == 0 else "left"

MAX_SEQ_LEN = 200
BATCH_SIZE = 32
LR = 5e-4
EPOCHS = 150
PATIENCE = 20

EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 3
NUM_HEADS = 4
DROPOUT = 0.3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Training: {HAND_NAME} hand")

# ============================================================
# DATA
# ============================================================
train_dataset = FingeringDataset(DATA_DIR, hand=HAND, max_seq_len=MAX_SEQ_LEN, split='train')
val_dataset = FingeringDataset(DATA_DIR, hand=HAND, max_seq_len=MAX_SEQ_LEN, split='val')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# ============================================================
# CLASS WEIGHTS
# ============================================================
finger_counts = {i: 0 for i in range(6)}
for features, fingers, mask in train_loader:
    valid_fingers = fingers[mask == 1]
    for f in valid_fingers.tolist():
        finger_counts[f] += 1

print(f"Finger counts: {finger_counts}")

total = sum(finger_counts[i] for i in range(1, 6))
class_weights = torch.zeros(6)
for i in range(1, 6):
    if finger_counts[i] > 0:
        class_weights[i] = total / (5 * finger_counts[i])
    else:
        class_weights[i] = 1.0

class_weights[3] *= 1.3
class_weights[4] *= 1.3

print(f"Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

# ============================================================
# MODEL
# ============================================================
model = FingeringModel(
    input_dim=5,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_fingers=6,
    dropout=DROPOUT,
    num_heads=NUM_HEADS
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

loss_fn = FingeringLoss(label_smoothing=0.1, class_weights=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ============================================================
# TRAINING
# ============================================================
def train_epoch():
    model.train()
    total_loss, total_acc, num_batches = 0, 0, 0
    
    for features, fingers, mask in train_loader:
        features = features.to(device)
        fingers = fingers.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        logits = model(features, mask)
        loss = loss_fn(logits, fingers, mask)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += compute_accuracy(logits, fingers, mask)
        num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


def validate():
    model.eval()
    total_loss, total_acc, num_batches = 0, 0, 0
    all_per_finger = {i: [] for i in range(1, 6)}
    
    with torch.no_grad():
        for features, fingers, mask in val_loader:
            features = features.to(device)
            fingers = fingers.to(device)
            mask = mask.to(device)
            
            logits = model(features, mask)
            loss = loss_fn(logits, fingers, mask)
            
            total_loss += loss.item()
            total_acc += compute_accuracy(logits, fingers, mask)
            
            per_finger = compute_per_finger_accuracy(logits, fingers, mask)
            for f in range(1, 6):
                all_per_finger[f].append(per_finger[f])
            
            num_batches += 1
    
    avg_per_finger = {f: np.mean(all_per_finger[f]) for f in range(1, 6)}
    return total_loss / num_batches, total_acc / num_batches, avg_per_finger


# ============================================================
# MAIN
# ============================================================
best_val_acc = 0
patience_counter = 0

print("\n" + "=" * 60)
print("Starting training...")
print("=" * 60 + "\n")

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch()
    val_loss, val_acc, per_finger = validate()
    scheduler.step()
    
    pf = f"1:{per_finger[1]*100:.0f}% 2:{per_finger[2]*100:.0f}% 3:{per_finger[3]*100:.0f}% 4:{per_finger[4]*100:.0f}% 5:{per_finger[5]*100:.0f}%"
    print(f"Epoch {epoch+1:3d} | Train: {train_acc*100:.1f}% | Val: {val_acc*100:.1f}% | {pf}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'per_finger': per_finger
        }, f'best_model_{HAND_NAME}.pth')
        print(f"         💾 New best! {val_acc*100:.1f}%")
    else:
        patience_counter += 1
    
    if patience_counter >= PATIENCE:
        print(f"\n⚠️ Early stopping at epoch {epoch+1}")
        break

print("\n" + "=" * 60)
print(f"✅ Done! Best: {best_val_acc*100:.1f}%")
print("=" * 60)
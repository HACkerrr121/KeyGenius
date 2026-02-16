import torch
from torch.utils.data import DataLoader
import numpy as np

from Datasets import FingeringDataset
from model import FingeringTransformer, compute_accuracy, compute_per_finger_accuracy

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "Music_Data/FingeringFiles"
HAND = 0
HAND_NAME = "right" if HAND == 0 else "left"

MAX_SEQ_LEN = 200
BATCH_SIZE = 32
LR = 3e-4
MIN_LR = 1e-6
EPOCHS = 10 
PATIENCE = 30
LR_PATIENCE = 10  # Drop LR if no improvement for this many epochs
LR_FACTOR = 0.5   # Multiply LR by this when dropping

D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 1024
DROPOUT = 0.15

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Training: {HAND_NAME} hand\n")

# ============================================================
# DATA
# ============================================================
train_dataset = FingeringDataset(DATA_DIR, hand=HAND, max_seq_len=MAX_SEQ_LEN, split='train')
val_dataset = FingeringDataset(DATA_DIR, hand=HAND, max_seq_len=MAX_SEQ_LEN, split='val')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"\nTrain batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# ============================================================
# COMPUTE CLASS WEIGHTS
# ============================================================
finger_counts = {i: 0 for i in range(6)}
for features, fingers, mask in train_loader:
    valid_fingers = fingers[mask == 1]
    for f in valid_fingers.tolist():
        finger_counts[f] += 1

print(f"\nFinger counts: {finger_counts}")

total = sum(finger_counts[i] for i in range(1, 6))
class_weights = torch.zeros(6)
for i in range(1, 6):
    if finger_counts[i] > 0:
        class_weights[i] = total / (5 * finger_counts[i])
    else:
        class_weights[i] = 1.0

class_weights[3] *= 1.3
class_weights[4] *= 1.3
class_weights[5] *= 1.2

class_weights[1:] = class_weights[1:] / class_weights[1:].mean()

print(f"Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

# ============================================================
# MODEL
# ============================================================
model = FingeringTransformer(
    input_dim=17,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    num_fingers=6,
    class_weights=class_weights,
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-9)

# ReduceLROnPlateau - drops LR when val loss stops improving
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=LR_FACTOR,
    patience=LR_PATIENCE,
    min_lr=MIN_LR
)

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
        pad_mask = (mask == 0)

        optimizer.zero_grad()
        emissions, loss = model(features, fingers=fingers, mask=mask, src_key_padding_mask=pad_mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            preds = model.generate(features, src_key_padding_mask=pad_mask, mask=mask)
        acc = compute_accuracy(preds, fingers, mask)

        total_loss += loss.item()
        total_acc += acc
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
            pad_mask = (mask == 0)

            emissions, loss = model(features, fingers=fingers, mask=mask, src_key_padding_mask=pad_mask)
            preds = model.generate(features, src_key_padding_mask=pad_mask, mask=mask)

            acc = compute_accuracy(preds, fingers, mask)
            per_finger = compute_per_finger_accuracy(preds, fingers, mask)
            for f in range(1, 6):
                all_per_finger[f].append(per_finger[f])

            total_loss += loss.item()
            total_acc += acc
            num_batches += 1

    avg_pf = {f: np.mean(all_per_finger[f]) for f in range(1, 6)}
    return total_loss / num_batches, total_acc / num_batches, avg_pf


# ============================================================
# MAIN
# ============================================================
best_val_acc = 0
best_val_loss = float('inf')
patience_counter = 0

print("\n" + "=" * 70)
print("Training Transformer + CRF with Focal Loss")
print("=" * 70 + "\n")

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch()
    val_loss, val_acc, per_finger = validate()
    
    # Step scheduler with val loss
    scheduler.step(val_loss)
    
    lr = optimizer.param_groups[0]['lr']
    pf = " ".join([f"{f}:{per_finger[f]*100:.0f}%" for f in range(1, 6)])
    print(f"Ep {epoch+1:3d} | TrL: {train_loss:.3f} TrA: {train_acc*100:.1f}% | "
          f"VaL: {val_loss:.3f} VaA: {val_acc*100:.1f}% | {pf} | lr={lr:.1e}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'per_finger': per_finger,
        }, f'best_model_{HAND_NAME}.pth')
        print(f"         -> New best! {val_acc*100:.1f}%")
    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break
    
    # Stop if LR too low
    if lr <= MIN_LR:
        print(f"\nLR reached minimum ({MIN_LR}), stopping.")
        break

print("\n" + "=" * 70)
print(f"Done! Best val accuracy: {best_val_acc*100:.1f}%")
print("=" * 70)

# Print learned CRF transitions
print("\nLearned finger transitions (CRF):")
with torch.no_grad():
    T = model.crf.transitions[1:, 1:].cpu()
    labels = ['Th', 'Ix', 'Md', 'Rn', 'Pk']
    print("     " + "  ".join(f"{l:>5s}" for l in labels))
    for i, row_label in enumerate(labels):
        row = "  ".join(f"{T[i,j]:+.2f}" for j in range(5))
        print(f"  {row_label}  {row}")
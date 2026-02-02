import torch
import torch.nn as nn
import torch.nn.functional as F


class FingeringModel(nn.Module):
    def __init__(
        self,
        input_dim=5,
        embed_dim=128,
        hidden_dim=256,
        num_layers=3,
        num_fingers=6,
        dropout=0.3,
        num_heads=4
    ):
        super().__init__()
        
        self.num_fingers = num_fingers
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim * 2)
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_fingers)
        )
    
    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        
        if mask is not None:
            attn_mask = mask == 0
        else:
            attn_mask = None
        
        attn_out, _ = self.attention(x, x, x, key_padding_mask=attn_mask)
        x = self.attn_norm(x + attn_out)
        
        logits = self.output_proj(x)
        
        return logits


class FingeringLoss(nn.Module):
    def __init__(self, label_smoothing=0.1, class_weights=None):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
    
    def forward(self, logits, targets, mask):
        batch_size, seq_len, num_classes = logits.shape
        
        logits_flat = logits.view(-1, num_classes)
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)
        
        valid_indices = mask_flat == 1
        
        if valid_indices.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        valid_logits = logits_flat[valid_indices]
        valid_targets = targets_flat[valid_indices]
        
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(valid_logits.device)
        
        loss = F.cross_entropy(
            valid_logits,
            valid_targets,
            weight=weight,
            label_smoothing=self.label_smoothing
        )
        
        return loss


def compute_accuracy(logits, targets, mask):
    preds = logits.argmax(dim=-1)
    valid_mask = mask == 1
    
    if valid_mask.sum() == 0:
        return 0.0
    
    correct = (preds == targets) & valid_mask
    accuracy = correct.sum().float() / valid_mask.sum().float()
    
    return accuracy.item()


def compute_per_finger_accuracy(logits, targets, mask):
    preds = logits.argmax(dim=-1)
    valid_mask = mask == 1
    
    per_finger = {}
    for finger in range(1, 6):
        finger_mask = (targets == finger) & valid_mask
        if finger_mask.sum() == 0:
            per_finger[finger] = 0.0
        else:
            correct = (preds == finger) & finger_mask
            per_finger[finger] = (correct.sum().float() / finger_mask.sum().float()).item()
    
    return per_finger
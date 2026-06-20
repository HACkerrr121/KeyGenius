"""
KeyGenius model.

Architecture (encoder-only sequence labeling, one CRF chain per hand):

  [continuous feats | pitch-class emb | hand emb]
        -> input projection -> d_model
        -> sinusoidal positional encoding
        -> LocalContextConv   (multi-kernel convs catch scales/arpeggios)
        -> Transformer encoder (global context across the whole line)
        -> emission head        (per-note scores over fingers 1..5)
        -> CRF                  (transition constraints -> coherent path)

Loss = crf_weight * CRF_NLL + focal_weight * focal(emissions).
The CRF term dominates so the model learns finger FLOW, not just which
finger is most common.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG, NUM_FINGERS, N_PITCH_CLASSES, N_HANDS
from model.crf import CRF


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class LocalContextConv(nn.Module):
    """Multi-scale depthwise-ish conv block over the time axis."""
    def __init__(self, d_model, kernels, dropout):
        super().__init__()
        ch = max(1, d_model // len(kernels))
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, ch, kernel_size=k, padding=k // 2) for k in kernels
        ])
        self.proj = nn.Linear(ch * len(kernels), d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                      # x: [B,T,D]
        res = x
        h = x.transpose(1, 2)                  # [B,D,T]
        h = torch.cat([F.gelu(c(h)) for c in self.convs], dim=1)  # [B,D,T]
        h = h.transpose(1, 2)                  # [B,T,D]
        h = self.proj(h)
        return self.norm(res + self.drop(h))


class KeyGenius(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        m = cfg.model
        self.pc_emb = nn.Embedding(N_PITCH_CLASSES, cfg.feat.pitch_class_emb)
        self.hand_emb = nn.Embedding(N_HANDS, cfg.feat.hand_emb)
        self.in_proj = nn.Linear(cfg.input_dim, m.d_model)
        self.pos = PositionalEncoding(m.d_model)
        self.local = LocalContextConv(m.d_model, m.conv_kernels, m.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=m.d_model, nhead=m.n_heads, dim_feedforward=m.ffn_dim,
            dropout=m.dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=m.n_layers)
        self.head = nn.Linear(m.d_model, NUM_FINGERS)
        self.crf = CRF(NUM_FINGERS, init_range=m.crf_init_range)
        self.dropout = nn.Dropout(m.dropout)

    def emissions(self, cont, pc, hand, mask):
        x = torch.cat([cont, self.pc_emb(pc), self.hand_emb(hand)], dim=-1)
        x = self.in_proj(x)
        x = self.pos(x)
        x = self.local(x)
        pad = ~mask                              # True where padding
        x = self.encoder(x, src_key_padding_mask=pad)
        x = self.dropout(x)
        return self.head(x)                      # [B,T,NUM_FINGERS]

    def loss(self, batch):
        cfg = CFG.train
        em = self.emissions(batch["cont"], batch["pc"], batch["hand"], batch["mask"])
        crf_nll = self.crf(em, batch["labels"], batch["mask"])
        focal = self._focal(em, batch["labels"], batch["mask"], cfg.focal_gamma)
        total = cfg.crf_weight * crf_nll + cfg.focal_weight * focal
        return total, {"crf": crf_nll.item(), "focal": focal.item()}

    @staticmethod
    def _focal(em, labels, mask, gamma):
        logp = F.log_softmax(em, dim=-1)
        logp_t = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B,T]
        p_t = logp_t.exp()
        loss = -((1 - p_t) ** gamma) * logp_t
        m = mask.float()
        return (loss * m).sum() / m.sum().clamp(min=1.0)

    def pretrain_loss(self, batch):
        """Masked focal loss for PARTIALLY-labeled data (ThumbSet).

        Labels are finger-1, so unlabeled notes are -1. We train the encoder /
        emission head only on the notes that have a real 1..5 label; unlabeled
        notes still pass through (providing context) but contribute no loss.
        The CRF transitions are left for PIG fine-tuning on complete labels.
        """
        cfg = CFG.train
        em = self.emissions(batch["cont"], batch["pc"], batch["hand"], batch["mask"])
        labels = batch["labels"]
        labeled = (labels >= 0) & batch["mask"]            # [B,T] bool
        if labeled.sum() == 0:
            return em.sum() * 0.0, {"focal": 0.0, "labeled": 0}
        safe_labels = labels.clamp(min=0)                  # avoid -1 in gather
        focal = self._focal(em, safe_labels, labeled, cfg.focal_gamma)
        return focal, {"focal": focal.item(), "labeled": int(labeled.sum().item())}

    @torch.no_grad()
    def predict(self, cont, pc, hand, mask):
        em = self.emissions(cont, pc, hand, mask)
        return self.crf.decode(em, mask)          # list[list[int]] of 0..4

    @torch.no_grad()
    def predict_with_confidence(self, cont, pc, hand, mask):
        """Single-sequence (B=1) decode + per-note confidence.

        Returns (fingers[0..4], confidence in [0,1]) where confidence[i] is the
        posterior probability the CRF assigns to the finger it chose for note i.
        """
        assert cont.size(0) == 1, "predict_with_confidence expects batch size 1"
        em = self.emissions(cont, pc, hand, mask)         # [1, T, K]
        path = self.crf.decode(em, mask)[0]               # list[int]
        marg = self.crf.marginals(em[0])                  # [T, K]
        conf = [float(marg[i, path[i]]) for i in range(len(path))]
        return path, conf

    @torch.no_grad()
    def predict_constrained(self, cont, pc, hand, mask, midis, onsets, hand_id):
        """Single-sequence decode with physical chord constraints + confidence.

        midis/onsets: per-note lists (sequence order); hand_id: 0=RH, 1=LH.
        Confidence is still the (unconstrained) CRF posterior, so it honestly
        reflects how sure the model was even where a constraint overrode it.
        """
        assert cont.size(0) == 1, "predict_constrained expects batch size 1"
        em = self.emissions(cont, pc, hand, mask)         # [1, T, K]
        path = self.crf.decode_constrained(em[0], midis, onsets, hand_id)
        marg = self.crf.marginals(em[0])                  # [T, K]
        conf = [float(marg[i, path[i]]) for i in range(len(path))]
        return path, conf
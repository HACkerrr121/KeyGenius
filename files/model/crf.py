"""
Linear-chain Conditional Random Field (batch-first, masked).

This is the piece that turns independent per-note finger guesses into a
physically coherent sequence: it learns transition scores between fingers so
the decoded path respects how a hand actually moves (e.g. you don't jump
thumb->pinky->thumb on a stepwise run).

Emissions: [B, T, K]    mask: [B, T] (bool, left-aligned, mask[:,0] all True)
Tags:      [B, T] (long, 0..K-1)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, num_tags: int, init_range: float = 0.5):
        super().__init__()
        self.num_tags = num_tags
        self.start = nn.Parameter(torch.empty(num_tags))
        self.end = nn.Parameter(torch.empty(num_tags))
        self.trans = nn.Parameter(torch.empty(num_tags, num_tags))
        nn.init.uniform_(self.start, -init_range, init_range)
        nn.init.uniform_(self.end, -init_range, init_range)
        nn.init.uniform_(self.trans, -init_range, init_range)

    # ---- loss ---------------------------------------------------------------
    def forward(self, emissions, tags, mask):
        """Return mean negative log-likelihood over the batch."""
        mask = mask.bool()
        numerator = self._score(emissions, tags, mask)
        denominator = self._partition(emissions, mask)
        nll = denominator - numerator           # [B]
        return nll.mean()

    def _score(self, emissions, tags, mask):
        B, T, K = emissions.shape
        score = self.start[tags[:, 0]]
        score = score + emissions[torch.arange(B), 0, tags[:, 0]]
        for t in range(1, T):
            m = mask[:, t].float()
            trans_t = self.trans[tags[:, t - 1], tags[:, t]]
            emit_t = emissions[torch.arange(B), t, tags[:, t]]
            score = score + (trans_t + emit_t) * m
        # add end transition at each sequence's last valid position
        last_idx = mask.long().sum(dim=1) - 1                # [B]
        last_tags = tags[torch.arange(B), last_idx]
        score = score + self.end[last_tags]
        return score

    def _partition(self, emissions, mask):
        B, T, K = emissions.shape
        alpha = self.start.unsqueeze(0) + emissions[:, 0]     # [B,K]
        for t in range(1, T):
            broadcast = (alpha.unsqueeze(2)                    # [B,K,1] prev
                         + self.trans.unsqueeze(0)            # [1,K,K]
                         + emissions[:, t].unsqueeze(1))      # [B,1,K] curr
            new_alpha = torch.logsumexp(broadcast, dim=1)     # [B,K]
            m = mask[:, t].unsqueeze(1)
            alpha = torch.where(m, new_alpha, alpha)
        alpha = alpha + self.end.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)                  # [B]

    # ---- marginals (per-note confidence) ------------------------------------
    @torch.no_grad()
    def marginals(self, emissions):
        """Posterior P(tag_t = k | emissions) for ONE sequence.

        emissions: [T, K] (no batch, no padding). Returns [T, K] probabilities.
        Used at inference to flag low-confidence notes.
        """
        T, K = emissions.shape
        # forward
        alpha = torch.empty(T, K, device=emissions.device)
        alpha[0] = self.start + emissions[0]
        for t in range(1, T):
            alpha[t] = torch.logsumexp(
                alpha[t - 1].unsqueeze(1) + self.trans, dim=0) + emissions[t]
        log_Z = torch.logsumexp(alpha[-1] + self.end, dim=0)
        # backward
        beta = torch.empty(T, K, device=emissions.device)
        beta[-1] = self.end
        for t in range(T - 2, -1, -1):
            beta[t] = torch.logsumexp(
                self.trans + (emissions[t + 1] + beta[t + 1]).unsqueeze(0), dim=1)
        return (alpha + beta - log_Z).exp()           # [T, K]

    # ---- constrained decode (single sequence) -------------------------------
    @torch.no_grad()
    def decode_constrained(self, emissions, midis, onsets, hand,
                           chord_eps: float = 1e-3, big: float = 1e4):
        """Viterbi for ONE per-hand sequence with PHYSICAL chord constraints.

        emissions : [T, K]   (no batch)
        midis     : list[int]  MIDI pitch per note (sequence order)
        onsets    : list[float] onset time per note
        hand      : 0 = right, 1 = left

        Notes sharing an onset form a CHORD. Within a chord (notes are ordered
        low->high pitch), a single hand physically must:
          * use a DIFFERENT finger for each note, and
          * order fingers monotonically with pitch
            - right hand: higher pitch -> higher finger number
            - left  hand: higher pitch -> lower  finger number (thumb on top)

        These are near-certain physical facts, so we forbid violations with a
        large penalty. Melodic (different-onset) transitions are left to the
        learned CRF, so confident/correct passages are untouched. This cleans
        up exactly the low-confidence chord guesses without retraining.
        """
        T, K = emissions.shape
        dev = emissions.device
        idx = torch.arange(K, device=dev)
        prev_i = idx.unsqueeze(1)      # [K,1]
        curr_i = idx.unsqueeze(0)      # [1,K]

        def chord_penalty(t):
            same_onset = abs(onsets[t] - onsets[t - 1]) < chord_eps
            if not same_onset:
                return torch.zeros(K, K, device=dev)
            if hand == 0:              # RH: curr finger must be strictly higher
                bad = curr_i <= prev_i
            else:                      # LH: curr finger must be strictly lower
                bad = curr_i >= prev_i
            return torch.where(bad, torch.full((K, K), -big, device=dev),
                               torch.zeros(K, K, device=dev))

        score = self.start + emissions[0]          # [K]
        history = []
        for t in range(1, T):
            pen = chord_penalty(t)                 # [K_prev, K_curr]
            broadcast = score.unsqueeze(1) + self.trans + pen   # [K_prev,K_curr]
            best_score, best_prev = broadcast.max(dim=0)        # [K_curr]
            score = best_score + emissions[t]
            history.append(best_prev)

        score = score + self.end
        last = int(score.argmax().item())
        path = [last]
        for t in range(T - 2, -1, -1):
            last = int(history[t][last].item())
            path.append(last)
        path.reverse()
        return path

    # ---- decode -------------------------------------------------------------
    @torch.no_grad()
    def decode(self, emissions, mask):
        """Viterbi best path. Returns list[list[int]] (one per sequence)."""
        mask = mask.bool()
        B, T, K = emissions.shape
        score = self.start.unsqueeze(0) + emissions[:, 0]     # [B,K]
        history = []
        for t in range(1, T):
            broadcast = score.unsqueeze(2) + self.trans.unsqueeze(0)  # [B,K,K]
            best_score, best_prev = broadcast.max(dim=1)      # [B,K]
            best_score = best_score + emissions[:, t]
            m = mask[:, t].unsqueeze(1)
            score = torch.where(m, best_score, score)
            history.append(best_prev)

        score = score + self.end.unsqueeze(0)
        lengths = mask.long().sum(dim=1)
        best_paths = []
        for b in range(B):
            L = int(lengths[b].item())
            last_tag = int(score[b].argmax().item())
            path = [last_tag]
            for t in range(L - 2, -1, -1):
                last_tag = int(history[t][b, last_tag].item())
                path.append(last_tag)
            path.reverse()
            best_paths.append(path)
        return best_paths
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
                           chord_eps: float = 1e-3, big: float = 1e4,
                           w_repeat: float = 2.5, w_cramp: float = 1.5,
                           w_thumb_black: float = 1.0):
        """Viterbi for ONE per-hand sequence with physical constraints.

        HARD chord constraints (same-onset notes), plus SOFT melodic nudges
        that encode real ergonomics. Soft penalties are small and the model
        can override them when confident — they only break ties on the
        low-confidence notes where the model was guessing.

        emissions : [T, K]      midis : list[int]    onsets : list[float]
        hand      : 0 = right, 1 = left

        Hard (chord, same onset; notes ordered low->high pitch):
          * distinct finger per note, and fingers ordered with pitch
            (RH higher pitch -> higher finger; LH -> lower).

        Soft (consecutive MELODIC notes, different onset):
          * w_repeat : discourage reusing the SAME finger across a small step
            when the notes differ (the "lazy 4-4 where 4-5 is better" case).
          * w_cramp  : discourage cramped finger pairs for the interval
            (e.g. 3-4 across a third when a wider spread is natural).
          * w_thumb_black: mild discouragement of thumb (finger 1) landing on
            a black key in stepwise motion (ergonomically awkward, not forbidden).

        Set any weight to 0.0 to disable that nudge. All are soft, so confident
        correct choices survive; they mainly clean up the dubious notes.
        """
        T, K = emissions.shape
        dev = emissions.device
        idx = torch.arange(K, device=dev)
        prev_i = idx.unsqueeze(1)      # [K,1]  previous finger label (0..4)
        curr_i = idx.unsqueeze(0)      # [1,K]  current finger label
        zeros = torch.zeros(K, K, device=dev)
        BLACK = {1, 3, 6, 8, 10}

        # finger spans (label i = finger i+1): comfortable max semitone reach
        # between adjacent fingers, used to flag cramped pairs on an interval.
        # rough, soft guidance only.
        finger_dist = (curr_i - prev_i).abs().float()   # how many fingers apart

        def penalty(t):
            same_onset = abs(onsets[t] - onsets[t - 1]) < chord_eps
            interval = abs(midis[t] - midis[t - 1])

            if same_onset:
                # ---- HARD chord constraints ----
                if hand == 0:
                    bad = curr_i <= prev_i
                else:
                    bad = curr_i >= prev_i
                return torch.where(bad, torch.full((K, K), -big, device=dev), zeros)

            # ---- SOFT melodic nudges (different onset) ----
            pen = zeros.clone()

            # SYMMETRIC ergonomic penalty: a hand's finger spread should roughly
            # match how far the notes move. We map the pitch interval to an
            # "ideal" finger gap, then penalize by how far the chosen finger gap
            # deviates from it -- in EITHER direction. This single rule covers:
            #   * lazy repeat  (notes move, fingers don't)  -> gap too small
            #   * awkward jump (notes barely move, fingers leap, e.g. 4->1) -> gap too big
            #   * cramped pair (wide interval, adjacent fingers, e.g. 3-4 on a 3rd) -> gap too small
            if w_cramp > 0:
                # ideal finger gap for an interval (semitones):
                #   unison/step (0-2): ~0-1 fingers apart
                #   third (3-4):       ~1-2
                #   fourth/fifth (5-7):~2-3
                #   sixth+ (8+):       ~3-4
                if interval <= 2:
                    ideal = 0.5
                elif interval <= 4:
                    ideal = 1.5
                elif interval <= 7:
                    ideal = 2.5
                else:
                    ideal = 3.5
                # deviation of chosen finger gap from ideal, penalized smoothly
                deviation = (finger_dist - ideal).abs()
                # penalize deviations beyond a small slack (0.5 finger), softly.
                # smaller slack so subtle cramped pairs (3-4 on a third) get nudged.
                excess = torch.clamp(deviation - 0.5, min=0.0)
                pen = pen - excess * w_cramp

            # extra nudge specifically against reusing the SAME finger when the
            # notes actually move (the clearest "lazy" case), on top of the above
            if w_repeat > 0 and interval != 0 and interval <= 4:
                same_finger = (curr_i == prev_i)
                pen = pen - same_finger.float() * w_repeat

            # thumb on a black key during stepwise motion: mildly awkward
            if w_thumb_black > 0 and interval <= 2:
                curr_black = (midis[t] % 12) in BLACK
                if curr_black:
                    thumb = (curr_i == 0).float()      # label 0 = finger 1 = thumb
                    pen = pen - thumb * w_thumb_black

            return pen

        score = self.start + emissions[0]          # [K]
        history = []
        for t in range(1, T):
            pen = penalty(t)                        # [K_prev, K_curr]
            broadcast = score.unsqueeze(1) + self.trans + pen
            best_score, best_prev = broadcast.max(dim=0)
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
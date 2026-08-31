"""Multi-block masking, following I-JEPA (Assran et al. 2023), Section 3 + App. A.

Paper settings (Table 7 / repo defaults):
- 4 target blocks, scale (0.15, 0.2) of the image, aspect ratio (0.75, 1.5)
- 1 context block, scale (0.85, 1.0), unit aspect ratio
- target patches are REMOVED from the context (context never overlaps targets)

All masks are lists of patch indices on a grid x grid layout.
Batch collation keeps a fixed count per mask (min over batch) so tensors stack —
same trick as the official mask collator.
"""
import math
import random
import torch


class MultiBlockMaskGenerator:
    def __init__(self, grid, num_targets=4,
                 target_scale=(0.15, 0.2), target_aspect=(0.75, 1.5),
                 context_scale=(0.85, 1.0), min_keep=4, rng=None):
        self.grid = grid
        self.num_targets = num_targets
        self.target_scale = target_scale
        self.target_aspect = target_aspect
        self.context_scale = context_scale
        self.min_keep = min_keep
        self.rng = rng or random.Random()

    def _sample_block(self, scale_range, aspect_range):
        """Returns a set of patch indices forming one rectangular block."""
        g = self.grid
        scale = self.rng.uniform(*scale_range)
        n_patches = scale * g * g
        log_lo, log_hi = math.log(aspect_range[0]), math.log(aspect_range[1])
        aspect = math.exp(self.rng.uniform(log_lo, log_hi))
        h = int(round(math.sqrt(n_patches * aspect)))
        w = int(round(math.sqrt(n_patches / aspect)))
        h = max(1, min(h, g))
        w = max(1, min(w, g))
        top = self.rng.randint(0, g - h)
        left = self.rng.randint(0, g - w)
        return {(top + i) * g + (left + j) for i in range(h) for j in range(w)}

    def sample(self):
        """One image's masks: (context_indices, [target_block_indices...])."""
        targets = [
            self._sample_block(self.target_scale, self.target_aspect)
            for _ in range(self.num_targets)
        ]
        union = set().union(*targets)
        ctx = self._sample_block(self.context_scale, (1.0, 1.0))
        ctx = ctx - union
        if len(ctx) < self.min_keep:  # pathological overlap; resample context
            all_idx = set(range(self.grid * self.grid))
            ctx = all_idx - union
        return sorted(ctx), [sorted(t) for t in targets]

    def collate(self, batch_size):
        """Batched masks with uniform lengths (truncate to min length per mask).
        Returns ctx_idx (B,K) and list of num_targets tensors (B,M_i)."""
        samples = [self.sample() for _ in range(batch_size)]
        k = min(len(s[0]) for s in samples)
        ctx = torch.stack([
            torch.tensor(self.rng.sample(s[0], k) if len(s[0]) > k else s[0])
            for s in samples])
        ctx, _ = ctx.sort(dim=1)
        tgts = []
        for t in range(self.num_targets):
            m = min(len(s[1][t]) for s in samples)
            tt = torch.stack([
                torch.tensor(self.rng.sample(s[1][t], m)
                             if len(s[1][t]) > m else s[1][t])
                for s in samples])
            tt, _ = tt.sort(dim=1)
            tgts.append(tt)
        return ctx, tgts


def random_mask(batch_size, num_patches, mask_ratio, device):
    """MAE-style random masking. Returns keep_idx (B,K) and masked_idx (B,N-K)."""
    n_keep = max(1, int(round(num_patches * (1.0 - mask_ratio))))
    noise = torch.rand(batch_size, num_patches, device=device)
    shuffle = noise.argsort(dim=1)
    keep_idx, _ = shuffle[:, :n_keep].sort(dim=1)
    masked_idx, _ = shuffle[:, n_keep:].sort(dim=1)
    return keep_idx, masked_idx

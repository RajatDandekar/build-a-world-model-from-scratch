"""Four self-supervised architectures on the toy world, each with the energy
function its own loss implies. Tiny conv nets; minutes to train.

  gen_det   deterministic generative: dec(enc(x)) -> y pixels, MSE.
            E(x,y) = ||dec(enc(x)) - y||^2. One output per context: at a
            bimodal fork it must predict the MEAN of the futures.
  ae        autoencoder on y with an 8-d bottleneck (capacity-limited).
            E(y) = ||dec(enc(y)) - y||^2 — note: x plays no role. Low energy
            on the whole data manifold regardless of context.
  jea       joint embedding, no guard: pull enc_x(x) and enc_y(y) together.
            E(x,y) = 1 - cos. Collapses -> flat landscape.
  jepa      predict enc_y(y) from enc_x(x) in representation space, EMA +
            stop-grad guard, and a DISCRETE latent z (K embeddings). Training
            and energy both take min over z: F(x,y) = min_z E(x,y,z) —
            LeCun's latent-variable energy, literally.
            K=1 gives the no-latent ablation ("jepa_noz").
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 64


def conv_enc(in_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, 16, 4, 2, 1), nn.ReLU(),   # 24
        nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),      # 12
        nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),      # 6
        nn.Flatten(), nn.Linear(64 * 36, D),
    )


def conv_dec(out_ch=1, zdim=D):
    class Dec(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(zdim, 64 * 36)
            self.net = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(16, out_ch, 4, 2, 1),
            )

        def forward(self, z):
            h = self.fc(z).view(-1, 64, 6, 6)
            return self.net(h)
    return Dec()


class GenDet(nn.Module):
    def __init__(self, ctx_ch):
        super().__init__()
        self.enc = conv_enc(ctx_ch)
        self.dec = conv_dec()

    def loss(self, x, y):
        return F.mse_loss(self.dec(self.enc(x)), y)

    @torch.no_grad()
    def energy(self, x, y):
        pred = self.dec(self.enc(x))
        return (pred - y).pow(2).flatten(1).mean(1)

    @torch.no_grad()
    def predict(self, x):
        return self.dec(self.enc(x))


class AE(nn.Module):
    def __init__(self, ctx_ch, zdim=8):
        super().__init__()
        self.enc = nn.Sequential(conv_enc(1), nn.Linear(D, zdim))
        self.dec = conv_dec(zdim=zdim)

    def loss(self, x, y):
        return F.mse_loss(self.dec(self.enc(y)), y)

    @torch.no_grad()
    def energy(self, x, y):
        return (self.dec(self.enc(y)) - y).pow(2).flatten(1).mean(1)


class JEA(nn.Module):
    def __init__(self, ctx_ch):
        super().__init__()
        self.enc_x = conv_enc(ctx_ch)
        self.enc_y = conv_enc(1)

    def loss(self, x, y):
        zx, zy = self.enc_x(x), self.enc_y(y)
        return 1.0 - F.cosine_similarity(zx, zy, dim=-1).mean()

    @torch.no_grad()
    def energy(self, x, y):
        return 1.0 - F.cosine_similarity(self.enc_x(x), self.enc_y(y), dim=-1)

    @torch.no_grad()
    def feat_std(self, y):
        z = F.normalize(self.enc_y(y), dim=-1)
        return z.std(0).mean().item()


class JEPA(nn.Module):
    """Faithful to the real thing: ONE frame encoder shared between context
    and target. Context frames are encoded individually and fused; the
    target branch is an EMA copy of that same frame encoder — so the teacher
    tracks a TRAINED encoder (gradients reach frame_enc through the context
    path), exactly like I-JEPA's context/target encoder pair."""

    def __init__(self, ctx_ch, K=4, ema=0.98):
        super().__init__()
        self.ctx_frames = ctx_ch
        self.frame_enc = conv_enc(1)
        self.fuse = nn.Sequential(
            nn.Linear(D * ctx_ch, 128), nn.ReLU(), nn.Linear(128, D))
        self.tgt_enc = copy.deepcopy(self.frame_enc)
        for p in self.tgt_enc.parameters():
            p.requires_grad = False
        self.z_embed = nn.Embedding(K, 16)
        self.pred = nn.Sequential(
            nn.Linear(D + 16, 128), nn.ReLU(), nn.Linear(128, D))
        self.K = K
        self.ema = ema

    def enc_x(self, x):
        B, C = x.shape[:2]
        f = self.frame_enc(x.reshape(B * C, 1, *x.shape[2:]))
        return self.fuse(f.reshape(B, C * D))

    def _pred_all_z(self, zx):
        B = zx.size(0)
        outs = []
        for k in range(self.K):
            zk = self.z_embed.weight[k][None].expand(B, -1)
            outs.append(self.pred(torch.cat([zx, zk], dim=-1)))
        return torch.stack(outs, dim=1)          # (B,K,D)

    def loss(self, x, y):
        with torch.no_grad():
            tgt = F.layer_norm(self.tgt_enc(y), (D,))
        preds = self._pred_all_z(self.enc_x(x))
        e = F.smooth_l1_loss(preds, tgt[:, None].expand_as(preds),
                             reduction="none").mean(-1)   # (B,K)
        # soft responsibilities instead of hard min: prevents one z from
        # winning every sample early and starving the rest (dead codes)
        w = torch.softmax(-e.detach() / 0.02, dim=1)
        return (w * e).sum(dim=1).mean()

    @torch.no_grad()
    def ema_update(self):
        for q, k in zip(self.frame_enc.parameters(),
                        self.tgt_enc.parameters()):
            k.mul_(self.ema).add_(q, alpha=1 - self.ema)

    @torch.no_grad()
    def energy(self, x, y, return_z=False):
        tgt = F.layer_norm(self.tgt_enc(y), (D,))
        preds = self._pred_all_z(self.enc_x(x))
        e = F.smooth_l1_loss(preds, tgt[:, None].expand_as(preds),
                             reduction="none").mean(-1)
        if return_z:
            return e.min(dim=1).values, e.argmin(dim=1)
        return e.min(dim=1).values


class CameraJEPA(nn.Module):
    """z = the 2D camera shift, inferred by minimization over a shift grid."""

    def __init__(self, ctx_ch=1, max_shift=6, step=1.0, ema=1.0):
        # ema=1.0 FREEZES the teacher at its random initialization — a
        # deliberate choice for this task: a trained teacher drifts toward
        # shift-INVARIANT features, squeezing out exactly the information z
        # must carry (criterion-2 vs criterion-4 tension, measured: trained
        # teachers gave 4.3-5.8 px error, the frozen random teacher ~2.4 px).
        super().__init__()
        self.enc_x = conv_enc(ctx_ch)
        self.tgt_y = copy.deepcopy(self.enc_x)
        for p in self.tgt_y.parameters():
            p.requires_grad = False
        vals = torch.arange(-max_shift, max_shift + 1e-6, step)
        gy, gx = torch.meshgrid(vals, vals, indexing="ij")
        self.register_buffer("zgrid",
                             torch.stack([gx.flatten(), gy.flatten()], 1))
        self.zproj = nn.Sequential(nn.Linear(2, 32), nn.ReLU(),
                                   nn.Linear(32, 16))
        self.pred = nn.Sequential(
            nn.Linear(D + 16, 128), nn.ReLU(), nn.Linear(128, D))
        self.ema = ema

    def _energies(self, x, y):
        """(B, Nz) energy for every candidate shift."""
        with torch.no_grad():
            tgt = F.layer_norm(self.tgt_y(y), (D,))
        zx = self.enc_x(x)
        zf = self.zproj(self.zgrid)                       # (Nz,16)
        B, Nz = zx.size(0), zf.size(0)
        p = self.pred(torch.cat(
            [zx[:, None].expand(B, Nz, D), zf[None].expand(B, Nz, 16)],
            dim=-1))
        return F.smooth_l1_loss(p, tgt[:, None].expand_as(p),
                                reduction="none").mean(-1)

    def loss(self, x, y, true_z=None):
        e = self._energies(x, y)
        # hard min works here: the shift grid is a fixed, identifiable
        # codebook (no dead-code risk — every shift occurs in the data)
        return e.min(dim=1).values.mean()

    @torch.no_grad()
    def infer_z(self, x, y):
        e = self._energies(x, y)
        return self.zgrid[e.argmin(dim=1)]

    @torch.no_grad()
    def ema_update(self):
        for q, k in zip(self.enc_x.parameters(), self.tgt_y.parameters()):
            k.mul_(self.ema).add_(q, alpha=1 - self.ema)

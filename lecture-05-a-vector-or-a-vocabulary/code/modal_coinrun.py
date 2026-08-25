"""Lecture 5 — CoinRun: one problem, two world models.

  A) RSSM        — continuous Gaussian latent + GRU        (Lecture 4's design)
  B) IRIS-mini   — discrete VQ codebook + causal transformer

Matched parameter budgets, identical data, identical evaluation. Everything the
lecture claims is measured here.

  modal run --detach modal_coinrun.py::gen_data
  modal run --detach modal_coinrun.py::train_rssm
  modal run --detach modal_coinrun.py::train_iris
  modal run --detach modal_coinrun.py::compare
"""
import modal

app = modal.App("l5-coinrun")
vol = modal.Volume.from_name("l5-coinrun", create_if_missing=True)

# procgen needs py3.10 wheels + GL/glib for its prebuilt libenv.so
data_image = (modal.Image.debian_slim(python_version="3.10")
              .apt_install("libgl1-mesa-glx", "libglib2.0-0")
              .pip_install("procgen==0.10.7", "gym3", "numpy<2"))
gpu_image = (modal.Image.debian_slim(python_version="3.11")
             .apt_install("ffmpeg")
             .pip_install("torch", "torchvision", "numpy", "matplotlib",
                          "pillow", "scikit-learn"))

N_EPISODES = 600
MIN_LEN = 64
HOLD_OUT = 40


# ─────────────────────────────────────────────────────────── data
@app.function(image=data_image, timeout=5400, volumes={"/out": vol}, cpu=8)
def gen_data():
    """Roll out CoinRun with a random policy; keep 64x64 frames + actions."""
    import numpy as np
    from procgen import ProcgenGym3Env
    from gym3 import types_np

    rng = np.random.RandomState(0)
    eps, kept, tries = [], 0, 0
    while kept < N_EPISODES and tries < N_EPISODES * 10:
        tries += 1
        env = ProcgenGym3Env(num=1, env_name="coinrun",
                             start_level=int(rng.randint(0, 100000)),
                             distribution_mode="easy")
        obs_l, act_l = [], []
        first_obs = True
        for _ in range(512):
            _, obs, first = env.observe()
            if first and not first_obs:
                break
            a = types_np.sample(env.ac_space, bshape=(env.num,))
            env.act(a)
            obs_l.append(obs["rgb"][0])
            act_l.append(int(a[0]))
            first_obs = False
        if len(obs_l) < MIN_LEN:
            continue
        kept += 1
        eps.append({"frames": np.asarray(obs_l, np.uint8),
                    "actions": np.asarray(act_l, np.int64)})
        if kept % 50 == 0:
            print(f"  {kept}/{N_EPISODES} episodes", flush=True)

    lens = [len(e["frames"]) for e in eps]
    print(f"{len(eps)} episodes · len min/mean/max "
          f"{min(lens)}/{np.mean(lens):.0f}/{max(lens)} · "
          f"{sum(lens):,} frames", flush=True)
    d = {"n": len(eps)}
    for i, e in enumerate(eps):
        d[f"f{i}"] = e["frames"]; d[f"a{i}"] = e["actions"]
    np.savez_compressed("/out/coinrun.npz", **d)
    vol.commit()
    print("saved /out/coinrun.npz")


# ─────────────────────────────────────────────────── shared model source
SRC = '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

A_DIM = 15                      # CoinRun action space
EMB = 512
MIN_STD = 0.1

# ---------------------------------------------------------------- A · RSSM
H, S = 256, 32

class ConvEnc(nn.Module):
    def __init__(s_, out=EMB):
        super().__init__()
        s_.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ELU(),     # 32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ELU(),    # 16
            nn.Conv2d(64, 128, 4, 2, 1), nn.ELU(),   # 8
            nn.Conv2d(128, 256, 4, 2, 1), nn.ELU(),  # 4
            nn.Flatten(), nn.Linear(256 * 16, out), nn.ELU())
    def forward(s_, x):                              # (B,64,64,3) in [0,1]
        return s_.net(x.permute(0, 3, 1, 2) - 0.5)

class ConvDec(nn.Module):
    def __init__(s_, in_dim):
        super().__init__()
        s_.fc = nn.Linear(in_dim, 256 * 16)
        s_.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1))
    def forward(s_, z):
        return s_.net(s_.fc(z).view(-1, 256, 4, 4)).permute(0, 2, 3, 1) + 0.5

class RSSM(nn.Module):
    """Lecture 4's design, on CoinRun. Continuous Gaussian latent."""
    def __init__(s_):
        super().__init__()
        s_.enc = ConvEnc(); s_.dec = ConvDec(H + S)
        s_.in_mlp = nn.Sequential(nn.Linear(S + A_DIM, 256), nn.ELU())
        s_.gru = nn.GRUCell(256, H)
        s_.prior_mlp = nn.Sequential(nn.Linear(H, 256), nn.ELU(), nn.Linear(256, 2 * S))
        s_.post_mlp = nn.Sequential(nn.Linear(H + EMB, 256), nn.ELU(), nn.Linear(256, 2 * S))
    def dist(s_, o):
        mu, std = o.chunk(2, -1); return mu, F.softplus(std) + MIN_STD
    def step_h(s_, h, s, a): return s_.gru(s_.in_mlp(torch.cat([s, a], -1)), h)
    def prior(s_, h): return s_.dist(s_.prior_mlp(h))
    def posterior(s_, h, e): return s_.dist(s_.post_mlp(torch.cat([h, e], -1)))
    def feat(s_, h, s): return torch.cat([h, s], -1)

# --------------------------------------------------------- B · IRIS-mini
K = 512            # codebook size  -> the "vocabulary"
D = 64             # code dimension
GRID = 4           # 4x4 = 16 tokens per frame
NTOK = GRID * GRID

class VQ(nn.Module):
    """VQ-VAE with EMA codebook + dead-code revival (collapse is the classic failure)."""
    def __init__(s_, k=K, d=D, decay=0.99):
        super().__init__()
        s_.k, s_.d, s_.decay = k, d, decay
        s_.register_buffer("emb", torch.randn(k, d) * 0.5)
        s_.register_buffer("cluster", torch.ones(k))
        s_.register_buffer("avg", s_.emb.clone())
    def forward(s_, z):                      # z: (B,d,H,W)
        B, d, Hh, Ww = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, d)
        dist = (flat.pow(2).sum(1, keepdim=True) - 2 * flat @ s_.emb.t()
                + s_.emb.pow(2).sum(1))
        idx = dist.argmin(1)
        q = s_.emb[idx].view(B, Hh, Ww, d).permute(0, 3, 1, 2)
        if s_.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, s_.k).float()
                s_.cluster.mul_(s_.decay).add_(onehot.sum(0), alpha=1 - s_.decay)
                s_.avg.mul_(s_.decay).add_(onehot.t() @ flat, alpha=1 - s_.decay)
                n = s_.cluster.sum()
                cl = (s_.cluster + 1e-5) / (n + s_.k * 1e-5) * n
                s_.emb.copy_(s_.avg / cl.unsqueeze(1))
                # revive codes that have gone unused
                dead = s_.cluster < 0.5
                if dead.any():
                    pick = flat[torch.randint(0, flat.shape[0], (int(dead.sum()),))]
                    s_.emb[dead] = pick
                    s_.cluster[dead] = 1.0
                    s_.avg[dead] = pick
        commit = F.mse_loss(z, q.detach())
        q = z + (q - z).detach()                       # straight-through
        return q, idx.view(B, Hh, Ww), commit
    def perplexity(s_):
        p = s_.cluster / s_.cluster.sum()
        return float(torch.exp(-(p * (p + 1e-10).log()).sum()))

class Tokenizer(nn.Module):
    """frame <-> 16 discrete tokens from a 512-word codebook"""
    def __init__(s_):
        super().__init__()
        s_.enc = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ELU(),      # 32
            nn.Conv2d(64, 128, 4, 2, 1), nn.ELU(),    # 16
            nn.Conv2d(128, 256, 4, 2, 1), nn.ELU(),   # 8
            nn.Conv2d(256, D, 4, 2, 1))               # 4  -> (B,D,4,4)
        s_.vq = VQ()
        s_.dec = nn.Sequential(
            nn.ConvTranspose2d(D, 256, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1))
    def encode(s_, x):
        return s_.enc(x.permute(0, 3, 1, 2) - 0.5)
    def decode_q(s_, q):
        return s_.dec(q).permute(0, 2, 3, 1) + 0.5
    def decode_tokens(s_, idx):                       # idx: (B,4,4) long
        q = s_.vq.emb[idx].permute(0, 3, 1, 2)
        return s_.decode_q(q)
    def forward(s_, x):
        z = s_.encode(x)
        q, idx, commit = s_.vq(z)
        return s_.decode_q(q), idx, commit

class Block(nn.Module):
    def __init__(s_, d, heads):
        super().__init__()
        s_.ln1 = nn.LayerNorm(d); s_.ln2 = nn.LayerNorm(d)
        s_.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        s_.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(s_, x, mask, need_w=False):
        h = s_.ln1(x)
        a, w = s_.attn(h, h, h, attn_mask=mask, need_weights=need_w,
                       average_attn_weights=True)
        x = x + a
        return x + s_.mlp(s_.ln2(x)), w

class IRISGPT(nn.Module):
    """causal transformer over [16 frame tokens + 1 action token] per step"""
    def __init__(s_, d=256, layers=6, heads=8, ctx_steps=8):
        super().__init__()
        s_.d, s_.ctx_steps = d, ctx_steps
        s_.per_step = NTOK + 1
        s_.T = ctx_steps * s_.per_step
        s_.tok_emb = nn.Embedding(K, d)
        s_.act_emb = nn.Embedding(A_DIM, d)
        s_.pos = nn.Parameter(torch.zeros(1, s_.T, d))
        s_.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        s_.ln = nn.LayerNorm(d)
        s_.head = nn.Linear(d, K)
    def build(s_, tokens, actions):
        """tokens (B,L,16) long, actions (B,L) long -> (B, L*17, d)"""
        B, L, _ = tokens.shape
        te = s_.tok_emb(tokens)                        # (B,L,16,d)
        ae = s_.act_emb(actions).unsqueeze(2)          # (B,L,1,d)
        x = torch.cat([te, ae], 2).view(B, L * s_.per_step, s_.d)
        return x + s_.pos[:, :x.shape[1]]
    def forward(s_, tokens, actions, need_w=False):
        x = s_.build(tokens, actions)
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        w_last = None
        for b in s_.blocks:
            x, w = b(x, mask, need_w)
            if w is not None:
                w_last = w
        return s_.head(s_.ln(x)), w_last
'''


def _batch_fn():
    """shared sampler; returns a closure over the loaded episodes"""
    import numpy as np
    z = np.load("/out/coinrun.npz")
    n = int(z["n"])
    eps = [{"frames": z[f"f{i}"], "actions": z[f"a{i}"]} for i in range(n)]
    return eps


# ─────────────────────────────────────────────────────────── A · RSSM
@app.function(image=gpu_image, gpu="A10G", timeout=14400, volumes={"/out": vol})
def train_rssm(steps: int = 12000):
    import time
    import numpy as np
    import torch
    import torch.nn.functional as F

    ns = {}; exec(SRC, ns)
    dev = torch.device("cuda")
    torch.manual_seed(0); np.random.seed(0)
    eps = _batch_fn(); train = eps[:-HOLD_OUT]

    B, L = 16, 16
    def batch():
        fr = np.zeros((B, L, 64, 64, 3), np.float32)
        ac = np.zeros((B, L), np.int64)
        for b in range(B):
            e = train[np.random.randint(len(train))]
            t = np.random.randint(0, len(e["frames"]) - L)
            fr[b] = e["frames"][t:t+L] / 255.0
            ac[b] = e["actions"][t:t+L]
        return torch.tensor(fr).to(dev), torch.tensor(ac).to(dev)

    m = ns["RSSM"]().to(dev)
    npar = sum(p.numel() for p in m.parameters())
    print(f"[rssm] parameters: {npar:,}", flush=True)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4, eps=1e-5)
    A_DIM = ns["A_DIM"]

    def kl(m1, s1, m2, s2):
        return (torch.log(s2/s1) + (s1**2 + (m1-m2)**2)/(2*s2**2) - 0.5).sum(-1)

    t0 = time.time()
    for step in range(steps):
        fr, ac = batch()
        a1h = F.one_hot(ac, A_DIM).float()
        emb = m.enc(fr.reshape(B*L, 64, 64, 3)).view(B, L, -1)
        h = torch.zeros(B, ns["H"], device=dev)
        s = torch.zeros(B, ns["S"], device=dev)
        l_rec = l_kl = 0.0
        for t in range(L):
            if t > 0:
                h = m.step_h(h, s, a1h[:, t-1])
            pm, ps = m.prior(h)
            qm, qs = m.posterior(h, emb[:, t])
            s = qm + qs * torch.randn_like(qs)
            l_rec = l_rec + ((m.dec(m.feat(h, s)) - fr[:, t])**2).sum(dim=(1,2,3)).mean()
            k = (0.8*kl(qm.detach(), qs.detach(), pm, ps).mean()
                 + 0.2*kl(qm, qs, pm.detach(), ps.detach()).mean())
            l_kl = l_kl + torch.clamp(k, min=1.0)
        loss = (l_rec + l_kl) / L
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 100.0); opt.step()
        if step % 1000 == 0:
            print(f"[rssm] {step:6d}  rec {float(l_rec)/L:8.1f}  kl {float(l_kl)/L:6.2f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"sd": m.state_dict(), "npar": npar,
                "train_s": time.time()-t0}, "/out/rssm.pt")
    vol.commit()
    print(f"[rssm] done in {(time.time()-t0)/60:.0f} min", flush=True)


# ────────────────────────────────────────────────────── B · IRIS-mini
@app.function(image=gpu_image, gpu="A10G", timeout=14400, volumes={"/out": vol})
def train_iris(tok_steps: int = 20000, gpt_steps: int = 60000):
    import time
    import numpy as np
    import torch
    import torch.nn.functional as F

    ns = {}; exec(SRC, ns)
    dev = torch.device("cuda")
    torch.manual_seed(0); np.random.seed(0)
    eps = _batch_fn(); train = eps[:-HOLD_OUT]
    NTOK, GRID = ns["NTOK"], ns["GRID"]

    # ---- stage 1: the tokenizer (learn the vocabulary)
    tok = ns["Tokenizer"]().to(dev)
    ntok_par = sum(p.numel() for p in tok.parameters())
    print(f"[iris] tokenizer parameters: {ntok_par:,}", flush=True)
    opt = torch.optim.Adam(tok.parameters(), lr=3e-4)
    t0 = time.time()
    for step in range(tok_steps):
        fr = np.zeros((32, 64, 64, 3), np.float32)
        for b in range(32):
            e = train[np.random.randint(len(train))]
            fr[b] = e["frames"][np.random.randint(len(e["frames"]))] / 255.0
        x = torch.tensor(fr).to(dev)
        xh, idx, commit = tok(x)
        loss = ((xh - x)**2).sum(dim=(1,2,3)).mean() + 0.25 * commit * 4096
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"[iris] tok {step:6d}  recon {float(((xh-x)**2).mean()):.5f}"
                  f"  perplexity {tok.vq.perplexity():6.1f}/{ns['K']}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
    tok_time = time.time() - t0
    torch.save({"sd": tok.state_dict(), "npar": ntok_par,
                "perplexity": tok.vq.perplexity()}, "/out/iris_tok.pt")
    vol.commit()

    # ---- stage 2: tokenise the corpus once, then train the GPT on tokens
    tok.eval()
    print("[iris] tokenising the corpus…", flush=True)
    tokenised = []
    with torch.no_grad():
        for e in eps:
            fr = torch.tensor(e["frames"] / 255.0, dtype=torch.float32).to(dev)
            ids = []
            for i in range(0, len(fr), 256):
                _, idx, _ = tok(fr[i:i+256])
                ids.append(idx.view(len(idx), -1).cpu())
            tokenised.append({"tokens": torch.cat(ids).numpy().astype(np.int64),
                              "actions": e["actions"]})
    np.savez_compressed("/out/coinrun_tokens.npz", n=len(tokenised),
        **{f"t{i}": d["tokens"] for i, d in enumerate(tokenised)},
        **{f"a{i}": d["actions"] for i, d in enumerate(tokenised)})
    vol.commit()

    gpt = ns["IRISGPT"]().to(dev)
    ngpt = sum(p.numel() for p in gpt.parameters())
    print(f"[iris] gpt parameters: {ngpt:,}  (total {ntok_par+ngpt:,})", flush=True)
    opt = torch.optim.Adam(gpt.parameters(), lr=3e-4)
    CTX = gpt.ctx_steps
    tr = tokenised[:-HOLD_OUT]
    t1 = time.time()
    for step in range(gpt_steps):
        B = 16
        tk = np.zeros((B, CTX, NTOK), np.int64)
        ac = np.zeros((B, CTX), np.int64)
        for b in range(B):
            d = tr[np.random.randint(len(tr))]
            t = np.random.randint(0, len(d["tokens"]) - CTX)
            tk[b] = d["tokens"][t:t+CTX]; ac[b] = d["actions"][t:t+CTX]
        tk_t = torch.tensor(tk).to(dev); ac_t = torch.tensor(ac).to(dev)
        logits, _ = gpt(tk_t, ac_t)
        # predict every frame token from everything before it
        flat_t = torch.cat([tk_t, torch.zeros(B, CTX, 1, dtype=torch.long,
                                              device=dev)], 2).view(B, -1)
        tgt = flat_t[:, 1:]
        pred = logits[:, :-1]
        # a position predicts the NEXT entry in the flattened stream.
        # positions 0..14 of a step predict frame tokens 1..15 of that step;
        # position 16 (the ACTION token) predicts token 0 of the NEXT frame —
        # that is the cross-frame prediction, and it is the whole point.
        # position 15 would predict the action placeholder: drop it.
        keep = (torch.arange(pred.shape[1], device=dev) % gpt.per_step) != (NTOK - 1)
        loss = F.cross_entropy(pred[:, keep].reshape(-1, ns["K"]),
                               tgt[:, keep].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0); opt.step()
        if step % 1000 == 0:
            print(f"[iris] gpt {step:6d}  ce {float(loss):.4f}"
                  f"  ({time.time()-t1:.0f}s)", flush=True)
    torch.save({"sd": gpt.state_dict(), "npar": ngpt,
                "train_s": tok_time + (time.time()-t1)}, "/out/iris_gpt.pt")
    vol.commit()
    print(f"[iris] done in {(tok_time + time.time()-t1)/60:.0f} min", flush=True)


@app.function(image=gpu_image, gpu="A10G", timeout=14400, volumes={"/out": vol})
def train_gpt(gpt_steps: int = 60000):
    """Stage 2 only — reuses the tokenizer and the tokenised corpus on the volume."""
    import time
    import numpy as np
    import torch
    import torch.nn.functional as F

    ns = {}; exec(SRC, ns)
    dev = torch.device("cuda")
    torch.manual_seed(0); np.random.seed(0)
    NTOK, K = ns["NTOK"], ns["K"]

    z = np.load("/out/coinrun_tokens.npz"); n = int(z["n"])
    tokenised = [{"tokens": z[f"t{i}"], "actions": z[f"a{i}"]} for i in range(n)]
    tok_ck = torch.load("/out/iris_tok.pt", map_location="cpu")
    tok_time = 0.0

    gpt = ns["IRISGPT"]().to(dev)
    ngpt = sum(p.numel() for p in gpt.parameters())
    print(f"[iris] gpt parameters: {ngpt:,}  "
          f"(total {tok_ck['npar']+ngpt:,})", flush=True)
    opt = torch.optim.Adam(gpt.parameters(), lr=3e-4)
    CTX = gpt.ctx_steps
    tr = tokenised[:-HOLD_OUT]
    t1 = time.time()
    for step in range(gpt_steps):
        B = 16
        tk = np.zeros((B, CTX, NTOK), np.int64)
        ac = np.zeros((B, CTX), np.int64)
        for b in range(B):
            d = tr[np.random.randint(len(tr))]
            t = np.random.randint(0, len(d["tokens"]) - CTX)
            tk[b] = d["tokens"][t:t+CTX]; ac[b] = d["actions"][t:t+CTX]
        tk_t = torch.tensor(tk).to(dev); ac_t = torch.tensor(ac).to(dev)
        logits, _ = gpt(tk_t, ac_t)
        flat_t = torch.cat([tk_t, torch.zeros(B, CTX, 1, dtype=torch.long,
                                              device=dev)], 2).view(B, -1)
        tgt = flat_t[:, 1:]; pred = logits[:, :-1]
        keep = (torch.arange(pred.shape[1], device=dev) % gpt.per_step) != (NTOK - 1)
        loss = F.cross_entropy(pred[:, keep].reshape(-1, K), tgt[:, keep].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0); opt.step()
        if step % 1000 == 0:
            print(f"[iris] gpt {step:6d}  ce {float(loss):.4f}"
                  f"  ({time.time()-t1:.0f}s)", flush=True)
    torch.save({"sd": gpt.state_dict(), "npar": ngpt,
                "train_s": tok_time + (time.time()-t1)}, "/out/iris_gpt.pt")
    vol.commit()
    print(f"[iris] gpt done in {(time.time()-t1)/60:.0f} min", flush=True)


# ─────────────────────────────────────────── the comparison figures
@app.function(image=gpu_image, gpu="A10G", timeout=7200, volumes={"/out": vol})
def compare():
    """Every figure the lecture claims. Nothing here is a schematic."""
    import os, time, json
    import numpy as np
    import torch
    import torch.nn.functional as F
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from PIL import Image

    PAPER, INK, MUTED = "#FBF9F1", "#16130D", "#6D665A"
    TEAL, GOLD, CLAY, BLUE = "#2E8F8F", "#DD9F3E", "#C96442", "#4A7FA8"
    rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif",
        "axes.titlesize": 13, "axes.titleweight": "bold"})
    os.makedirs("/out/figs", exist_ok=True)

    ns = {}; exec(SRC, ns)
    dev = torch.device("cuda")
    K, NTOK, GRID = ns["K"], ns["NTOK"], ns["GRID"]
    A_DIM = ns["A_DIM"]

    z = np.load("/out/coinrun.npz"); n = int(z["n"])
    eps = [{"frames": z[f"f{i}"], "actions": z[f"a{i}"]} for i in range(n)]
    held = eps[-HOLD_OUT:]

    rssm_ck = torch.load("/out/rssm.pt", map_location=dev)
    rssm = ns["RSSM"]().to(dev); rssm.load_state_dict(rssm_ck["sd"]); rssm.eval()
    tok_ck = torch.load("/out/iris_tok.pt", map_location=dev)
    tok = ns["Tokenizer"]().to(dev); tok.load_state_dict(tok_ck["sd"]); tok.eval()
    gpt_ck = torch.load("/out/iris_gpt.pt", map_location=dev)
    gpt = ns["IRISGPT"]().to(dev); gpt.load_state_dict(gpt_ck["sd"]); gpt.eval()
    print(f"rssm {rssm_ck['npar']:,} · iris {tok_ck['npar']+gpt_ck['npar']:,}", flush=True)

    def up(a, k=4):
        return np.array(Image.fromarray((np.clip(a,0,1)*255).astype(np.uint8))
                        .resize((64*k, 64*k), Image.NEAREST))/255.0

    stats = {"rssm_params": int(rssm_ck["npar"]),
             "iris_params": int(tok_ck["npar"] + gpt_ck["npar"]),
             "codebook_perplexity": float(tok_ck["perplexity"]), "K": K}

    # ════════════ FIG 1 · the vocabulary made visible
    # A word's meaning is the set of real 16x16 patches assigned to it. Decoding
    # one code replicated over the whole grid hands the decoder a constant field
    # and paints a flat wash — technically its output, visually nothing.
    acc = np.zeros((K, 16, 16, 3), np.float32)
    cnt = np.zeros(K, np.int64)
    for ep_ in held[:24]:
        f_ = ep_["frames"][:90] / 255.0
        with torch.no_grad():
            _, ii_, _ = tok(torch.tensor(f_, dtype=torch.float32).to(dev))
        ii_ = ii_.cpu().numpy()
        for b_ in range(len(f_)):
            for gi in range(GRID):
                for gj in range(GRID):
                    c_ = ii_[b_, gi, gj]
                    acc[c_] += f_[b_, gi*16:(gi+1)*16, gj*16:(gj+1)*16]
                    cnt[c_] += 1
    patches = acc / np.maximum(cnt, 1)[:, None, None, None]
    for c_ in range(K):
        if cnt[c_] == 0:
            patches[c_] = 0.86
    print(f"vocabulary gallery: {int((cnt>0).sum())}/{K} words seen in "
          f"{int(cnt.sum()):,} real patches", flush=True)
    order = np.argsort(-tok.vq.cluster.cpu().numpy())      # most-used first
    NR, NC = 16, 32                                        # 512 = the WHOLE dictionary
    fig, axes = plt.subplots(NR, NC, figsize=(20.4, 8.9))
    for i, ax in enumerate(axes.flat):
        ax.imshow(patches[order[i]]); ax.axis("off")
    plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005,
                        wspace=0.06, hspace=0.06)
    plt.savefig("/out/figs/fig_vocabulary.png", dpi=115, bbox_inches="tight")
    plt.close(fig)
    print("fig_vocabulary", flush=True)

    # ════════════ FIG 1b · is 16 words per frame actually enough?
    ep_r = held[1]
    picks_r = [30, 90, 150, 210, 270, 330]
    picks_r = [p for p in picks_r if p < len(ep_r["frames"])]
    xr = torch.tensor(ep_r["frames"][picks_r]/255.0, dtype=torch.float32).to(dev)
    with torch.no_grad():
        xr_h, _, _ = tok(xr)
        rec_mse = float(((xr_h - xr)**2).mean())
    fig, axes = plt.subplots(2, len(picks_r), figsize=(2.3*len(picks_r), 5.2))
    for i in range(len(picks_r)):
        axes[0, i].imshow(up(xr[i].cpu().numpy(), 3)); axes[0, i].axis("off")
        axes[1, i].imshow(up(xr_h[i].clamp(0,1).cpu().numpy(), 3)); axes[1, i].axis("off")
    axes[0,0].text(-0.10, 0.5, "the frame", transform=axes[0,0].transAxes,
                   ha="right", va="center", fontsize=13, color=INK, fontweight="bold")
    axes[1,0].text(-0.10, 0.5, "rebuilt from\n16 numbers",
                   transform=axes[1,0].transAxes, ha="right", va="center",
                   fontsize=13, color=CLAY, fontweight="bold")
    fig.text(0.5, 0.03, f"pixel MSE {rec_mse:.5f} on episodes the tokenizer never saw · "
             "12,288 numbers in, 16 out, and the game is still legible",
             ha="center", fontsize=12.5, color=INK)
    plt.subplots_adjust(left=0.11, top=0.985, bottom=0.105, wspace=0.05, hspace=0.06)
    plt.savefig("/out/figs/fig_reconstruction.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    stats["recon_mse_heldout"] = rec_mse
    print(f"fig_reconstruction · heldout mse {rec_mse:.5f}", flush=True)

    # ════════════ FIG 2 · one frame, two descriptions
    ep = held[0]; t0 = 120
    x = torch.tensor(ep["frames"][t0:t0+1]/255.0, dtype=torch.float32).to(dev)
    with torch.no_grad():
        xh, idx, _ = tok(x)
        e = rssm.enc(x)
        h0 = torch.zeros(1, ns["H"], device=dev); s0 = torch.zeros(1, ns["S"], device=dev)
        qm, _ = rssm.posterior(h0, e)
    tokens = idx[0].cpu().numpy()
    fig = plt.figure(figsize=(15.0, 6.6))
    HEAD_Y = 0.855                       # ONE baseline for all three headers
    axf = fig.add_axes([0.045, 0.17, 0.22, 0.63]); axf.imshow(up(x[0].cpu().numpy()))
    axf.axis("off")
    fig.text(0.155, HEAD_Y, "one CoinRun frame", fontsize=14, ha="center", color=INK)
    axt = fig.add_axes([0.325, 0.17, 0.20, 0.63])
    axt.imshow(tokens, cmap="tab20"); axt.set_xticks([]); axt.set_yticks([])
    for i in range(GRID):
        for j in range(GRID):
            axt.text(j, i, str(tokens[i,j]), ha="center", va="center",
                     fontsize=13, color="white", fontweight="bold")
    fig.text(0.425, HEAD_Y, "IRIS · 16 words", fontsize=14, ha="center",
             color=CLAY, fontweight="bold")
    axn = fig.add_axes([0.575, 0.17, 0.40, 0.63]); axn.axis("off")
    fig.text(0.575, HEAD_Y, "RSSM · 32 continuous numbers", fontsize=14,
             ha="left", color=TEAL, fontweight="bold")
    v = qm[0].cpu().numpy()
    for r in range(4):
        axn.text(0, 0.80 - r*0.17, "  ".join(f"{val:+.2f}" for val in v[r*8:(r+1)*8]),
                 fontsize=12.5, family="monospace", color=INK)
    axn.text(0, 0.06, "you can read the tokens as words in a dictionary.\n"
                      "you cannot enumerate the space these numbers live in.",
             fontsize=12, color=MUTED, style="italic")
    plt.savefig("/out/figs/fig_two_descriptions.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("fig_two_descriptions", flush=True)

    # ════════════ FIG 3 · interpolation: smooth vs snapping
    fA = torch.tensor(held[0]["frames"][60:61]/255.0, dtype=torch.float32).to(dev)
    fB = torch.tensor(held[1]["frames"][200:201]/255.0, dtype=torch.float32).to(dev)
    NA = 13
    alphas = np.linspace(0, 1, NA)
    with torch.no_grad():
        h0 = torch.zeros(1, ns["H"], device=dev)
        qa, _ = rssm.posterior(h0, rssm.enc(fA))
        qb, _ = rssm.posterior(h0, rssm.enc(fB))
        rssm_row = [rssm.dec(rssm.feat(h0, (1-a)*qa + a*qb))[0].clamp(0,1).cpu().numpy()
                    for a in alphas]
        za, zb = tok.encode(fA), tok.encode(fB)
        iris_row, iris_ids = [], []
        for a in alphas:
            zi = (1-a)*za + a*zb
            q, ii, _ = tok.vq(zi)                     # snaps to nearest words
            iris_row.append(tok.decode_q(q)[0].clamp(0,1).cpu().numpy())
            iris_ids.append(ii.view(-1).cpu().numpy())
    # HOW MUCH did the picture change at each step? this is the whole point,
    # and it is a number rather than something you have to squint for.
    d_r = [float(np.abs(rssm_row[i+1] - rssm_row[i]).mean()) for i in range(NA-1)]
    d_i = [float(np.abs(iris_row[i+1] - iris_row[i]).mean()) for i in range(NA-1)]
    n_sw = [int((iris_ids[i+1] != iris_ids[i]).sum()) for i in range(NA-1)]
    stats["interp_step_rssm"] = d_r
    stats["interp_step_iris"] = d_i
    stats["interp_tokens_changed"] = n_sw

    show = list(range(0, NA, 2))                      # 7 frames on the strips
    fig = plt.figure(figsize=(15.4, 8.2))
    gs = fig.add_gridspec(3, len(show), height_ratios=[1.0, 1.0, 0.95],
                          left=0.11, right=0.985, top=0.885, bottom=0.10,
                          wspace=0.05, hspace=0.16)
    for c, i in enumerate(show):
        a0 = fig.add_subplot(gs[0, c]); a0.imshow(up(rssm_row[i], 3)); a0.axis("off")
        a0.set_title(f"{alphas[i]:.2f}", fontsize=11, color=MUTED)
        a1 = fig.add_subplot(gs[1, c]); a1.imshow(up(iris_row[i], 3)); a1.axis("off")
        if c == 0:
            a0.text(-0.09, 0.5, "RSSM\ncontinuous", transform=a0.transAxes,
                    ha="right", va="center", fontsize=14, color=TEAL,
                    fontweight="bold")
            a1.text(-0.09, 0.5, "IRIS\ndiscrete", transform=a1.transAxes,
                    ha="right", va="center", fontsize=14, color=CLAY,
                    fontweight="bold")
    axd = fig.add_subplot(gs[2, :])
    mid = (alphas[:-1] + alphas[1:]) / 2
    axd.plot(mid, d_r, "-o", lw=3.0, ms=8, color=TEAL, label="RSSM · continuous")
    axd.plot(mid, d_i, "-o", lw=3.0, ms=8, color=CLAY, label="IRIS · discrete")
    top = max(max(d_r), max(d_i))
    axd.set_ylim(0, top * 1.60)
    # the per-step token counts sit on their OWN band, well clear of the legend
    axd.text(mid[0] - 0.045, top * 1.30, "tokens that flipped:", ha="left",
             va="center", fontsize=13, color=CLAY, fontweight="bold")
    for m, n in zip(mid, n_sw):
        axd.text(m, top * 1.14, f"{n}", ha="center", va="center", fontsize=13,
                 color=CLAY, fontweight="bold")
    axd.set_xlabel("position along the walk", fontsize=14)
    axd.set_ylabel("how much the\npicture changed", fontsize=13)
    axd.tick_params(labelsize=12)
    axd.legend(frameon=False, fontsize=14, ncol=2, loc="upper right")
    axd.set_title("The continuous walk moves a little at every step. The discrete "
                  "walk stands still, then jumps.", loc="left", fontsize=15)
    fig.text(0.5, 0.018, "nothing happens until a token flips — and then the "
             "picture jumps",
             ha="center", fontsize=13.5, color=MUTED)
    plt.savefig("/out/figs/fig_interpolation.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("fig_interpolation", flush=True)

    # ════════════ FIG 4 · the two latent spaces, projected
    from sklearn.decomposition import PCA
    ep = held[2]; T = min(300, len(ep["frames"]))
    fr = torch.tensor(ep["frames"][:T]/255.0, dtype=torch.float32).to(dev)
    with torch.no_grad():
        h = torch.zeros(1, ns["H"], device=dev); s = torch.zeros(1, ns["S"], device=dev)
        a1h = F.one_hot(torch.tensor(ep["actions"][:T]).to(dev), A_DIM).float()
        embs = rssm.enc(fr)
        cont = []
        for t in range(T):
            if t > 0:
                h = rssm.step_h(h, s, a1h[t-1:t])
            qm, _ = rssm.posterior(h, embs[t:t+1]); s = qm
            cont.append(qm[0].cpu().numpy())
        _, idxs, _ = tok(fr)
        disc = idxs.view(T, -1).cpu().numpy()
    cont = np.array(cont)
    onehot = np.zeros((T, K), np.float32)
    for t in range(T):
        for c in disc[t]:
            onehot[t, c] += 1
    # 300 points is a hairball on a projector — 120 consecutive steps reads.
    SEG = min(120, T)
    p_c = PCA(2).fit_transform(cont)[:SEG]
    p_d = PCA(2).fit_transform(onehot)[:SEG]
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.2))
    for ax, p, title, col in ((axes[0], p_c, "RSSM · continuous latent", TEAL),
                              (axes[1], p_d, "IRIS · bag of visual words", CLAY)):
        ax.plot(p[:,0], p[:,1], "-", lw=1.3, color=MUTED, alpha=0.45, zorder=1)
        sc = ax.scatter(p[:,0], p[:,1], c=np.arange(SEG), cmap="viridis", s=46,
                        zorder=2, edgecolors="none")
        ax.scatter(p[0,0], p[0,1], s=190, marker="o", facecolors="none",
                   edgecolors=INK, lw=2.0, zorder=3)
        ax.annotate("start", (p[0,0], p[0,1]), textcoords="offset points",
                    xytext=(12, 10), fontsize=12, fontweight="bold", color=INK)
        ax.set_title(title, loc="left", color=col, pad=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.margins(0.12)
    plt.tight_layout(rect=[0.0, 0.055, 0.955, 0.92])
    cax = fig.add_axes([0.965, 0.14, 0.012, 0.66])
    plt.colorbar(sc, cax=cax, label="timestep")
    fig.text(0.48, 0.015, "left: short, even steps along a smooth path  ·  "
             "right: the state sits still, then jumps to another configuration",
             ha="center", fontsize=12.5, color=INK)
    plt.savefig("/out/figs/fig_latent_spaces.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("fig_latent_spaces", flush=True)

    # ════════════ FIG 5 · codebook health
    use = tok.vq.cluster.cpu().numpy(); use = use/use.sum()
    srt = np.sort(use)[::-1]
    dead = int((use < 1e-6).sum())
    fig, ax = plt.subplots(figsize=(11.4, 4.4))
    ax.bar(range(K), srt, color=CLAY, width=1.0, zorder=2)
    ax.axhline(1.0/K, ls=":", lw=1.6, color=TEAL, zorder=3)
    ax.text(K*0.985, 1.0/K*1.35, "perfectly even use would be this flat line",
            ha="right", fontsize=11.5, color=TEAL)
    ax.set_xlim(0, K); ax.set_ylim(0, srt.max()*1.55)
    ax.set_xlabel(f"codebook entry, sorted by how often it is used (all {K})")
    ax.set_ylabel("share of tokens")
    ax.text(0.985, 0.86,
            f"perplexity {tok_ck['perplexity']:.0f} of {K}\n"
            f"{K-dead} of {K} codes alive\n"
            f"the busiest code takes only {srt[0]*100:.1f}% of all tokens",
            transform=ax.transAxes, ha="right", va="top", fontsize=13,
            color=TEAL, fontweight="bold", linespacing=1.5)
    plt.tight_layout()
    plt.savefig("/out/figs/fig_codebook.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    stats["codes_alive"] = K - dead
    print("fig_codebook", flush=True)

    # ════════════ FIG 6 · dream quality, side by side
    CTX, HOR = gpt.ctx_steps, 12
    ep = held[3]; t0 = 100
    fr = torch.tensor(ep["frames"][t0:t0+CTX+HOR]/255.0, dtype=torch.float32).to(dev)
    acts = torch.tensor(ep["actions"][t0:t0+CTX+HOR]).to(dev)
    with torch.no_grad():
        # RSSM dream
        a1h = F.one_hot(acts, A_DIM).float()
        embs = rssm.enc(fr)
        h = torch.zeros(1, ns["H"], device=dev); s = torch.zeros(1, ns["S"], device=dev)
        for t in range(CTX):
            if t > 0: h = rssm.step_h(h, s, a1h[t-1:t])
            qm, _ = rssm.posterior(h, embs[t:t+1]); s = qm
        rssm_dream = []
        for k in range(HOR):
            h = rssm.step_h(h, s, a1h[CTX+k-1:CTX+k])
            pm, _ = rssm.prior(h); s = pm
            rssm_dream.append(rssm.dec(rssm.feat(h, s))[0].clamp(0,1).cpu().numpy())
        # IRIS dream — genuine autoregressive sampling, one token at a time.
        # The context ends with the previous step's ACTION token, whose output
        # predicts token 0 of the frame we are about to imagine. Each sampled
        # token is appended to the sequence before the next one is drawn.
        _, idx_all, _ = tok(fr)
        toks = idx_all.view(-1, NTOK)[:CTX].unsqueeze(0)

        def iris_rollout(seed):
            torch.manual_seed(seed)
            out = []
            hist_t = toks.clone(); hist_a = acts[:CTX].unsqueeze(0)
            for k in range(HOR):
                # keep at most ctx_steps-1 whole steps so the new step fits in `pos`
                ht = hist_t[:, -(CTX-1):]; ha = hist_a[:, -(CTX-1):]
                x = gpt.build(ht, ha)                   # (1, L*17, d), pos added
                nxt = []
                for j in range(NTOK):
                    Tn = x.shape[1]
                    m = torch.triu(torch.ones(Tn, Tn, device=dev,
                                              dtype=torch.bool), 1)
                    hcur = x
                    for b in gpt.blocks:
                        hcur, _ = b(hcur, m, False)
                    probs = F.softmax(gpt.head(gpt.ln(hcur))[0, -1], -1)
                    tk = torch.multinomial(probs, 1)
                    nxt.append(int(tk))
                    x = torch.cat([x, gpt.tok_emb(tk.view(1, 1)) +
                                      gpt.pos[:, Tn:Tn+1]], 1)
                nt = torch.tensor(nxt, device=dev).view(1, GRID, GRID)
                out.append(tok.decode_tokens(nt)[0].clamp(0,1).cpu().numpy())
                hist_t = torch.cat([hist_t, nt.view(1, 1, NTOK)], 1)
                hist_a = torch.cat([hist_a, acts[CTX+k:CTX+k+1].view(1, 1)], 1)
            return out

        # sampling is stochastic, so one rollout is not a measurement.
        # Take N seeded rollouts; show the first, report the spread.
        N_ROLL = 5
        rolls = [iris_rollout(s) for s in range(N_ROLL)]
        iris_dream = rolls[0]
    picks = [0, 3, 6, 9, 11]
    fig, axes = plt.subplots(3, len(picks), figsize=(17.2, 7.4))
    for i, k in enumerate(picks):
        axes[0,i].imshow(up(fr[CTX+k].cpu().numpy(), 3)); axes[0,i].axis("off")
        axes[0,i].set_title(f"+{k+1}", fontsize=10, color=MUTED)
        axes[1,i].imshow(up(rssm_dream[k], 3)); axes[1,i].axis("off")
        axes[2,i].imshow(up(iris_dream[k], 3)); axes[2,i].axis("off")
    for r, (lab, col) in enumerate([("the real game", INK), ("RSSM", TEAL),
                                    ("IRIS", CLAY)]):
        axes[r,0].text(-0.12, 0.5, lab, transform=axes[r,0].transAxes, ha="right",
                       va="center", fontsize=13, color=col, fontweight="bold")
    # sharpness: mean gradient magnitude
    def sharp(ims):
        g = [np.abs(np.diff(im, axis=0)).mean() + np.abs(np.diff(im, axis=1)).mean()
             for im in ims]
        return float(np.mean(g))
    s_real = sharp([fr[CTX+k].cpu().numpy() for k in range(HOR)])
    s_r = sharp(rssm_dream)
    s_all = [sharp(r) for r in rolls]
    s_i, s_i_sd = float(np.mean(s_all)), float(np.std(s_all))
    stats.update({"sharp_real": s_real, "sharp_rssm": s_r, "sharp_iris": s_i,
                  "sharp_iris_sd": s_i_sd, "sharp_iris_rolls": s_all,
                  "n_rollouts": N_ROLL})
    fig.text(0.5, 0.02, f"edge sharpness — real {s_real:.4f} · RSSM {s_r:.4f} · "
             f"IRIS {s_i:.4f} ± {s_i_sd:.4f} over {N_ROLL} seeded rollouts",
             ha="center", fontsize=12.5, color=INK)
    plt.subplots_adjust(left=0.115, top=0.955, bottom=0.085, wspace=0.05, hspace=0.08)
    plt.savefig("/out/figs/fig_dreams.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"fig_dreams · sharpness real {s_real:.4f} rssm {s_r:.4f} iris {s_i:.4f}",
          flush=True)

    # ════════════ FIG 7 · how far back does attention need? (causal masking)
    ep = held[1]
    fr = torch.tensor(ep["frames"][:CTX+1]/255.0, dtype=torch.float32).to(dev)
    acts = torch.tensor(ep["actions"][:CTX+1]).to(dev)
    with torch.no_grad():
        _, idx_all, _ = tok(fr)
        toks = idx_all.view(-1, NTOK)[:CTX].unsqueeze(0)
        a_in = acts[:CTX].unsqueeze(0)
        # score ONLY the newest frame's 16 tokens, while progressively hiding
        # the oldest frames. token 0 of the last step is predicted from the
        # previous step's action position (base-1); tokens 1..15 from base+0..14.
        base = (CTX - 1) * gpt.per_step
        sel = torch.tensor([base - 1] + [base + j for j in range(NTOK - 1)],
                           device=dev)
        flat = torch.cat([toks, torch.zeros(1, CTX, 1, dtype=torch.long,
                                            device=dev)], 2).view(1, -1)
        losses = []
        for keep in range(1, CTX+1):
            t_in = toks.clone(); ai = a_in.clone()
            t_in[:, :CTX-keep] = 0            # hide the oldest frames…
            ai[:, :CTX-keep] = 0              # …and the actions that went with them
            logits, _ = gpt(t_in, ai)
            tgt = flat[:, 1:]; pred = logits[:, :-1]
            losses.append(float(F.cross_entropy(pred[0, sel].reshape(-1, K),
                                                tgt[0, sel].reshape(-1))))
    fig, ax = plt.subplots(figsize=(11.4, 5.0))
    xs = list(range(1, CTX+1))
    ax.plot(xs, losses, "-o", lw=2.8, ms=9, color=CLAY, zorder=3)
    ax.set_xlim(0.6, CTX + 0.5)
    ax.set_ylim(-0.06, max(losses) * 1.30)
    ax.set_xticks(xs)
    ax.axvspan(2.6, CTX + 0.5, color=TEAL, alpha=0.06, zorder=0)
    ax.text(CTX, max(losses) * 0.10, "everything past three frames\nbuys nothing",
            ha="right", va="bottom", fontsize=12.5, color=TEAL, fontweight="bold")
    ax.annotate(f"one extra frame → {losses[0]/max(losses[1],1e-9):,.0f}× better\n"
                "(this is where motion lives)",
                xy=(2.06, losses[1]), xytext=(2.75, max(losses) * 0.62),
                fontsize=13, color=CLAY, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="->", color=CLAY, lw=1.8,
                                connectionstyle="arc3,rad=0.18"))
    # labels go BELOW the curve where it is falling, above where it is flat
    for k, (x, y) in enumerate(zip(xs[:3], losses[:3])):
        off = (0, 15) if k == 0 else ((14, -6) if k == 1 else (16, 4))
        ax.annotate(f"{y:.2f}" if y > 0.1 else f"{y:.3f}", (x, y),
                    textcoords="offset points", xytext=off, ha="center",
                    fontsize=12, color=MUTED)
    ax.set_xlabel("how many recent frames the model is allowed to see")
    ax.set_ylabel("prediction error (cross-entropy)")
    plt.tight_layout()
    plt.savefig("/out/figs/fig_context_need.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    stats["context_curve"] = losses
    print("fig_context_need", flush=True)

    # ════════════ FIG 8 · efficiency
    def bench_rssm(steps=200):
        h = torch.zeros(1, ns["H"], device=dev); s = torch.zeros(1, ns["S"], device=dev)
        a = F.one_hot(torch.zeros(1, dtype=torch.long, device=dev), A_DIM).float()
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad():
            for _ in range(steps):
                h = rssm.step_h(h, s, a); pm, _ = rssm.prior(h); s = pm
        torch.cuda.synchronize(); return steps/(time.time()-t)
    def bench_iris(steps=20, ctx=None):
        ctx = ctx or gpt.ctx_steps
        t_in = torch.zeros(1, ctx, NTOK, dtype=torch.long, device=dev)
        a_in = torch.zeros(1, ctx, dtype=torch.long, device=dev)
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad():
            for _ in range(steps):
                for _ in range(NTOK):
                    gpt(t_in, a_in)
        torch.cuda.synchronize(); return steps/(time.time()-t)
    fps_r = bench_rssm(); fps_i = bench_iris()
    ctxs = [2, 4, 8]
    fps_i_ctx = [bench_iris(10, c) for c in ctxs]
    stats.update({"fps_rssm": fps_r, "fps_iris": fps_i,
                  "fps_iris_by_ctx": dict(zip(map(str, ctxs), fps_i_ctx))})
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8))
    axes[0].bar(["RSSM", "IRIS"], [fps_r, fps_i], color=[TEAL, CLAY], width=0.5)
    axes[0].set_ylabel("imagined frames per second"); axes[0].set_yscale("log")
    axes[0].set_ylim(1, fps_r * 12)                 # headroom so labels clear the title
    for i, v in enumerate([fps_r, fps_i]):
        axes[0].text(i, v * 1.35, f"{v:,.0f}", ha="center", fontsize=15,
                     fontweight="bold", color=[TEAL, CLAY][i])
    axes[0].set_title(f"Imagination throughput — {fps_r/max(fps_i,1e-9):,.0f}× apart",
                      loc="left", pad=14)
    axes[1].plot(ctxs, fps_i_ctx, "-o", lw=2.8, ms=8, color=CLAY,
                 label="IRIS · 16 tokens per frame")
    axes[1].axhline(fps_r, color=TEAL, lw=2.8, label="RSSM · one GRU step")
    axes[1].set_xlabel("context length (frames)")
    axes[1].set_ylabel("frames per second"); axes[1].set_yscale("log")
    axes[1].set_ylim(min(fps_i_ctx) / 6, fps_r * 30)   # keep both lines off the spines
    axes[1].set_xticks(ctxs)
    axes[1].legend(frameon=False, fontsize=11, loc="upper center")
    axes[1].set_title("Context length is not what costs", loc="left", pad=14)
    axes[1].annotate("flat — the bill is the 16 sequential decodes,\n"
                     "not attention's quadratic term",
                     xy=(ctxs[1], fps_i_ctx[1] * 0.86), xytext=(ctxs[0] + 0.15,
                     min(fps_i_ctx) / 4.4), fontsize=12, color=CLAY,
                     arrowprops=dict(arrowstyle="->", color=CLAY, lw=1.6))
    plt.tight_layout(rect=[0, 0.085, 1, 1])
    fig.text(0.5, 0.02, "measured on one A10G, batch 1. Our transformer recomputes the "
             "whole window for every token; a KV cache is standard and would narrow "
             "this gap substantially.",
             ha="center", fontsize=11, color=MUTED, style="italic")
    plt.savefig("/out/figs/fig_efficiency.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"fig_efficiency · rssm {fps_r:.0f} fps · iris {fps_i:.1f} fps", flush=True)

    json.dump(stats, open("/out/figs/stats.json", "w"), indent=1)
    vol.commit()
    print("ALL FIGURES DONE", flush=True)
    return stats


# ═══════════════════════════ inside the transformer: one real forward pass
@app.function(image=gpu_image, gpu="A10G", timeout=7200, volumes={"/out": vol})
def internals():
    """Trace ONE real forward pass through every layer and draw what happens.

    Nothing here is a schematic. Every number is read out of the trained model
    on a held-out episode.
    """
    import os, json
    import numpy as np
    import torch
    import torch.nn.functional as F
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from PIL import Image

    PAPER, INK, MUTED = "#FBF9F1", "#16130D", "#6D665A"
    TEAL, GOLD, CLAY, BLUE, PURPLE = "#2E8F8F", "#DD9F3E", "#C96442", "#4A7FA8", "#8D6FC0"
    rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 12,
        "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif",
        "axes.titlesize": 14, "axes.titleweight": "bold"})
    os.makedirs("/out/figs", exist_ok=True)

    ns = {}; exec(SRC, ns)
    dev = torch.device("cuda")
    K, NTOK, GRID, A_DIM = ns["K"], ns["NTOK"], ns["GRID"], ns["A_DIM"]

    z = np.load("/out/coinrun.npz"); n = int(z["n"])
    held = [{"frames": z[f"f{i}"], "actions": z[f"a{i}"]} for i in range(n - HOLD_OUT, n)]
    tok = ns["Tokenizer"]().to(dev)
    tok.load_state_dict(torch.load("/out/iris_tok.pt", map_location=dev)["sd"]); tok.eval()
    gpt = ns["IRISGPT"]().to(dev)
    gpt.load_state_dict(torch.load("/out/iris_gpt.pt", map_location=dev)["sd"]); gpt.eval()

    CTX, PS, D = gpt.ctx_steps, gpt.per_step, gpt.d
    NL = len(gpt.blocks)
    HEADS = gpt.blocks[0].attn.num_heads
    ep = held[3]; t0 = 100
    fr = torch.tensor(ep["frames"][t0:t0+CTX]/255.0, dtype=torch.float32).to(dev)
    acts = torch.tensor(ep["actions"][t0:t0+CTX]).to(dev)
    with torch.no_grad():
        _, idx_all, _ = tok(fr)
    toks = idx_all.view(-1, NTOK)[:CTX].unsqueeze(0)          # (1,8,16)
    a_in = acts[:CTX].unsqueeze(0)                            # (1,8)
    T = CTX * PS                                              # 136

    st = {"ctx_steps": CTX, "per_step": PS, "d_model": D, "layers": NL,
          "heads": HEADS, "seq_len": T, "vocab": K}

    def up(a, k=4):
        h_, w_ = a.shape[:2]
        return np.array(Image.fromarray((np.clip(a,0,1)*255).astype(np.uint8))
                        .resize((w_*k*(64//max(w_,1)) if False else w_*k,
                                 h_*k), Image.NEAREST))/255.0

    # ── what does one codebook word actually LOOK like?
    # Decoding a single code replicated across the whole 4x4 grid gives the
    # decoder a constant field, so it paints a flat wash — true, but useless.
    # A word's real meaning is the set of 16x16 image patches assigned to it,
    # so average those. This is what the model actually sees that word as.
    def code_gallery(n_ep=24, n_fr=90):
        acc = np.zeros((K, 16, 16, 3), np.float32)
        cnt = np.zeros(K, np.int64)
        for ep_ in held[:n_ep]:
            f = ep_["frames"][:n_fr] / 255.0
            with torch.no_grad():
                _, ii, _ = tok(torch.tensor(f, dtype=torch.float32).to(dev))
            ii = ii.cpu().numpy()
            for b in range(len(f)):
                for gi in range(GRID):
                    for gj in range(GRID):
                        c = ii[b, gi, gj]
                        acc[c] += f[b, gi*16:(gi+1)*16, gj*16:(gj+1)*16]
                        cnt[c] += 1
        gal = acc / np.maximum(cnt, 1)[:, None, None, None]
        blank = np.full((16, 16, 3), 0.86, np.float32)
        for c in range(K):
            if cnt[c] == 0:
                gal[c] = blank
        print(f"code gallery: {int((cnt>0).sum())}/{K} words seen, "
              f"{int(cnt.sum()):,} patches", flush=True)
        return gal, cnt
    gallery, gal_cnt = code_gallery()

    # ───────── trace: residual stream + per-layer, per-head attention
    with torch.no_grad():
        x0 = gpt.build(toks, a_in)                            # (1,T,d) embed+pos
        mask = torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), 1)
        stream = [x0.clone()]
        attn_l = []                                           # per layer (heads,T,T)
        x = x0
        for b in gpt.blocks:
            h = b.ln1(x)
            a, w = b.attn(h, h, h, attn_mask=mask, need_weights=True,
                          average_attn_weights=False)
            attn_l.append(w[0].cpu().numpy())
            x = x + a
            x = x + b.mlp(b.ln2(x))
            stream.append(x.clone())
        logits = gpt.head(gpt.ln(x))
    QPOS = T - 1                                              # the action token:
    # its output predicts token 0 of the NEXT frame — the position we sample from.

    # ═══════ FIG A · the sequence the transformer actually sees
    fig, ax = plt.subplots(figsize=(17.4, 6.4))
    flat_ids, kinds = [], []
    for s_i in range(CTX):
        flat_ids += [int(v) for v in toks[0, s_i]]; kinds += ["f"]*NTOK
        flat_ids += [int(a_in[0, s_i])];            kinds += ["a"]
    for p, (v, k) in enumerate(zip(flat_ids, kinds)):
        c = GOLD if k == "f" else BLUE
        ax.add_patch(plt.Rectangle((p, 0), 0.92, 1, color=c,
                                   alpha=0.30 if k == "f" else 0.75))
    ax.add_patch(plt.Rectangle((QPOS, 0), 0.92, 1, fill=False, ec=CLAY, lw=3, zorder=5))
    for s_i in range(CTX):
        ax.text(s_i*PS + NTOK/2, -0.36, f"frame {s_i}", ha="center", fontsize=11.5,
                color=INK)
        ax.plot([s_i*PS - 0.05]*2, [-0.15, 1.15], color=MUTED, lw=0.9, alpha=0.5)
    ax.annotate("we sample the next frame's\nfirst token from HERE",
                xy=(QPOS + 0.5, 1.05), xytext=(QPOS - 26, 1.95),
                fontsize=13, color=CLAY, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=CLAY, lw=2))
    ax.text(0, 2.35, f"{CTX} frames × ({NTOK} picture words + 1 action) "
            f"= {T} positions", fontsize=15, color=INK, fontweight="bold")
    ax.text(0, -0.85, "gold = a word from the 512-word picture vocabulary     "
            "blue = the action that was taken", fontsize=12.5, color=MUTED)
    ax.set_xlim(-1, T + 1); ax.set_ylim(-1.5, 2.55); ax.axis("off")
    plt.tight_layout()
    plt.savefig("/out/figs/fig_x_sequence.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print("fig_x_sequence", flush=True)

    # ═══════ FIG B · the token ID is NOT the codebook vector
    tid = int(toks[0, -1, 0])
    cb = tok.vq.emb[tid].cpu().numpy()                        # 64-d, the VQ code
    em = gpt.tok_emb.weight[tid].detach().cpu().numpy()       # 256-d, learned here
    pe = gpt.pos[0, QPOS].detach().cpu().numpy()
    patch = gallery[tid]
    fig = plt.figure(figsize=(16.4, 7.0))
    axp = fig.add_axes([0.035, 0.30, 0.13, 0.52]); axp.imshow(up(patch, 12)); axp.axis("off")
    fig.text(0.10, 0.87, f"word #{tid}", ha="center", fontsize=15, color=INK,
             fontweight="bold")
    fig.text(0.10, 0.20, "what it paints", ha="center", fontsize=12, color=MUTED)
    axc = fig.add_axes([0.225, 0.46, 0.33, 0.20])
    axc.imshow(cb.reshape(1, -1), aspect="auto", cmap="RdBu_r",
               vmin=-abs(cb).max(), vmax=abs(cb).max())
    axc.set_yticks([]); axc.set_xticks([])
    fig.text(0.225, 0.72, f"the CODEBOOK vector — {cb.size} numbers",
             fontsize=14, color=TEAL, fontweight="bold")
    fig.text(0.225, 0.36, "lives in the tokenizer. Used to REBUILD the picture.\n"
             "The transformer never sees it.", fontsize=12.5, color=MUTED)
    axe = fig.add_axes([0.625, 0.46, 0.345, 0.20])
    axe.imshow(em.reshape(1, -1), aspect="auto", cmap="RdBu_r",
               vmin=-abs(em).max(), vmax=abs(em).max())
    axe.set_yticks([]); axe.set_xticks([])
    fig.text(0.625, 0.72, f"the EMBEDDING vector — {em.size} numbers",
             fontsize=14, color=CLAY, fontweight="bold")
    fig.text(0.625, 0.36, "a SEPARATE table the transformer learned for itself.\n"
             "This is what actually enters the stack.", fontsize=12.5, color=MUTED)
    fig.text(0.5, 0.055, f"only the integer {tid} passes between them · "
             "cosine similarity between the two is undefined — "
             "they do not even live in the same space",
             ha="center", fontsize=13, color=INK, style="italic")
    plt.savefig("/out/figs/fig_x_embedding.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print("fig_x_embedding", flush=True)
    st["token_shown"] = tid

    # ═══════ FIG B2 · the learned position table, and the rhythm in it
    with torch.no_grad():
        P = gpt.pos[0, :T].detach()
        Pn = F.normalize(P, dim=-1)
        sim = (Pn @ Pn.t()).cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.2))
    im0 = axes[0].imshow(P.cpu().numpy().T, aspect="auto", cmap="RdBu_r",
                         vmin=-float(P.abs().max()), vmax=float(P.abs().max()))
    axes[0].set_xlabel("position in the window (0 … 135)")
    axes[0].set_ylabel("the 256 numbers")
    axes[0].set_title("the position table the model learned", loc="left")
    for s_i in range(1, CTX):
        axes[0].axvline(s_i*PS, color=INK, lw=0.7, alpha=0.5)
    im1 = axes[1].imshow(sim, cmap="magma", vmin=-1, vmax=1)
    axes[1].set_title("how similar is each position vector to every other?",
                      loc="left")
    axes[1].set_xlabel("position"); axes[1].set_ylabel("position")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    off = [float(np.mean([sim[i, i+k] for i in range(T-k)])) for k in range(1, 35)]
    best = int(np.argmax(off[10:]) + 11)
    stats_extra = {"pos_offset_sim": off, "pos_peak_offset": best}
    fig.text(0.5, 0.02, "nobody told it that a frame is 17 positions long. "
             f"The strongest off-diagonal echo sits at an offset of {best} — "
             "the model discovered the rhythm of its own input.",
             ha="center", fontsize=14, color=INK)
    plt.tight_layout(rect=[0, 0.075, 1, 1])
    plt.savefig("/out/figs/fig_x_pos.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    st.update(stats_extra)
    print(f"fig_x_pos · strongest echo at offset {best}", flush=True)

    # ═══════ FIG C · the causal mask, and its 17-position heartbeat
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.0))
    m = mask.cpu().numpy()
    axes[0].imshow(~m, cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
    for s_i in range(1, CTX):
        axes[0].axhline(s_i*PS-0.5, color=CLAY, lw=0.8, alpha=0.55)
        axes[0].axvline(s_i*PS-0.5, color=CLAY, lw=0.8, alpha=0.55)
    axes[0].set_title(f"all {T} × {T} positions", loc="left")
    axes[0].set_xlabel("…may look at this position"); axes[0].set_ylabel("this position…")
    z0 = 5*PS
    axes[1].imshow(~m[z0:z0+2*PS, z0:z0+2*PS], cmap="Greys", vmin=0, vmax=1.6,
                   interpolation="nearest")
    axes[1].axhline(PS-0.5, color=CLAY, lw=1.6); axes[1].axvline(PS-0.5, color=CLAY, lw=1.6)
    axes[1].axhline(NTOK-0.5, color=BLUE, lw=1.6, ls=":")
    axes[1].axvline(NTOK-0.5, color=BLUE, lw=1.6, ls=":")
    axes[1].set_title("zoomed: two frames' worth", loc="left")
    axes[1].text(0.5, -0.135, "dotted = the action token",
                 transform=axes[1].transAxes, ha="center", va="top",
                 fontsize=12.5, color=BLUE, fontweight="bold")
    fig.text(0.5, 0.015, "black = allowed to look · white = masked out. "
             f"The clay grid every {PS} positions is one frame; "
             "nothing can see its own future.",
             ha="center", fontsize=13, color=INK)
    plt.tight_layout(rect=[0, 0.055, 1, 1])
    plt.savefig("/out/figs/fig_x_mask.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print("fig_x_mask", flush=True)

    # ═══════ FIG D · query · key · value, with the real numbers
    L_SHOW, H_SHOW = 2, 0
    blk = gpt.blocks[L_SHOW]
    with torch.no_grad():
        hin = blk.ln1(stream[L_SHOW])
        W, B = blk.attn.in_proj_weight, blk.attn.in_proj_bias
        qkv = hin[0] @ W.t() + B
        q, k, v = qkv.split(D, dim=-1)
        dh = D // HEADS
        qh = q[:, H_SHOW*dh:(H_SHOW+1)*dh]; kh = k[:, H_SHOW*dh:(H_SHOW+1)*dh]
        scores = (qh[QPOS] @ kh.t()) / (dh ** 0.5)
        w_row = F.softmax(scores, -1).cpu().numpy()
        scores = scores.cpu().numpy()
    fig = plt.figure(figsize=(15.6, 7.4))
    axq = fig.add_axes([0.06, 0.72, 0.88, 0.075])
    qv = qh[QPOS].cpu().numpy()
    axq.imshow(qv.reshape(1, -1), aspect="auto", cmap="RdBu_r",
               vmin=-abs(qv).max(), vmax=abs(qv).max())
    axq.set_yticks([]); axq.set_xticks([])
    fig.text(0.06, 0.815, f"the QUERY built at position {QPOS} "
             f"({dh} numbers) — \"what am I looking for?\"",
             fontsize=14, color=CLAY, fontweight="bold")
    axs = fig.add_axes([0.06, 0.40, 0.88, 0.20])
    axs.plot(scores, lw=1.2, color=MUTED)
    axs.set_ylabel("query · key", fontsize=12); axs.set_xlim(0, T)
    fig.text(0.06, 0.635, "…dotted against every KEY before it — one score per position",
             fontsize=14, color=INK, fontweight="bold")
    axw = fig.add_axes([0.06, 0.10, 0.88, 0.20])
    axw.bar(range(T), w_row, color=CLAY, width=1.0)
    axw.set_xlim(0, T); axw.set_ylabel("attention", fontsize=12)
    axw.set_xlabel("position in the 136-token window")
    for s_i in range(1, CTX):
        axw.axvline(s_i*PS, color=MUTED, lw=0.7, alpha=0.5)
        axs.axvline(s_i*PS, color=MUTED, lw=0.7, alpha=0.5)
    fig.text(0.06, 0.335, "…softmax turns the scores into weights that sum to 1. "
             "The VALUES are then blended in exactly these proportions.",
             fontsize=14, color=INK, fontweight="bold")
    fig.text(0.5, 0.015, f"layer {L_SHOW+1}, head {H_SHOW+1}, real weights — "
             f"the top {int((np.sort(w_row)[::-1][:8]).sum()*100)}% of this head's "
             "attention sits on just 8 of the 136 positions",
             ha="center", fontsize=12.5, color=MUTED, style="italic")
    plt.savefig("/out/figs/fig_x_qkv.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print("fig_x_qkv", flush=True)

    # ═══════ FIG E · eight heads, eight different questions
    fig, axes = plt.subplots(2, 4, figsize=(16.4, 7.0), sharex=True, sharey=True)
    A = attn_l[L_SHOW]
    ymax = float(A[:, QPOS].max()) * 1.18
    for hh, ax in enumerate(axes.flat):
        ax.bar(range(T), A[hh, QPOS], color=PURPLE, width=1.0)
        ax.set_title(f"head {hh+1}", loc="left", fontsize=13, color=PURPLE)
        for s_i in range(1, CTX):
            ax.axvline(s_i*PS, color=MUTED, lw=0.6, alpha=0.45)
        ax.set_ylim(0, ymax)
        ax.tick_params(labelsize=11)
    for ax in axes[1]:
        ax.set_xlabel("position", fontsize=13)
    for ax in (axes[0][0], axes[1][0]):
        ax.set_ylabel("attention", fontsize=13)
    fig.text(0.5, 0.015, f"the same position, layer {L_SHOW+1}, all {HEADS} heads · "
             "each head asks its own question and they are visibly different — "
             "their answers are concatenated and mixed by one more matrix",
             ha="center", fontsize=13, color=INK)
    plt.tight_layout(rect=[0, 0.055, 1, 1])
    plt.savefig("/out/figs/fig_x_heads.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print("fig_x_heads", flush=True)

    # ═══════ FIG F · what each LAYER looks at (measured, not asserted)
    cur_lo = (CTX - 1) * PS                       # start of the newest frame
    inside, action_mass = [], []
    fig, axes = plt.subplots(2, 3, figsize=(16.4, 6.6), sharex=True, sharey=True)
    for l, ax in enumerate(axes.flat):
        row = attn_l[l].mean(0)[QPOS]
        col = [TEAL, GOLD, CLAY, BLUE, PURPLE, "#4A9467"][l]
        ax.bar(range(T), row, color=col, width=1.0)
        frac = float(row[cur_lo:].sum())
        # shade the newest frame and print its share INSIDE the panel, so the
        # number in the caption is something you can see rather than trust
        ax.axvspan(cur_lo, T, color=col, alpha=0.13, zorder=0)
        ax.axvline(cur_lo, color=INK, lw=1.6)
        ax.set_title(f"layer {l+1}", loc="left", fontsize=13)
        ax.text(0.985, 0.86, f"{frac*100:.0f}%\ninside", transform=ax.transAxes,
                ha="right", va="top", fontsize=15, color=col, fontweight="bold",
                linespacing=1.25)
        ax.text(0.015, 0.86, f"{(1-frac)*100:.0f}%\nearlier frames",
                transform=ax.transAxes, ha="left", va="top", fontsize=12,
                color=MUTED, linespacing=1.25)
        for s_i in range(1, CTX):
            ax.axvline(s_i*PS, color=MUTED, lw=0.6, alpha=0.45)
        inside.append(frac)
        action_mass.append(float(row[NTOK::PS].sum()))
    for ax in axes[1]:
        ax.set_xlabel("position in the window")
    fig.text(0.5, 0.075, "attention from the sampling position, averaged over heads · "
             "the SHADED band to the right of the black line is the newest frame\n"
             f"share of attention INSIDE the newest frame, layer 1→{NL}:   "
             + "   ".join(f"{v*100:.0f}%" for v in inside),
             ha="center", va="top", fontsize=16, color=INK, linespacing=1.6)
    plt.tight_layout(rect=[0, 0.155, 1, 1])
    plt.savefig("/out/figs/fig_x_layers.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print("fig_x_layers", flush=True)
    st["attn_inside_current_frame"] = inside
    st["attn_on_action_tokens"] = action_mass

    # ═══════ FIG G · the residual stream — the "context vector" being built
    # logit lens: decode every intermediate vector with the FINAL head.
    with torch.no_grad():
        norms, cos_in, top_id, top_p, ent = [], [], [], [], []
        base = stream[0][0, QPOS]
        for l in range(NL + 1):
            vv = stream[l][0, QPOS]
            norms.append(float(vv.norm()))
            cos_in.append(float(F.cosine_similarity(vv, base, dim=0)))
            lg = gpt.head(gpt.ln(vv))
            p = F.softmax(lg, -1)
            top_id.append(int(p.argmax())); top_p.append(float(p.max()))
            ent.append(float(-(p * (p + 1e-9).log()).sum()))
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 7.2))
    xs = list(range(NL + 1))
    axes[0].plot(xs, norms, "-o", lw=2.8, ms=8, color=TEAL)
    axes[0].set_title("the vector grows", loc="left")
    axes[0].set_xlabel("after layer"); axes[0].set_ylabel("length of the vector")
    axes[1].plot(xs, cos_in, "-o", lw=2.8, ms=8, color=CLAY)
    axes[1].set_title("and stops being the word it started as", loc="left")
    axes[1].set_xlabel("after layer")
    axes[1].set_ylabel("similarity to the input embedding")
    axes[1].set_ylim(-0.15, 1.05); axes[1].axhline(0, color=MUTED, lw=0.9, ls=":")
    axes[2].plot(xs, ent, "-o", lw=2.8, ms=8, color=PURPLE)
    axes[2].set_title("but the prediction does not settle smoothly", loc="left")
    axes[2].set_xlabel("after layer")
    axes[2].set_ylabel("uncertainty about the next word (nats)")
    fig.text(0.5, 0.055, "One position, traced through the stack. By the top of the "
             f"network the vector has almost nothing left in common with the word\n"
             f"that was fed in (cosine {cos_in[0]:.2f} → {cos_in[-1]:.2f}). "
             "It is no longer a word — it is a summary of the whole window.",
             ha="center", va="top", fontsize=17, color=INK, linespacing=1.5)
    plt.tight_layout(rect=[0, 0.19, 1, 1])
    plt.savefig("/out/figs/fig_x_residual.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print("fig_x_residual", flush=True)
    st.update({"stream_norm": norms, "stream_cos_to_input": cos_in,
               "stream_entropy": ent, "stream_top_id": top_id,
               "stream_top_p": top_p})

    # ═══════ FIG H · the logit lens as pictures — what it thinks, layer by layer
    pics = [gallery[i] for i in top_id]
    NC2 = 4
    NR2 = (NL + 1 + NC2 - 1) // NC2
    fig, axes = plt.subplots(NR2, NC2, figsize=(3.95*NC2, 3.42*NR2))
    for l, ax in enumerate(axes.flat):
        if l > NL:
            ax.axis("off"); continue
        ax.imshow(up(pics[l], 12)); ax.axis("off")
        ax.set_title(("input" if l == 0 else f"after layer {l}"), fontsize=16,
                     color=MUTED)
        ax.text(0.5, -0.055, f"word #{top_id[l]} · {top_p[l]*100:.0f}% sure",
                transform=ax.transAxes, ha="center", va="top", fontsize=15,
                color=CLAY if l == NL else INK,
                fontweight="bold" if l == NL else "normal")
    fig.text(0.5, 0.015, "the same vector, decoded with the final output head after "
             "every layer — its best guess, and how sure it is",
             ha="center", fontsize=15, color=INK)
    plt.tight_layout(rect=[0, 0.055, 1, 1], h_pad=4.2)
    plt.savefig("/out/figs/fig_x_logitlens.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print("fig_x_logitlens", flush=True)

    # ═══════ FIG I · the final distribution over the 512 words
    with torch.no_grad():
        p = F.softmax(logits[0, QPOS], -1).cpu().numpy()
    order = np.argsort(-p)[:8]
    pics = [gallery[int(i)] for i in order]
    fig = plt.figure(figsize=(16.4, 7.0))
    axd = fig.add_axes([0.055, 0.545, 0.90, 0.335])
    axd.bar(range(K), p, color=CLAY, width=1.2)
    axd.set_xlim(0, K); axd.set_ylabel("probability")
    axd.set_xlabel("one of the 512 picture words")
    fig.text(0.055, 0.90, "The output is a probability for EVERY word in the "
             "vocabulary — never a blend of them", fontsize=15, color=INK,
             fontweight="bold")
    for i, (cid, pic) in enumerate(zip(order, pics)):
        axi = fig.add_axes([0.055 + i*0.116, 0.12, 0.093, 0.24])
        axi.imshow(up(pic, 12)); axi.axis("off")
        axi.set_title(f"#{int(cid)} · {p[int(cid)]*100:.0f}%", fontsize=12,
                      color=CLAY if i == 0 else MUTED)
    fig.text(0.5, 0.015, "the eight most likely next words, decoded · "
             "we sample one of them, write it into the sequence, and ask again — "
             "sixteen times to finish one imagined frame",
             ha="center", fontsize=12.5, color=INK)
    plt.savefig("/out/figs/fig_x_logits.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print("fig_x_logits", flush=True)
    st["top8"] = [[int(i), float(p[int(i)])] for i in order]

    json.dump(st, open("/out/figs/internals.json", "w"), indent=1)
    vol.commit()
    print("INTERNALS DONE", flush=True)
    return st


# ═════════════════ the class demo: a branching tree of real imagined futures
@app.function(image=gpu_image, gpu="A10G", timeout=7200, volumes={"/out": vol})
def simulator(depth: int = 7, branch: int = 3):
    """Pre-compute a tree of genuine model rollouts for a live classroom demo.

    From one real context, the presenter picks an action; we show what BOTH world
    models imagine. Every frame in the tree is produced by the trained networks —
    nothing is replayed from the game after the context ends.
    """
    import os, json, base64, io
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    ns = {}; exec(SRC, ns)
    dev = torch.device("cuda")
    K, NTOK, GRID, A_DIM = ns["K"], ns["NTOK"], ns["GRID"], ns["A_DIM"]
    os.makedirs("/out/sim", exist_ok=True)

    z = np.load("/out/coinrun.npz"); n = int(z["n"])
    held = [{"frames": z[f"f{i}"], "actions": z[f"a{i}"]} for i in range(n-HOLD_OUT, n)]
    rssm = ns["RSSM"]().to(dev)
    rssm.load_state_dict(torch.load("/out/rssm.pt", map_location=dev)["sd"]); rssm.eval()
    tok = ns["Tokenizer"]().to(dev)
    tok.load_state_dict(torch.load("/out/iris_tok.pt", map_location=dev)["sd"]); tok.eval()
    gpt = ns["IRISGPT"]().to(dev)
    gpt.load_state_dict(torch.load("/out/iris_gpt.pt", map_location=dev)["sd"]); gpt.eval()
    CTX = gpt.ctx_steps

    # CoinRun's 15 actions; these three are the legible ones for a demo
    ACTIONS = [("LEFT", 1), ("RIGHT", 7), ("JUMP", 5)][:branch]

    ep = held[5]; t0 = 90
    ctx_fr = torch.tensor(ep["frames"][t0:t0+CTX]/255.0, dtype=torch.float32).to(dev)
    ctx_ac = torch.tensor(ep["actions"][t0:t0+CTX]).to(dev)

    frames, index = [], {}

    def put(img):
        frames.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
        return len(frames) - 1

    # ---- prime both models on the same real context
    with torch.no_grad():
        a1h = F.one_hot(ctx_ac, A_DIM).float()
        embs = rssm.enc(ctx_fr)
        h = torch.zeros(1, ns["H"], device=dev); s = torch.zeros(1, ns["S"], device=dev)
        for t in range(CTX):
            if t > 0:
                h = rssm.step_h(h, s, a1h[t-1:t])
            qm, _ = rssm.posterior(h, embs[t:t+1]); s = qm
        _, idx_all, _ = tok(ctx_fr)
    root_rssm = (h.clone(), s.clone())
    root_iris = (idx_all.view(-1, NTOK)[:CTX].unsqueeze(0).clone(),
                 ctx_ac[:CTX].unsqueeze(0).clone())
    ctx_ids = [put(f) for f in ctx_fr.cpu().numpy()]

    @torch.no_grad()
    def step_rssm(state, a):
        h, s = state
        a1 = F.one_hot(torch.tensor([a], device=dev), A_DIM).float()
        h2 = rssm.step_h(h, s, a1)
        pm, _ = rssm.prior(h2)
        img = rssm.dec(rssm.feat(h2, pm))[0].clamp(0, 1).cpu().numpy()
        return (h2, pm), img

    @torch.no_grad()
    def step_iris(state, a):
        hist_t, hist_a = state
        hist_a = torch.cat([hist_a[:, :-1],
                            torch.tensor([[a]], device=dev)], 1)  # the chosen action
        ht = hist_t[:, -(CTX-1):]; ha = hist_a[:, -(CTX-1):]
        x = gpt.build(ht, ha)
        nxt = []
        for _ in range(NTOK):
            Tn = x.shape[1]
            m = torch.triu(torch.ones(Tn, Tn, device=dev, dtype=torch.bool), 1)
            hc = x
            for b in gpt.blocks:
                hc, _ = b(hc, m, False)
            p = F.softmax(gpt.head(gpt.ln(hc))[0, -1], -1)
            tk = torch.multinomial(p, 1)
            nxt.append(int(tk))
            x = torch.cat([x, gpt.tok_emb(tk.view(1, 1)) + gpt.pos[:, Tn:Tn+1]], 1)
        nt = torch.tensor(nxt, device=dev).view(1, GRID, GRID)
        img = tok.decode_tokens(nt)[0].clamp(0, 1).cpu().numpy()
        hist_t = torch.cat([hist_t, nt.view(1, 1, NTOK)], 1)
        hist_a = torch.cat([hist_a, torch.tensor([[a]], device=dev)], 1)
        return (hist_t, hist_a), img, nxt

    torch.manual_seed(0)
    total = 0

    def walk(path, s_r, s_i, d):
        nonlocal total
        if d == depth:
            return
        for name, a in ACTIONS:
            key = "/".join(path + [name])
            s_r2, img_r = step_rssm(s_r, a)
            s_i2, img_i, ids = step_iris(s_i, a)
            index[key] = {"rssm": put(img_r), "iris": put(img_i), "tokens": ids}
            total += 1
            if total % 100 == 0:
                print(f"  {total} nodes", flush=True)
            walk(path + [name], s_r2, s_i2, d + 1)

    walk([], root_rssm, root_iris, 0)

    # one sprite sheet keeps the demo to two files and works offline
    COLS = 40
    rows = (len(frames) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS*64, rows*64), (0, 0, 0))
    for i, f in enumerate(frames):
        sheet.paste(Image.fromarray(f), ((i % COLS)*64, (i // COLS)*64))
    sheet.save("/out/sim/sprites.png")
    json.dump({"cols": COLS, "n": len(frames), "context": ctx_ids,
               "actions": [a[0] for a in ACTIONS], "depth": depth,
               "index": index},
              open("/out/sim/tree.json", "w"))
    vol.commit()
    print(f"SIM DONE · {len(frames)} frames · {total} nodes · depth {depth}", flush=True)
    return {"frames": len(frames), "nodes": total}


# ═══════════════════════════ the opener: a results video for slide 1
@app.function(image=gpu_image, gpu="A10G", timeout=7200, volumes={"/out": vol})
def opener(fps: int = 20):
    """Render the 'what we built' video that opens the lecture.

    Real footage, then the same scene imagined by both world models with the
    camera switched off. Every imagined frame comes from the trained networks.
    """
    import os, subprocess
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageDraw, ImageFont

    ns = {}; exec(SRC, ns)
    dev = torch.device("cuda")
    K, NTOK, GRID, A_DIM = ns["K"], ns["NTOK"], ns["GRID"], ns["A_DIM"]
    W, H = 1600, 900
    PAPER, INK, MUTED = (251, 249, 241), (22, 19, 13), (109, 102, 90)
    TEAL, GOLD, CLAY = (46, 143, 143), (221, 159, 62), (201, 100, 66)
    os.makedirs("/out/vid/frames", exist_ok=True)
    for f in os.listdir("/out/vid/frames"):
        os.remove("/out/vid/frames/" + f)

    def font(sz, bold=False):
        for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else
                  "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            if os.path.exists(p):
                return ImageFont.truetype(p, sz)
        return ImageFont.load_default()

    F_TITLE, F_H, F_B, F_S = font(66, True), font(40, True), font(28), font(22)

    z = np.load("/out/coinrun.npz"); n = int(z["n"])
    held = [{"frames": z[f"f{i}"], "actions": z[f"a{i}"]} for i in range(n-HOLD_OUT, n)]
    rssm = ns["RSSM"]().to(dev)
    rssm.load_state_dict(torch.load("/out/rssm.pt", map_location=dev)["sd"]); rssm.eval()
    tok = ns["Tokenizer"]().to(dev)
    tok.load_state_dict(torch.load("/out/iris_tok.pt", map_location=dev)["sd"]); tok.eval()
    gpt = ns["IRISGPT"]().to(dev)
    gpt.load_state_dict(torch.load("/out/iris_gpt.pt", map_location=dev)["sd"]); gpt.eval()
    CTX, HOR = gpt.ctx_steps, 24

    ep = held[7]; t0 = 80
    fr = torch.tensor(ep["frames"][t0:t0+CTX+HOR]/255.0, dtype=torch.float32).to(dev)
    acts = torch.tensor(ep["actions"][t0:t0+CTX+HOR]).to(dev)
    torch.manual_seed(0)
    with torch.no_grad():
        a1h = F.one_hot(acts, A_DIM).float()
        embs = rssm.enc(fr)
        h = torch.zeros(1, ns["H"], device=dev); s = torch.zeros(1, ns["S"], device=dev)
        for t in range(CTX):
            if t > 0:
                h = rssm.step_h(h, s, a1h[t-1:t])
            qm, _ = rssm.posterior(h, embs[t:t+1]); s = qm
        dream_r = []
        for k in range(HOR):
            h = rssm.step_h(h, s, a1h[CTX+k-1:CTX+k])
            pm, _ = rssm.prior(h); s = pm
            dream_r.append(rssm.dec(rssm.feat(h, s))[0].clamp(0,1).cpu().numpy())
        _, idx_all, _ = tok(fr)
        hist_t = idx_all.view(-1, NTOK)[:CTX].unsqueeze(0).clone()
        hist_a = acts[:CTX].unsqueeze(0).clone()
        dream_i = []
        for k in range(HOR):
            ht = hist_t[:, -(CTX-1):]; ha = hist_a[:, -(CTX-1):]
            x = gpt.build(ht, ha)
            nxt = []
            for _ in range(NTOK):
                Tn = x.shape[1]
                m = torch.triu(torch.ones(Tn, Tn, device=dev, dtype=torch.bool), 1)
                hc = x
                for b in gpt.blocks:
                    hc, _ = b(hc, m, False)
                p = F.softmax(gpt.head(gpt.ln(hc))[0, -1], -1)
                tk = torch.multinomial(p, 1); nxt.append(int(tk))
                x = torch.cat([x, gpt.tok_emb(tk.view(1,1)) + gpt.pos[:, Tn:Tn+1]], 1)
            nt = torch.tensor(nxt, device=dev).view(1, GRID, GRID)
            dream_i.append(tok.decode_tokens(nt)[0].clamp(0,1).cpu().numpy())
            hist_t = torch.cat([hist_t, nt.view(1,1,NTOK)], 1)
            hist_a = torch.cat([hist_a, acts[CTX+k:CTX+k+1].view(1,1)], 1)
        codes = tok.vq.emb.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, GRID, GRID)
        patches = torch.cat([tok.decode_q(codes[i:i+64]).clamp(0,1)
                             for i in range(0, K, 64)]).cpu().numpy()
    real = fr.cpu().numpy()

    def tile(a, px):
        return Image.fromarray((np.clip(a,0,1)*255).astype(np.uint8)).resize(
            (px, px), Image.NEAREST)

    idx = [0]
    def emit(im):
        im.save(f"/out/vid/frames/{idx[0]:05d}.png"); idx[0] += 1

    def card():
        im = Image.new("RGB", (W, H), PAPER)
        return im, ImageDraw.Draw(im)

    def dots(d, x, y, r=7):
        for i, c in enumerate((TEAL, GOLD, CLAY)):
            d.ellipse([x+i*(r*2+8), y, x+i*(r*2+8)+r*2, y+r*2], fill=c)

    # ── 1 · title over live footage
    for t in range(int(fps*4.5)):
        im, d = card()
        f = tile(real[t % (CTX+HOR)], 460)
        im.paste(f, (W//2 - 230, 250))
        dots(d, W//2 - 31, 120)
        d.text((W//2, 175), "BUILD A WORLD MODEL FROM SCRATCH · LECTURE 5",
               font=F_S, fill=MUTED, anchor="mm")
        d.text((W//2, 790), "This is CoinRun. A real game, 64×64 pixels.",
               font=F_B, fill=INK, anchor="mm")
        emit(im)

    # ── 2 · the question
    for t in range(int(fps*3.5)):
        im, d = card()
        d.text((W//2, 330), "We trained two world models", font=F_TITLE, fill=INK,
               anchor="mm")
        d.text((W//2, 430), "to imagine it with the screen turned off.",
               font=F_TITLE, fill=CLAY, anchor="mm")
        d.text((W//2, 560), "600 episodes · 256,151 frames · neither model has seen "
               "the level you are about to watch", font=F_B, fill=MUTED, anchor="mm")
        emit(im)

    # ── 3 · context, then the camera goes off
    PX = 300
    def triptych(k, showing_ctx):
        im, d = card()
        xs = [150, 650, 1150]
        labs = [("THE REAL GAME", INK), ("RSSM · a vector", TEAL),
                ("IRIS · a vocabulary", CLAY)]
        if showing_ctx:
            imgs = [real[k]] * 3
        else:
            imgs = [real[CTX + k], dream_r[k], dream_i[k]]
        for x, img, (lab, col) in zip(xs, imgs, labs):
            im.paste(tile(img, PX), (x, 320))
            d.text((x + PX//2, 290), lab, font=F_S, fill=col, anchor="mm")
            d.rectangle([x-2, 318, x+PX+2, 322+PX], outline=(201,195,180))
        if showing_ctx:
            d.text((W//2, 180), f"both models watch {CTX} real frames",
                   font=F_H, fill=INK, anchor="mm")
            d.text((W//2, 700), f"frame {k+1} of {CTX}", font=F_B, fill=MUTED,
                   anchor="mm")
        else:
            d.text((W//2, 180), "now the camera is off — everything on the right "
                   "is imagined", font=F_H, fill=CLAY, anchor="mm")
            d.text((W//2, 700), f"+{k+1} steps into the dream", font=F_B, fill=INK,
                   anchor="mm")
        dots(d, W//2 - 31, 90)
        return im
    for k in range(CTX):
        for _ in range(max(1, fps//5)):
            emit(triptych(k, True))
    for k in range(HOR):
        for _ in range(max(1, fps//5)):
            emit(triptych(k, False))

    # ── 4 · the vocabulary it invented
    order = np.argsort(-tok.vq.cluster.cpu().numpy())
    NRr, NCc, TP = 16, 32, 26
    grid = Image.new("RGB", (NCc*TP, NRr*TP), PAPER)
    for i in range(NRr*NCc):
        grid.paste(tile(patches[order[i]], TP), ((i % NCc)*TP, (i // NCc)*TP))
    for t in range(int(fps*4.5)):
        im, d = card()
        im.paste(grid, (W//2 - grid.width//2, 300))
        d.text((W//2, 200), "and IRIS invented this vocabulary by itself",
               font=F_H, fill=INK, anchor="mm")
        d.text((W//2, 760), "512 visual words, learned from raw frames — "
               "nobody labelled any of them", font=F_B, fill=MUTED, anchor="mm")
        dots(d, W//2 - 31, 110)
        emit(im)

    # ── 5 · the close
    for t in range(int(fps*4.0)):
        im, d = card()
        d.text((W//2, 250), "A Vector, or a Vocabulary?", font=F_TITLE, fill=INK,
               anchor="mm")
        rows = [("256,151", "frames of CoinRun"),
                ("472 / 512", "codebook words actually used"),
                ("6", "transformer layers we will open up"),
                ("live", "you can steer the dream yourself")]
        y = 400
        for big, small in rows:
            d.text((W//2 - 40, y), big, font=F_H, fill=CLAY, anchor="rm")
            d.text((W//2 + 10, y), small, font=F_B, fill=INK, anchor="lm")
            y += 78
        dots(d, W//2 - 31, 140)
        emit(im)

    out = "/out/vid/opener.mp4"
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps),
                    "-i", "/out/vid/frames/%05d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    "-movflags", "+faststart", out], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["ffmpeg", "-y", "-i", out, "-vf",
                    "fps=12,scale=800:-1:flags=lanczos", "/out/vid/opener.gif"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    vol.commit()
    print(f"OPENER DONE · {idx[0]} frames · {idx[0]/fps:.1f}s", flush=True)
    return {"frames": idx[0], "seconds": idx[0]/fps}

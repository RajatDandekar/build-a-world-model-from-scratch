"""Unified SSL pretraining loop. One function, five objectives — everything
else (encoder, data, optimizer, schedule, seed, steps) is held identical.

Logs JSONL every `log_every` steps: step, epoch, loss, feat_std, eff_rank,
lr, ema momentum. Collapse metrics use a FIXED held-out probe batch so curves
are comparable across runs.
"""
import json
import math
import os
import time
import torch
from torch.utils.data import DataLoader

from .models import SSLModel, collapse_metrics
from .data import get_ssl_dataset, get_probe_datasets


def cosine_lr(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * p))


def train_run(arch, dataset="stl10", data_root="/data", out_root="/results",
              epochs=100, batch_size=256, base_lr=5e-4, weight_decay=0.05,
              warmup_epochs=5, ema_start=0.996, seed=0, log_every=50,
              num_workers=12, run_name=None, max_steps=None):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = run_name or f"{dataset}_{arch}_s{seed}"
    out_dir = os.path.join(out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    if dataset == "stl10":
        model_kw = dict(img_size=96, patch_size=8)
    else:
        model_kw = dict(img_size=224, patch_size=16)
    model = SSLModel(arch, **model_kw).to(device)

    ds = get_ssl_dataset(dataset, arch, data_root)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                    num_workers=num_workers, pin_memory=True,
                    persistent_workers=num_workers > 0)

    # fixed eval batch (labeled train split, eval transform) for collapse metrics
    probe_tr, _ = get_probe_datasets(dataset, data_root)
    fixed = torch.stack([probe_tr[i][0] for i in range(256)]).to(device)

    steps_per_epoch = len(dl)
    total_steps = min(max_steps or 10**9, epochs * steps_per_epoch)
    warmup = warmup_epochs * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr,
                            betas=(0.9, 0.95), weight_decay=weight_decay)

    log_path = os.path.join(out_dir, "log.jsonl")
    logf = open(log_path, "a")
    meta = dict(arch=arch, dataset=dataset, epochs=epochs,
                batch_size=batch_size, base_lr=base_lr,
                weight_decay=weight_decay, warmup_epochs=warmup_epochs,
                seed=seed, total_steps=total_steps,
                steps_per_epoch=steps_per_epoch, **model_kw)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    step, t0 = 0, time.time()
    done = False
    for epoch in range(epochs):
        if done:
            break
        for batch in dl:
            if step >= total_steps:
                done = True
                break
            lr = cosine_lr(step, total_steps, base_lr, warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            if isinstance(batch[0], (list, tuple)):  # TwoView collate
                inp = (batch[0][0].to(device, non_blocking=True),
                       batch[0][1].to(device, non_blocking=True))
            else:
                inp = batch[0].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                loss, aux = model(inp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            m = ema_start + (1.0 - ema_start) * step / max(1, total_steps)
            model.ema_update(m)

            if step % log_every == 0:
                model.eval()
                with torch.no_grad(), torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16,
                        enabled=device == "cuda"):
                    pooled = model.encoder(fixed).mean(dim=1)
                feat_std, eff_rank = collapse_metrics(pooled)
                model.train()
                rec = dict(step=step, epoch=epoch, loss=float(loss.item()),
                           feat_std=feat_std, eff_rank=eff_rank, lr=lr,
                           ema=m, wall=time.time() - t0)
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
            step += 1

        if (epoch + 1) % 25 == 0 or epoch == epochs - 1:
            torch.save({"encoder": model.encoder.state_dict(),
                        "arch": arch, "step": step, "epoch": epoch,
                        "meta": meta},
                       os.path.join(out_dir, f"ckpt_ep{epoch + 1:04d}.pt"))

    torch.save({"encoder": model.encoder.state_dict(), "arch": arch,
                "step": step, "epoch": epochs, "meta": meta,
                "full_state": model.state_dict()},
               os.path.join(out_dir, "ckpt_final.pt"))
    logf.close()
    return out_dir

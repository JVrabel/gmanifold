"""Decoder capacity / hyper-parameter study: does a bigger or differently trained autoencoder capture the manifold
better, and does it change the sampler? Scores every setting by held-out reconstruction (early-stopped on the
validation split) and by sampling quality at alpha = 0.3 (u, coverage, ESS).
python experiments/capacity.py --out results/capacity"""
import argparse
import itertools
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gmanifold as gm
from gmanifold import transformer as tr

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/capacity"); ap.add_argument("--model", default="SimpleStories/SimpleStories-5M")
ap.add_argument("--locs", nargs="+", default=["embed", "L2.ffn", "L5.ffn"]); ap.add_argument("--quick", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); T0 = time.time()
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).cuda().eval()
tok = AutoTokenizer.from_pretrained(a.model)
X_emb, ids = tr.vocab_states(model, tok); keep = ~gm.degenerate_mask(X_emb); X_emb, ids = X_emb[keep], [i for i, k in zip(ids, keep.tolist()) if k]
prefix = tok("Once upon a time, there was a little", add_special_tokens=False)["input_ids"]
seqs = torch.tensor([prefix + [v] for v in ids]); pos = torch.full((len(ids),), len(prefix))
states = tr.collect_states(model, seqs, pos, [l for l in a.locs if l != "embed"]); states["embed"] = X_emb.cpu()
perm = torch.randperm(len(ids), generator=torch.Generator().manual_seed(0)); n_val = len(ids) // 10; tr_i, va_i = perm[:-n_val], perm[-n_val:]

grid = []
for hidden in [(256, 128), (512, 256), (1024, 512), (2048, 1024), (512,), (512, 512, 256), (1024, 1024, 512)]:
    grid.append(dict(hidden=hidden))
for lr in (1e-3, 5e-3):
    grid.append(dict(lr=lr))
for bs in (128, 2048):
    grid.append(dict(batch_size=bs))
for lam in (0.03, 0.3):
    grid.append(dict(lam_geom=lam))
grid.append(dict(epochs=100)); grid.append(dict(epochs=600))
ms = [8, 16, 24, 32, 48, 64]
if a.quick:
    grid, ms = grid[:2], [16]
rows = []
for loc in a.locs:
    X = states[loc].cuda(); Xtr, Xva = X[tr_i], X[va_i]; kernel = gm.KernelScore(Xtr)
    for m in ms:
        for cfg in ([dict()] if m != 16 else [dict()] + grid):
            kw = dict(cfg); hidden = kw.pop("hidden", (512, 256)); epochs = kw.pop("epochs", 300)
            t = time.time()
            M = gm.GlobalManifold(latent_dim=m, hidden=hidden).fit(Xtr, epochs=epochs, X_val=Xva, seed=0, log_every=25, **kw)
            curve = [(h["epoch"], h["val_recon_over_spacing"]) for h in M.history if "val_recon_over_spacing" in h]
            best_ep, best_val = min(curve, key=lambda e: e[1])
            x, info = M.sample(2000, alpha=0.3, auto_tau=True, seed=0)
            rep = gm.support_report(M, x, kernel=kernel)["samples"]
            rows.append(dict(loc=loc, m=m, hidden=list(hidden), epochs=epochs, **{k: v for k, v in kw.items()}, params=sum(p.numel() for p in M.parameters()),
                             val_recon_final=curve[-1][1], val_recon_best=best_val, best_epoch=best_ep, train_recon=M.history[-1]["recon"],
                             u=rep["u"], nearest=rep["nearest_real"], recon_gap=rep["recon"], coverage=M.coverage(x), ess=info["ess"], tau=info["tau"], seconds=time.time() - t))
            print(f"[{time.time() - T0:5.0f}s] {loc} m={m} {cfg}: val {curve[-1][1]:.3f} (best {best_val:.3f}@{best_ep}) u {rep['u']:.3f} cov {rows[-1]['coverage']:.2f} ess {info['ess']:.0f}", flush=True)
    torch.cuda.empty_cache()
json.dump(rows, open(os.path.join(a.out, "capacity.json"), "w"), indent=1); print("saved", f"({(time.time() - T0) / 60:.1f} min)")

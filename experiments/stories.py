"""Real-occurrence clouds: residual states of random token occurrences in real SimpleStories stories (split by story),
fitted and sampled exactly like the prefix clouds; the FFN sub-layer map is used for propagation (tokenwise, no context).
python experiments/stories.py --model SimpleStories/SimpleStories-5M --out results/5M_stories --n 20000"""
import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import gmanifold as gm
from gmanifold import transformer as tr

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="SimpleStories/SimpleStories-5M"); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=20000); ap.add_argument("--per-story", type=int, default=8); ap.add_argument("--max-len", type=int, default=256)
ap.add_argument("--m", type=int, default=16); ap.add_argument("--epochs", type=int, default=150); ap.add_argument("--locs", nargs="*")
ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5])
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); T0 = time.time()

model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).cuda().eval()
for p in model.parameters():
    p.requires_grad_(False)
tok = AutoTokenizer.from_pretrained(a.model)
g = torch.Generator().manual_seed(0)
seqs, story, positions = [], [], []
for ex in load_dataset("SimpleStories/SimpleStories", split="train", streaming=True):
    ids = tok(ex["story"], add_special_tokens=False)["input_ids"][:a.max_len]
    if len(ids) < 2:
        continue
    sid = len(set(story))
    for p in torch.randperm(len(ids), generator=g)[:a.per_story].tolist():
        seqs.append(ids); story.append(sid); positions.append(p)
    if len(positions) >= a.n:
        break
story = torch.tensor(story)
L = max(len(s) for s in seqs)
input_ids = torch.full((len(seqs), L), tok.eos_token_id); mask = torch.zeros((len(seqs), L), dtype=torch.long)
for r, s in enumerate(seqs):
    input_ids[r, :len(s)] = torch.tensor(s); mask[r, :len(s)] = 1
positions = torch.tensor(positions)
locs = a.locs or [l for l in tr.locations(model) if l.endswith("attn") or l.endswith("ffn")]
states = tr.collect_states(model, input_ids, positions, locs, attention_mask=mask, batch_size=64)
n_story = int(story.max()) + 1
val_story = torch.randperm(n_story, generator=torch.Generator().manual_seed(1))[: n_story // 10]
is_val = torch.isin(story, val_story)
out = dict(model=a.model, n=len(positions), n_stories=n_story, n_val=int(is_val.sum()), locations=locs, dim=[], fits=[], alpha=[], propagation=[])
fits = {}
for loc in locs:
    X = states[loc]
    Xtr, Xva = X[~is_val].cuda(), X[is_val].cuda()
    keep = ~gm.degenerate_mask(Xtr) & ~gm.outlier_mask(Xtr); Xtr = Xtr[keep]
    d = gm.intrinsic_dimension(Xtr); out["dim"].append(dict(loc=loc, n_train=len(Xtr), n_dropped=int((~keep).sum()), **d))
    M = gm.GlobalManifold(latent_dim=a.m).fit(Xtr, epochs=a.epochs, X_val=Xva, seed=0); kernel = gm.KernelScore(Xtr)
    fits[loc] = (M, kernel, Xva)
    out["fits"].append(dict(loc=loc, val_recon=M.history[-1]["val_recon_over_spacing"], train_recon=M.history[-1]["recon"], rank=M.jacobian_rank()["rank_median"]))
    bands = {k: v for k, v in gm.support_report(M, Xva, Xva, kernel).items() if k != "samples"}
    for alpha in a.alphas:
        x, info = M.sample(2000, alpha=alpha, seed=0)
        out["alpha"].append(dict(loc=loc, alpha=alpha, ess=info["ess"], coverage=M.coverage(x), **gm.support_report(M, x, kernel=kernel)["samples"], bands=bands))
    print(f"[{time.time() - T0:5.0f}s] {loc}: n={len(Xtr)} twonn {d['twonn']:.1f} val recon {out['fits'][-1]['val_recon']:.3f} u@0.3 {next(r['u'] for r in out['alpha'] if r['loc'] == loc and r['alpha'] == 0.3):.3f}", flush=True)
for k in range(model.config.num_hidden_layers):                                   # FFN maps Lk.attn -> Lk.ffn
    src, dst = f"L{k}.attn", f"L{k}.ffn"
    if src in fits and dst in fits:
        M_in, k_in, Xva_in = fits[src]; M_out, k_out, Xva_out = fits[dst]
        f = tr.make_map(model, src, dst, torch.zeros(1, dtype=torch.long))
        rows = {f"sample α={al}": gm.support_report(M_out, f(M_in.sample(2000, alpha=al, seed=0)[0]), kernel=k_out)["samples"] for al in (0.3, 0.7, 1.0)}
        rows["real held-out images"] = gm.support_report(M_out, f(Xva_in), kernel=k_out)["samples"]
        j = gm.knn(M_in.X, 1, Xva_in)[1][:, 0]; nz = torch.randn(Xva_in.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(0))
        rows["real + 1x noise images"] = gm.support_report(M_out, f(Xva_in + nz / nz.norm(dim=1, keepdim=True) * M_in.unit(j)[:, None]), kernel=k_out)["samples"]
        out["propagation"].append(dict(src=src, dst=dst, rows=rows, bands={k_: v for k_, v in gm.support_report(M_out, Xva_out, Xva_out, k_out).items() if k_ != "samples"}))
out["seconds"] = time.time() - T0
json.dump(out, open(os.path.join(a.out, "stories.json"), "w"), indent=1); print("saved", f"({out['seconds'] / 60:.1f} min)")

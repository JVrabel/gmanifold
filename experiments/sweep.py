"""Sampling-centric experiments for the results document.
python experiments/sweep.py --model SimpleStories/SimpleStories-5M --out results/5M --m 16 --seeds 0 1 2
For each residual location of the prefix + all-tokens cloud: intrinsic dimension, GlobalManifold fit (val recon,
Jacobian rank), alpha sweep of the samples (nearest-real, recon, kernel u, coverage, ESS) next to the held-out and
noise bands, ablation of the sampler variants and loss weights (at 3 locations), and propagation of embedding
samples through the real model to several destinations with independent destination fits."""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gmanifold as gm
from gmanifold import transformer as tr

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="SimpleStories/SimpleStories-5M"); ap.add_argument("--out", required=True)
ap.add_argument("--m", type=int, nargs="+", default=[16]); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5])
ap.add_argument("--n", type=int, default=2000); ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--prefix", default="Once upon a time, there was a little"); ap.add_argument("--locs", nargs="*")
ap.add_argument("--ablation", action="store_true"); ap.add_argument("--no-propagation", action="store_true")
ap.add_argument("--max-tokens", type=int, default=None, help="use only the first N real tokens (large vocabularies)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T0 = time.time()

model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).cuda().eval()
for p in model.parameters():
    p.requires_grad_(False)
tok = AutoTokenizer.from_pretrained(a.model)
X_emb, ids = tr.vocab_states(model, tok)
if a.max_tokens:
    X_emb, ids = X_emb[:a.max_tokens], ids[:a.max_tokens]
keep = ~gm.degenerate_mask(X_emb)                # drop near-duplicate embedding rows (untrained tokens); paired across locations
n_dropped = int((~keep).sum()); X_emb, ids = X_emb[keep], [i for i, k in zip(ids, keep.tolist()) if k]
print(f"dropped {n_dropped} degenerate tokens; {len(ids)} remain", flush=True)
prefix = tok(a.prefix, add_special_tokens=False)["input_ids"]
if tok.bos_token_id is not None:                 # Llama tokenizers expect a BOS (SimpleStories has none)
    prefix = [tok.bos_token_id] + prefix
seqs = torch.tensor([prefix + [v] for v in ids]); pos = torch.full((len(ids),), len(prefix))
locs = a.locs or tr.locations(model)
states = tr.collect_states(model, seqs, pos, locs)
states["embed"] = X_emb.cpu() if "embed" in locs else states.get("embed")
bad = torch.zeros(len(ids), dtype=torch.bool)                 # isolated outlier states at any location: drop the token everywhere
for l in locs:
    bad |= gm.outlier_mask(states[l].cuda()).cpu()
n_outliers = int(bad.sum())
if n_outliers:
    states = {l: v[~bad] for l, v in states.items()}; X_emb = X_emb[~bad.cuda()]; ids = [i for i, b in zip(ids, bad.tolist()) if not b]
    seqs, pos = seqs[~bad], pos[~bad]
print(f"dropped {n_outliers} outlier tokens (isolated states at some location); {len(ids)} remain", flush=True)
perm = torch.randperm(len(ids), generator=torch.Generator().manual_seed(0))
n_val = len(ids) // 10; tr_i, va_i = perm[:-n_val], perm[-n_val:]
out = dict(model=a.model, D=X_emb.shape[1], n_states=len(ids), n_dropped=n_dropped, n_outliers=n_outliers, n_train=len(tr_i), n_val=n_val, prefix=a.prefix, locations=locs,
           dim=[], fits=[], alpha=[], ablation=[], propagation=[])
fits = {}


def medians(M, Y, kernel, Xh=None):
    r = gm.support_report(M, Y, Xh, kernel)
    return {k: v for k, v in r.items()}


for loc in locs:
    X = states[loc].cuda(); Xtr, Xva = X[tr_i], X[va_i]
    d = gm.intrinsic_dimension(Xtr); out["dim"].append(dict(loc=loc, **d))
    kernel = gm.KernelScore(Xtr)
    for m in a.m:
        T = gm.TangentCharts(Xtr, m)
        for seed in a.seeds:
            t = time.time()
            M = gm.GlobalManifold(latent_dim=m).fit(Xtr, epochs=a.epochs, X_val=Xva, seed=seed)
            fits[(loc, m, seed)] = (M, kernel)
            jr = M.jacobian_rank()
            out["fits"].append(dict(loc=loc, m=m, seed=seed, spacing_unit=M.spacing_unit, hub_share=M.hub_share, val_recon=M.history[-1]["val_recon_over_spacing"], train_recon=M.history[-1]["recon"],
                                    rank=jr["rank_median"], jac_sv=jr["singular_values"][:m].tolist(), seconds=time.time() - t))
            bands = gm.support_report(M, Xva, Xva, kernel, T)            # bands: held-out real, 0.5x, 1x noise
            for alpha in a.alphas:
                x, info = M.sample(a.n, alpha=alpha, auto_tau=True, seed=seed)
                rep = gm.support_report(M, x, kernel=kernel, tangent=T)["samples"]
                out["alpha"].append(dict(loc=loc, m=m, seed=seed, alpha=alpha, ess=info["ess"], tau=info["tau"], anchor_participation=info["anchor_participation"], multiplicity=float(info["multiplicity"].float().mean()),
                                         coverage=M.coverage(x), **rep, bands={k: v for k, v in bands.items() if k != "samples"}))
            print(f"[{time.time() - T0:6.0f}s] {loc} m={m} seed={seed}: val recon {out['fits'][-1]['val_recon']:.3f}, "
                  f"alpha 0.3 -> u {next(r['u'] for r in out['alpha'] if r['loc'] == loc and r['m'] == m and r['seed'] == seed and r['alpha'] == 0.3):.3f}", flush=True)
        del T
    torch.cuda.empty_cache()

if a.ablation:                                                          # sampler variants and loss weights at 3 locations
    for loc in [locs[0], locs[len(locs) // 2], locs[-1]]:
        X = states[loc].cuda(); Xtr, Xva = X[tr_i], X[va_i]; kernel = gm.KernelScore(Xtr)
        M, _ = fits[(loc, a.m[0], a.seeds[0])]; T = gm.TangentCharts(Xtr, a.m[0])
        xt, _ = T.sample(a.n, alpha=0.3, auto_tau=True, seed=0)
        out["ablation"].append(dict(loc=loc, kind="sampler", variant="tangent charts (cross-check)", ess=float(a.n), anchors=None, coverage=M.coverage(xt),
                                    **gm.support_report(M, xt, kernel=kernel, tangent=T)["samples"]))
        for radius, p_, rw in (("global", None, True), ("global", 0, False), ("local", None, True), ("local", 0, True), ("local", 0, False)):
            x, info = M.sample(a.n, alpha=0.3, radius=radius, anchor_power=p_, reweight=rw, auto_tau=True, seed=0)
            out["ablation"].append(dict(loc=loc, kind="sampler", variant=f"radius={radius}, p={'m' if p_ is None else p_}, reweight={rw}", ess=info["ess"], tau=info["tau"], anchors=info["anchor_participation"], coverage=M.coverage(x),
                                        **gm.support_report(M, x, kernel=kernel)["samples"]))
        for name, kw in (("default", {}), ("no_geom", dict(lam_geom=0.0)), ("no_curv", dict(lam_curv=0.0)), ("strong_geom", dict(lam_geom=1.0)),
                         ("wide", dict()), ("epochs_900", dict(epochs=900))):
            hidden = (1024, 512) if name == "wide" else (512, 256)
            Mv = gm.GlobalManifold(latent_dim=a.m[0], hidden=hidden).fit(Xtr, X_val=Xva, seed=0, **{"epochs": a.epochs, **kw})
            x, info = Mv.sample(a.n, alpha=0.3, auto_tau=True, seed=0)
            out["ablation"].append(dict(loc=loc, kind="loss", variant=name, val_recon=Mv.history[-1]["val_recon_over_spacing"], train_recon=Mv.history[-1]["recon"],
                                        ess=info["ess"], coverage=Mv.coverage(x), **gm.support_report(Mv, x, kernel=kernel)["samples"]))
        print(f"[{time.time() - T0:6.0f}s] ablation {loc} done", flush=True)

if not a.no_propagation and locs[0] == "embed":
    dsts = list(dict.fromkeys(locs[i] for i in (1, 2, len(locs) // 2, len(locs) - 1) if 0 < i < len(locs)))
    M_in, k_in = fits[("embed", a.m[0], a.seeds[0])]
    samples = {alpha: M_in.sample(a.n, alpha=alpha, auto_tau=True, seed=0)[0] for alpha in (0.25, 0.5, 1.0)}
    Xva_in = states["embed"][va_i].cuda()
    for dst in dsts:
        f = tr.make_map(model, "embed", dst, seqs[0], pos=len(prefix))
        M_out, k_out = fits[(dst, a.m[0], a.seeds[0])]; T_out = gm.TangentCharts(states[dst][tr_i].cuda(), a.m[0])
        Xva_out = states[dst][va_i].cuda()
        bands = {k: v for k, v in gm.support_report(M_out, Xva_out, Xva_out, k_out, T_out).items() if k != "samples"}
        rows = {f"sample α={alpha}": gm.support_report(M_out, f(x), kernel=k_out, tangent=T_out)["samples"] for alpha, x in samples.items()}
        rows["real held-out images"] = gm.support_report(M_out, f(Xva_in), kernel=k_out, tangent=T_out)["samples"]
        g = torch.Generator(device="cuda").manual_seed(0)
        j = gm.knn(M_in.X, 1, Xva_in)[1][:, 0]
        for fac in (0.5, 1.0):
            nz = torch.randn(Xva_in.shape, device="cuda", generator=g)
            rows[f"real + {fac:g}x noise images"] = gm.support_report(M_out, f(Xva_in + nz / nz.norm(dim=1, keepdim=True) * (fac * M_in.unit(j))[:, None]), kernel=k_out, tangent=T_out)["samples"]
        out["propagation"].append(dict(src="embed", dst=dst, rows=rows, bands=bands))
        print(f"[{time.time() - T0:6.0f}s] propagation embed -> {dst}: " + ", ".join(f"{k}: u={v['u']:.3f}" for k, v in rows.items()), flush=True)

out["seconds"] = time.time() - T0
json.dump(out, open(os.path.join(a.out, "sweep.json"), "w"), indent=1)
print("saved", os.path.join(a.out, "sweep.json"), f"({out['seconds'] / 60:.1f} min)")

"""How do samples look to the model? (1) next-token distributions of samples vs their nearest real tokens vs random
tokens; (2) layer-by-layer trajectory of the samples' images against every location's own manifold and bands, with a
'betweenness' measure relative to the images of the two nearest real anchors; (3) PCA pictures at a few layers.
python experiments/downstream.py --model SimpleStories/SimpleStories-5M --out results/5M_downstream"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gmanifold as gm
from gmanifold import transformer as tr

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="SimpleStories/SimpleStories-5M"); ap.add_argument("--out", default="results/5M_downstream")
ap.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.7, 1.5]); ap.add_argument("--n", type=int, default=2000); ap.add_argument("--epochs", type=int, default=300)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).cuda().eval(); tok = AutoTokenizer.from_pretrained(a.model)
X_emb, ids = tr.vocab_states(model, tok); keep = ~gm.degenerate_mask(X_emb); X_emb, ids = X_emb[keep], [i for i, k in zip(ids, keep.tolist()) if k]
prefix = tok("Once upon a time, there was a little", add_special_tokens=False)["input_ids"]
if tok.bos_token_id is not None:
    prefix = [tok.bos_token_id] + prefix
seqs = torch.tensor([prefix + [v] for v in ids]); pos = torch.full((len(ids),), len(prefix)); L = len(prefix)
locs = tr.locations(model)
states = tr.collect_states(model, seqs, pos, locs[1:]); states["embed"] = X_emb.cpu()
perm = torch.randperm(len(ids), generator=torch.Generator().manual_seed(0)); n_val = len(ids) // 10; tr_i, va_i = perm[:-n_val], perm[-n_val:]
M_in = gm.GlobalManifold(latent_dim=16).fit(X_emb[tr_i], epochs=a.epochs, X_val=X_emb[va_i], seed=0)
Xtr_emb = X_emb[tr_i]


@torch.no_grad()
def logits_at_last(x):
    """Next-token log-probabilities when x replaces the residual at the last position of the prefix context."""
    ctx = seqs[0].cuda(); out = []
    for xb in x.split(1024):
        with tr._Hooks(model, [], {"embed": (xb, torch.full((len(xb),), L, device="cuda"))}):
            lp = model(input_ids=ctx[None].expand(len(xb), -1)).logits[:, L].log_softmax(-1)
        out.append(lp)
    return torch.cat(out)


def js(p, q):
    m = 0.5 * (p.exp() + q.exp())
    return 0.5 * ((p.exp() * (p - m.log())).sum(-1) + (q.exp() * (q - m.log())).sum(-1))


lp_real = logits_at_last(Xtr_emb)                                     # next-token distributions of the real training tokens
ent_real = -(lp_real.exp() * lp_real).sum(-1)
out = dict(model=a.model, next_token=[], trajectory=[], locations=locs)
# (1) next-token behaviour
d2, j2 = gm.knn(Xtr_emb, 2, X_emb[va_i])
lp_val = logits_at_last(X_emb[va_i])
rnd = torch.randint(len(Xtr_emb), (len(va_i),), generator=torch.Generator().manual_seed(1)).cuda()
out["next_token"].append(dict(set="held-out real tokens", js_nearest=float(js(lp_val, lp_real[j2[:, 0]]).median()), js_second=float(js(lp_val, lp_real[j2[:, 1]]).median()),
                              js_random=float(js(lp_val, lp_real[rnd]).median()), entropy=float((-(lp_val.exp() * lp_val).sum(-1)).median()), entropy_real=float(ent_real.median()),
                              top1_agree_nearest=float((lp_val.argmax(-1) == lp_real[j2[:, 0]].argmax(-1)).float().mean()), top1_agree_random=float((lp_val.argmax(-1) == lp_real[rnd].argmax(-1)).float().mean())))
samples = {}
for alpha in a.alphas:
    x, info = M_in.sample(a.n, alpha=alpha, seed=0); samples[alpha] = x
    d2, j2 = gm.knn(Xtr_emb, 2, x); lp = logits_at_last(x)
    rnd = torch.randint(len(Xtr_emb), (len(x),), generator=torch.Generator().manual_seed(1)).cuda()
    out["next_token"].append(dict(set=f"samples α={alpha}", js_nearest=float(js(lp, lp_real[j2[:, 0]]).median()), js_second=float(js(lp, lp_real[j2[:, 1]]).median()),
                                  js_random=float(js(lp, lp_real[rnd]).median()), entropy=float((-(lp.exp() * lp).sum(-1)).median()), entropy_real=float(ent_real.median()),
                                  top1_agree_nearest=float((lp.argmax(-1) == lp_real[j2[:, 0]].argmax(-1)).float().mean()), top1_agree_random=float((lp.argmax(-1) == lp_real[rnd].argmax(-1)).float().mean())))
    # a few decoded examples: sample between which tokens, predicting what
    ex = []
    for i in range(6):
        t0, t1 = tok.convert_ids_to_tokens(ids[int(tr_i[j2[i, 0]])]), tok.convert_ids_to_tokens(ids[int(tr_i[j2[i, 1]])])
        top = [tok.convert_ids_to_tokens(int(k)) for k in lp[i].topk(3).indices]
        top0 = [tok.convert_ids_to_tokens(int(k)) for k in lp_real[j2[i, 0]].topk(3).indices]
        ex.append(dict(nearest=t0, second=t1, sample_top3=top, nearest_top3=top0))
    out["next_token"][-1]["examples"] = ex
nz = torch.randn(len(va_i), X_emb.shape[1], device="cuda", generator=torch.Generator(device="cuda").manual_seed(0))
j1 = gm.knn(Xtr_emb, 1, X_emb[va_i])[1][:, 0]
noise1 = X_emb[va_i] + nz / nz.norm(dim=1, keepdim=True) * M_in.spacing[j1][:, None]
lp = logits_at_last(noise1); d2, j2 = gm.knn(Xtr_emb, 2, noise1)
out["next_token"].append(dict(set="real + 1x noise", js_nearest=float(js(lp, lp_real[j2[:, 0]]).median()), js_second=float(js(lp, lp_real[j2[:, 1]]).median()), js_random=float(js(lp, lp_real[rnd[:len(va_i)]]).median()),
                              entropy=float((-(lp.exp() * lp).sum(-1)).median()), entropy_real=float(ent_real.median()), top1_agree_nearest=float((lp.argmax(-1) == lp_real[j2[:, 0]].argmax(-1)).float().mean()), top1_agree_random=float((lp.argmax(-1) == lp_real[rnd[:len(va_i)]].argmax(-1)).float().mean())))
print("next-token:", *[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "examples"} for r in out["next_token"]], sep="\n  ", flush=True)

# (2) layer-by-layer trajectory of the samples' images
sets = {f"samples α={al}": x for al, x in samples.items()}; sets["held-out real"] = X_emb[va_i]; sets["real + 1x noise"] = noise1
anchors = {name: gm.knn(Xtr_emb, 2, x)[1] for name, x in sets.items()}          # two nearest real anchors at the embedding
f = tr.make_map(model, "embed", locs[1:], seqs[0], pos=L) if False else None
images = {}
for name, x in sets.items():
    imgs = {}
    for xb_start in range(0, len(x), 1024):
        xb = x[xb_start:xb_start + 1024]
        with tr._Hooks(model, locs[1:], {"embed": (xb, torch.full((len(xb),), L, device="cuda"))}) as hk:
            model(input_ids=seqs[0].cuda()[None].expand(len(xb), -1))
        for l in locs[1:]:
            imgs.setdefault(l, []).append(hk.states[l][torch.arange(len(xb), device="cuda"), L])
    images[name] = {l: torch.cat(v) for l, v in imgs.items()}; images[name]["embed"] = x
fig_locs = [locs[1], locs[len(locs) // 3], locs[2 * len(locs) // 3], locs[-1]]
fig, ax = plt.subplots(1, len(fig_locs), figsize=(4 * len(fig_locs), 3.6))
for loc in locs:
    X = states[loc].cuda(); Xtr, Xva = X[tr_i], X[va_i]
    M = M_in if loc == "embed" else gm.GlobalManifold(latent_dim=16).fit(Xtr, epochs=a.epochs, X_val=Xva, seed=0)
    kernel, T = gm.KernelScore(Xtr), gm.TangentCharts(Xtr, 16)
    bands = {k: v for k, v in gm.support_report(M, Xva, Xva, kernel, T).items() if k != "samples"}
    rows = {}
    for name, x in sets.items():
        y = images[name][loc]; rep = gm.support_report(M, y, kernel=kernel, tangent=T)["samples"]
        A, B = Xtr[anchors[name][:, 0]], Xtr[anchors[name][:, 1]]                  # images of the two nearest anchors at this location
        ab = B - A; t = ((y - A) * ab).sum(1) / (ab * ab).sum(1).clamp_min(1e-12)
        perp = (y - A - t[:, None] * ab).norm(dim=1) / M.spacing[gm.knn(Xtr, 1, y)[1][:, 0]]
        rows[name] = dict(**rep, between_t_median=float(t.median()), frac_t_in_01=float(((t > 0) & (t < 1)).float().mean()), perp_over_spacing=float(perp.median()), coverage=M.coverage(y))
    out["trajectory"].append(dict(loc=loc, rows=rows, bands=bands))
    print(f"{loc}: " + " | ".join(f"{n}: u {r['u']:.2f} near {r['nearest_real']:.2f} tan {r['tangent']:.2f} t∈(0,1) {r['frac_t_in_01']:.2f} perp {r['perp_over_spacing']:.2f}" for n, r in rows.items()), flush=True)
    if loc in fig_locs:
        k = fig_locs.index(loc); mu = Xtr.mean(0); W = torch.linalg.svd(Xtr - mu, full_matrices=False)[2][:2].T
        P = lambda Z: ((Z - mu) @ W).cpu()
        for name, col, sz in (("held-out real", "#8a8985", 6), (f"samples α={a.alphas[0]}", "#2a78d6", 5), (f"samples α={a.alphas[-1]}", "#eb6834", 5)):
            p = P(images[name][loc]); ax[k].scatter(p[:, 0], p[:, 1], s=sz, c=col, alpha=0.5, lw=0, label=name)
        ax[k].set_title(loc); ax[k].legend(fontsize=6); ax[k].set_xlabel("PC1"); ax[k].set_ylabel("PC2")
    del M, kernel, T; torch.cuda.empty_cache()
fig.tight_layout(); fig.savefig(os.path.join(a.out, "layers.png"), dpi=110); plt.close(fig)
json.dump(out, open(os.path.join(a.out, "downstream.json"), "w"), indent=1); print("saved")

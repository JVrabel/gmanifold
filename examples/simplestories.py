"""Embedding manifold of SimpleStories-5M -> volume-uniform samples -> the real block-0 map -> validation against
an independently fitted destination manifold.  python examples/simplestories.py [--alpha 0.3] [--n 2000]"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gmanifold as gm
from gmanifold import transformer as tr

ap = argparse.ArgumentParser(); ap.add_argument("--alpha", type=float, default=0.3); ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--dst", default="L0.ffn"); ap.add_argument("--latent-dim", type=int, default=16); ap.add_argument("--epochs", type=int, default=300)
a = ap.parse_args()

name = "SimpleStories/SimpleStories-5M"
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).cuda().eval()
tok = AutoTokenizer.from_pretrained(name)

# 1. real states: X_in = vocabulary embeddings; X_out = the same tokens' residual at `dst` behind a fixed prefix
X_in, ids = tr.vocab_states(model, tok)
prefix = tok("Once upon a time, there was a little", add_special_tokens=False)["input_ids"]
seqs = torch.tensor([prefix + [v] for v in ids]); pos = torch.full((len(ids),), len(prefix))
X_out = tr.collect_states(model, seqs, pos, [a.dst])[a.dst].cuda()          # collected on CPU
perm = torch.randperm(len(ids), generator=torch.Generator().manual_seed(0)); tr_i, va_i = perm[:-400], perm[-400:]

# 2. two independent manifolds (source and destination); nothing about the model enters here
M_in = gm.GlobalManifold(latent_dim=a.latent_dim).fit(X_in[tr_i], epochs=a.epochs, X_val=X_in[va_i])
M_out = gm.GlobalManifold(latent_dim=a.latent_dim).fit(X_out[tr_i], epochs=a.epochs, X_val=X_out[va_i])

# 3. sample the source manifold, push through the real Transformer stretch embed -> dst, validate at both ends
X_sample, info = M_in.sample(a.n, alpha=a.alpha, seed=0)
map_fn = tr.make_map(model, "embed", a.dst, seqs[0], pos=len(prefix))
Y_sample = map_fn(X_sample)

def show(title, rep):
    print(f"\n{title}\n" + "\n".join(f"  {k:20s} " + "  ".join(f"{kk}={vv:.3f}" for kk, vv in v.items()) for k, v in rep.items()))
T_in, T_out = gm.TangentCharts(X_in[tr_i], a.latent_dim), gm.TangentCharts(X_out[tr_i], a.latent_dim)
show(f"source (embed), alpha={a.alpha}, ESS={info['ess']:.0f}, coverage={M_in.coverage(X_sample):.2f}",
     gm.support_report(M_in, X_sample, X_in[va_i], gm.KernelScore(X_in[tr_i]), T_in))
X_tan, _ = T_in.sample(a.n, alpha=a.alpha, seed=0)                               # conservative cross-check sampler
show("source (embed) — tangent-chart samples (cross-check)", {"tangent samples": gm.support_report(M_in, X_tan, kernel=gm.KernelScore(X_in[tr_i]), tangent=T_in)["samples"]})
show(f"destination ({a.dst}) — images of the samples vs the independent destination fit",
     gm.support_report(M_out, Y_sample, X_out[va_i], gm.KernelScore(X_out[tr_i]), T_out))
show(f"destination ({a.dst}) — images of held-out REAL states (control)",
     {"real images": gm.support_report(M_out, map_fn(X_in[va_i]), kernel=gm.KernelScore(X_out[tr_i]))["samples"]})

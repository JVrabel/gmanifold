"""What would a VAE change? Same encoder/decoder, but the encoder outputs (mu, log sigma^2), codes are sampled during
training and beta * KL(q(z|x) || N(0, I)) is added. Compared on: held-out reconstruction (through mu), active latent
units, samples decoded from the PRIOR N(0,I) (the VAE way), and OUR volume-uniform sampler run on the posterior means."""
import json
import math
import sys

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import gmanifold as gm
from gmanifold import transformer as tr
from gmanifold.geometry import knn
from gmanifold.manifold import _mlp


class VAE(gm.GlobalManifold):
    def __init__(self, latent_dim, beta, **kw):
        super().__init__(latent_dim, **kw); self.beta = beta

    def encode(self, x):                                                  # posterior mean
        return self.enc((x.to(self.device) - self.center) / self.scale)[:, :self.m]

    def fit(self, X, epochs=300, batch_size=512, lr=2e-3, lam_geom=0.1, seed=0, X_val=None):
        torch.manual_seed(seed)
        X = X.to(self.device).float(); D, N = X.shape[1], len(X)
        self.enc, self.dec = _mlp([D, *self.hidden, 2 * self.m]).to(self.device), _mlp([self.m, *reversed(self.hidden), D]).to(self.device)
        self.center = X.mean(0); self.scale = ((X - self.center) ** 2).sum(1).mean().sqrt(); Xn = (X - self.center) / self.scale
        dist, idx = knn(X, self.K); self.spacing = dist[:, self.K_s - 1]; self.X = X
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * math.ceil(N / batch_size), pct_start=0.05)
        for ep in range(epochs):
            perm = torch.randperm(N, device=self.device)
            for s in range(0, N, batch_size):
                b = perm[s:s + batch_size]; x, nb = Xn[b], Xn[idx[b]]
                h = self.enc(x); mu, logvar = h[:, :self.m], h[:, self.m:].clamp(-10, 4)
                z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
                rec = ((self.dec(z) - x) ** 2).sum(1).mean()
                kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(1).mean()
                zn = self.enc(nb.reshape(-1, D))[:, :self.m].reshape(len(b), self.K, self.m)
                dx, dz = (nb - x[:, None]).norm(dim=2), (zn - mu[:, None]).norm(dim=2)
                geom = ((dz - dx) ** 2).mean() / (dx ** 2).mean()
                loss = rec + self.beta * kl + lam_geom * geom
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        self.eval()
        with torch.no_grad():
            h = self.enc(Xn); self.Z = h[:, :self.m]; self.logvar = h[:, self.m:].clamp(-10, 4)
            self.rho = knn(self.Z, self.K_s)[0][:, self.K_s - 1]
            self.kl_per_dim = 0.5 * (self.Z ** 2 + self.logvar.exp() - 1 - self.logvar).mean(0)
        return self


name = "SimpleStories/SimpleStories-5M"
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).cuda().eval(); tok = AutoTokenizer.from_pretrained(name)
X_emb, ids = tr.vocab_states(model, tok); keep = ~gm.degenerate_mask(X_emb); X_emb, ids = X_emb[keep], [i for i, k in zip(ids, keep.tolist()) if k]
prefix = tok("Once upon a time, there was a little", add_special_tokens=False)["input_ids"]
seqs = torch.tensor([prefix + [v] for v in ids]); pos = torch.full((len(ids),), len(prefix))
states = tr.collect_states(model, seqs, pos, ["L5.ffn"]); states["embed"] = X_emb.cpu()
perm = torch.randperm(len(ids), generator=torch.Generator().manual_seed(0)); n_val = len(ids) // 10; tr_i, va_i = perm[:-n_val], perm[-n_val:]
rows = []
for loc in ("embed", "L5.ffn"):
    X = states[loc].cuda(); Xtr, Xva = X[tr_i], X[va_i]; kernel = gm.KernelScore(Xtr); T = gm.TangentCharts(Xtr, 16)
    bands = {k: v for k, v in gm.support_report(gm.GlobalManifold(16).fit(Xtr, epochs=1), Xva, Xva, kernel, T).items() if k != "samples"}
    for beta, geom in ((0.0, 0.1), (1e-4, 0.1), (1e-3, 0.1), (1e-2, 0.1), (1e-3, 0.0), (1e-2, 0.0)):
        M = VAE(16, beta).fit(Xtr, X_val=Xva, lam_geom=geom)
        active = int((M.kl_per_dim > 0.01).sum())
        z_prior = torch.randn(2000, 16, device="cuda", generator=torch.Generator(device="cuda").manual_seed(0))
        x_prior = M.decode(z_prior)
        x_ours, info = M.sample(2000, alpha=0.3, auto_tau=True, seed=0)
        r = dict(loc=loc, beta=beta, lam_geom=geom, val_recon=float(M.recon_error(Xva).median()), active_units=active, mean_sigma=float((0.5 * M.logvar).exp().mean()),
                 prior=gm.support_report(M, x_prior, kernel=kernel, tangent=T)["samples"], prior_coverage=M.coverage(x_prior),
                 ours=gm.support_report(M, x_ours, kernel=kernel, tangent=T)["samples"], ours_coverage=M.coverage(x_ours), ours_ess=info["ess"])
        rows.append(r)
        print(f"{loc} beta={beta:g} geom={geom}: val recon {r['val_recon']:.3f} active {active}/16 sigma {r['mean_sigma']:.3f} | prior: u {r['prior']['u']:.3f} nearest {r['prior']['nearest_real']:.2f} tangent {r['prior']['tangent']:.2f} cov {r['prior_coverage']:.2f} | ours: u {r['ours']['u']:.3f} nearest {r['ours']['nearest_real']:.2f} cov {r['ours_coverage']:.2f} ess {info['ess']:.0f}", flush=True)
    print(f"{loc} bands:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in bands.items()}, flush=True)
json.dump(rows, open("results/vae_check.json", "w"), indent=1)

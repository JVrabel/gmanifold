"""GlobalManifold: learn G: R^m -> R^D from real states X and sample ~uniformly (w.r.t. manifold volume)
near the observed support. Knows nothing about Transformers."""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from .geometry import knn, uniform_ball


def _mlp(sizes, act=nn.SiLU):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), act()]
    return nn.Sequential(*layers[:-1])


class GlobalManifold(nn.Module):
    """Encoder E: D -> hidden -> m, decoder G: m -> reversed(hidden) -> D (SiLU).
    Preprocessing: subtract the training centre, divide by one global RMS scale (a similarity, so Euclidean
    geometry is preserved). The manifold ansatz is G(Omega) with Omega = union of latent balls B(z_i, alpha*rho_i)."""

    def __init__(self, latent_dim: int, hidden=(512, 256), K: int = 32, K_s: int = 8, device=None):
        super().__init__()
        self.m, self.hidden, self.K, self.K_s = int(latent_dim), tuple(hidden), K, K_s
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.enc = self.dec = None
        self.register_buffer("center", torch.zeros(0))
        self.register_buffer("scale", torch.ones(()))
        self.register_buffer("Z", torch.zeros(0, self.m))     # latent codes of the fitted states
        self.register_buffer("rho", torch.zeros(0))           # local latent scale (K_s-th latent neighbour)
        self.register_buffer("spacing", torch.zeros(0))       # local ambient spacing (K_s-th neighbour) of X
        self.X = None
        self.history = []
        self.spacing_unit = "anchor"                          # 'anchor': divide by the nearest anchor's spacing; 'global': by the median
        self.hub_share = 0.0                                  # share of points whose nearest neighbour is one of the 5 most common anchors

    # ------------------------------------------------------------------ maps
    def encode(self, x):
        return self.enc((x.to(self.device) - self.center) / self.scale)

    def decode(self, z):
        return self.dec(z.to(self.device)) * self.scale + self.center

    def project(self, x):
        return self.decode(self.encode(x))

    def jacobian(self, z, chunk=1024):
        """J_G(z) in ambient units, (B, D, m)."""
        f = torch.func.vmap(torch.func.jacfwd(self.dec))
        return torch.cat([f(z[s:s + chunk]) for s in range(0, len(z), chunk)]) * self.scale

    def log_volume(self, z, chunk=1024):
        """0.5 * log det(J^T J): log of the Riemannian volume element of the chart at z."""
        out = []
        for s in range(0, len(z), chunk):
            J = self.jacobian(z[s:s + chunk])
            out.append(0.5 * torch.linalg.slogdet(J.transpose(1, 2) @ J)[1])
        return torch.cat(out)

    # ------------------------------------------------------------------ fitting
    def _prepare(self, X, nn=None):
        """Centre/scale, K-neighbourhoods, spacing, hub check and fresh networks for a fit on X (already on device).
        Returns (Xn, idx): the normalised states and the (N, K) neighbour indices."""
        D, N = X.shape[1], len(X)
        self.enc, self.dec = _mlp([D, *self.hidden, self.m]).to(self.device), _mlp([self.m, *reversed(self.hidden), D]).to(self.device)
        self.center = X.mean(0)
        self.scale = ((X - self.center) ** 2).sum(1).mean().sqrt()
        if nn is None:
            dist, idx = knn(X, self.K)
        else:
            dist, idx = nn
            if len(dist) != N or dist.shape[1] < self.K:
                raise ValueError(f"nn must be knn(X, K) with K >= {self.K}: got shape {tuple(dist.shape)} for {N} rows")
            dist, idx = dist[:, :self.K].to(self.device), idx[:, :self.K].to(self.device)
        self.spacing = dist[:, self.K_s - 1]
        self.X = X
        counts = torch.bincount(idx[:, 0], minlength=N).float()
        self.hub_share = float(counts.sort(descending=True).values[:5].sum() / N)
        if float(self.spacing[idx[:, 0]].median()) < 0.5 * float(self.spacing.median()):   # hub-dominated cloud: anchors' spacings are not representative
            self.spacing_unit = "global"
            print(f"[gmanifold] hub-dominated cloud (top-5 anchors are the nearest neighbour of {self.hub_share:.0%} of the points): "
                  "distances are reported in units of the global median spacing")
        return (X - self.center) / self.scale, idx

    @staticmethod
    def _terms(enc, dec, x, nb, lam_geom, lam_curv, curv_delta):
        """Loss terms of one batch: x (B, D), nb (B, K, D) normalised states and their K neighbours; enc/dec callables."""
        B, K, D = nb.shape
        z, zn = enc(x), enc(nb.reshape(-1, D)).reshape(B, K, -1)
        xr = dec(z)
        dx, dz = (nb - x[:, None]).norm(dim=2), (zn - z[:, None]).norm(dim=2)
        terms = {"recon": ((xr - x) ** 2).sum(1).mean(), "geom": ((dz - dx) ** 2).mean() / (dx ** 2).mean()}
        if lam_curv > 0:
            d = torch.randn_like(z); d = d / d.norm(dim=1, keepdim=True) * (curv_delta * dz.detach().mean(1, keepdim=True))
            sec = dec(z + d) - 2 * xr + dec(z - d)
            terms["curv"] = ((sec ** 2).sum(1) / (d ** 2).sum(1)).mean()
        loss = terms["recon"] + lam_geom * terms["geom"] + lam_curv * terms.get("curv", 0.0)
        return loss, terms

    def _finish(self, X):
        self.eval()
        with torch.no_grad():
            self.Z = self.encode(X)
            self.rho = knn(self.Z, self.K_s)[0][:, self.K_s - 1]
        return self

    def fit(self, X, epochs=300, batch_size=512, lr=2e-3, lam_geom=0.1, lam_curv=1e-3, curv_delta=0.5, seed=0,
            X_val=None, log_every=50, verbose=False, nn=None):
        """L = recon + lam_geom * local-distance preservation (K Euclidean neighbours) + lam_curv * second
        differences along random latent directions. The geometry term ties latent distances to ambient
        distances; that is what makes the latent balls (and alpha) meaningful.
        `nn = knn(X, K')` with K' >= K reuses a neighbour search already done for X (e.g. by `intrinsic_dimension`
        or the masks) instead of repeating the N x N pass here."""
        torch.manual_seed(seed)
        X = X.to(self.device).float()
        N = len(X)
        Xn, idx = self._prepare(X, nn)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * math.ceil(N / batch_size), pct_start=0.05)
        self.train()
        for ep in range(epochs):
            perm, losses = torch.randperm(N, device=self.device), {}
            for s in range(0, N, batch_size):
                b = perm[s:s + batch_size]
                loss, terms = self._terms(self.enc, self.dec, Xn[b], Xn[idx[b]], lam_geom, lam_curv, curv_delta)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
                for k, v in terms.items():
                    losses[k] = losses.get(k, 0.0) + float(v.detach())
            rec = {k: v / math.ceil(N / batch_size) for k, v in losses.items()} | {"epoch": ep}
            if X_val is not None and (ep % log_every == 0 or ep == epochs - 1):
                rec["val_recon_over_spacing"] = float(self.recon_error(X_val).median())
            self.history.append(rec)
            if verbose and (ep % log_every == 0 or ep == epochs - 1):
                print({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()})
        return self._finish(X)

    # ------------------------------------------------------------------ sampling
    def radii(self, alpha, radius="global"):
        """Latent ball radii: 'global' = alpha x median rho for every anchor (a uniform-thickness neighbourhood of
        the data; robust to outliers), 'local' = alpha x rho_i per anchor (the union of local balls is then dominated
        by the largest balls, since ball volume scales like rho^m)."""
        rho = self.rho if radius == "local" else self.rho.median().expand_as(self.rho)
        return alpha * rho

    @torch.no_grad()
    def support_logq(self, z, R, anchor_logp, chunk=4096):
        """Exact log-density of the latent-ball mixture proposal (up to a constant) and ball multiplicity."""
        logq, cnt = [], []
        for s in range(0, len(z), chunk):
            d = torch.cdist(z[s:s + chunk], self.Z)
            inside = d <= R * (1 + 1e-5)
            term = torch.where(inside, anchor_logp - self.m * R.log(), torch.full_like(d, -float("inf")))
            logq.append(term.logsumexp(1)); cnt.append(inside.sum(1))
        return torch.cat(logq), torch.cat(cnt)

    @torch.no_grad()
    def sample(self, n, alpha=0.3, radius="global", n_candidates=None, anchor_power=None, reweight=True, tau=1.0, auto_tau=False,
               min_ess_frac=0.05, seed=None):
        """Approximately volume-uniform samples on G(Omega), Omega = union_i B(z_i, R_i) with R_i from `radii`.
        1. anchor i ~ P(i) ∝ R_i^p (p = m: the proposal is then uniform on Omega up to ball multiplicity);
        2. z uniform in B(z_i, R_i);  3. importance weight w = sqrt(det J^T J)^tau / q(z) with the exact mixture
        density q (tau = 1: exact surface-area weighting);  4. multinomial resampling of n points;  5. x = G(z).
        The weights are checked every call: if their effective sample size falls below `min_ess_frac` of the
        candidates (a few candidates with extreme volume elements absorb the mass), a warning is printed with the
        tau that would restore the floor. With `auto_tau=True` that tau is applied automatically.
        Returns (x, info) with info = z, anchor, multiplicity, ess, tau, tau_suggested, warning, anchor participation."""
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        n_c = n_candidates or 8 * n
        p = self.m if anchor_power is None else anchor_power
        R = self.radii(alpha, radius)
        anchor_logp = p * R.log(); anchor_logp = anchor_logp - anchor_logp.logsumexp(0)
        i = torch.multinomial(anchor_logp.exp(), n_c, replacement=True, generator=g)
        z = self.Z[i] + uniform_ball(n_c, self.m, R[i], g, self.device)
        logq, cnt = self.support_logq(z, R, anchor_logp)
        logvol = torch.nan_to_num(self.log_volume(z), nan=-float("inf"), posinf=-float("inf")) if reweight else torch.zeros(n_c, device=self.device)

        def weights(t):
            lw = t * logvol - logq if reweight else torch.zeros(n_c, device=self.device)   # no reweighting = the proposal itself
            lw = torch.nan_to_num(lw, nan=-float("inf"), posinf=-float("inf")); return (lw - lw.logsumexp(0)).exp()

        ess = lambda w: float(1 / (w ** 2).sum())
        w = weights(tau); info = dict(tau=tau, tau_suggested=tau, warning=None)
        if reweight and min_ess_frac and ess(w) < min_ess_frac * n_c:        # the checker
            lo, hi = 0.0, tau
            for _ in range(20):                                               # largest tempering exponent meeting the ESS floor
                mid_ = 0.5 * (lo + hi); lo, hi = (mid_, hi) if ess(weights(mid_)) >= min_ess_frac * n_c else (lo, mid_)
            top = float(w.sort(descending=True).values[:10].sum())
            info["tau_suggested"] = lo
            if auto_tau:
                w = weights(lo); info["tau"] = lo
            else:
                info["warning"] = (f"degenerate importance weights: ESS {ess(w):.0f} of {n_c} candidates, the 10 heaviest carry {top:.0%} of the mass; "
                                   f"the samples will be near-copies of a few points. Temper the volume weights with tau≈{lo:.2f} "
                                   f"(sample(..., tau={lo:.2f}) or auto_tau=True), and check the fit (held-out recon, latent dim, outlier states).")
                print("[gmanifold] WARNING " + info["warning"])
        pick = torch.multinomial(w, n, replacement=True, generator=g)
        info.update(z=z[pick], anchor=i[pick], multiplicity=cnt[pick], ess=ess(w), alpha=alpha, radius=radius, n_candidates=n_c,
                    anchor_participation=float(1 / (anchor_logp.exp() ** 2).sum()))
        return self.decode(z[pick]), info

    # ------------------------------------------------------------------ diagnostics
    def unit(self, j):
        """Length unit for points whose nearest fitted state is j: that state's spacing, or the global median."""
        return self.spacing[j] if self.spacing_unit == "anchor" else self.spacing.median().expand(len(j))

    @torch.no_grad()
    def recon_error(self, Y):
        """||G(E(y)) - y|| in spacing units (the off-manifold distance to the ansatz)."""
        Y = Y.to(self.device).float()
        j = knn(self.X, 1, Y)[1][:, 0]
        return (self.project(Y) - Y).norm(dim=1) / self.unit(j)

    @torch.no_grad()
    def nearest_real(self, Y):
        """Distance from Y to the nearest fitted real state, in spacing units."""
        d, j = knn(self.X, 1, Y.to(self.device).float())
        return d[:, 0] / self.unit(j[:, 0])

    @torch.no_grad()
    def coverage(self, Y):
        """Fraction of fitted real states that have a sample within one spacing unit."""
        d = knn(Y.to(self.device).float(), 1, self.X)[0][:, 0]
        return float((d / self.unit(torch.arange(len(self.X), device=self.device)) <= 1).float().mean())

    @torch.no_grad()
    def jacobian_rank(self, n=256, tol=1e-3):
        sv = torch.linalg.svdvals(self.jacobian(self.Z[:n]))
        return dict(rank_median=float((sv > tol * sv[:, :1]).sum(1).float().median()), singular_values=sv.mean(0).cpu())

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        torch.save(dict(m=self.m, hidden=self.hidden, K=self.K, K_s=self.K_s, D=self.X.shape[1], X=self.X.cpu(),
                        state=self.state_dict(), history=self.history, spacing_unit=self.spacing_unit, hub_share=self.hub_share), path)

    @classmethod
    def load(cls, path, device=None):
        s = torch.load(path, map_location="cpu")
        M = cls(s["m"], s["hidden"], s["K"], s["K_s"], device)
        M.enc, M.dec = _mlp([s["D"], *M.hidden, M.m]).to(M.device), _mlp([M.m, *reversed(M.hidden), s["D"]]).to(M.device)
        M.Z, M.rho, M.spacing, M.center = (torch.zeros_like(s["state"][k]) for k in ("Z", "rho", "spacing", "center"))
        M.load_state_dict(s["state"]); M.to(M.device); M.X = s["X"].to(M.device); M.history = s["history"]
        M.spacing_unit, M.hub_share = s.get("spacing_unit", "anchor"), s.get("hub_share", 0.0)
        return M.eval()


def fit_many(models, Xs, epochs=300, batch_size=512, lr=2e-3, lam_geom=0.1, lam_curv=1e-3, curv_delta=0.5, seed=0,
             X_vals=None, log_every=50, verbose=False, nn=None):
    """Fit several GlobalManifolds at once (one per cloud) with a single vmapped training loop: the same loss,
    optimiser and schedule as `fit`, but the C small MLPs are stacked so every step runs one batched matmul instead of
    C launches of a 512-row one, which is what keeps a GPU busy. Adam is elementwise and each model only receives the
    gradient of its own cloud, so the C fits are exactly independent; only the batch sampling differs from `fit` (each
    epoch has ceil(max N_c / batch_size) steps, clouds with fewer points wrap around their permutation).
    All models must share latent_dim, hidden and K, all clouds the ambient dimension D; N may differ.
    `nn[c] = knn(Xs[c], K')` reuses neighbour searches; `X_vals[c]` gives the held-out recon logged every `log_every`.
    Memory per step is C x batch_size x (K + 1) x D floats, so batch ~8-16 clouds at D = 1024-2048."""
    from torch.func import functional_call, stack_module_state, vmap
    C = len(models)
    if len(Xs) != C or (X_vals is not None and len(X_vals) != C) or (nn is not None and len(nn) != C):
        raise ValueError("models, Xs, X_vals and nn must have one entry per cloud")
    M0 = models[0]
    if any((M.m, M.hidden, M.K, M.K_s, M.device) != (M0.m, M0.hidden, M0.K, M0.K_s, M0.device) for M in models):
        raise ValueError("all models must share latent_dim, hidden, K, K_s and device")
    dev = M0.device
    torch.manual_seed(seed)
    Xs = [X.to(dev).float() for X in Xs]
    if any(X.shape[1] != Xs[0].shape[1] for X in Xs):
        raise ValueError("all clouds must have the same ambient dimension")
    prep = [M._prepare(X, None if nn is None else nn[c]) for c, (M, X) in enumerate(zip(models, Xs))]
    Xn, idx, Ns = [p[0] for p in prep], [p[1] for p in prep], [len(X) for X in Xs]
    steps = math.ceil(max(Ns) / batch_size)
    P_enc, _ = stack_module_state([M.enc for M in models]); P_dec, _ = stack_module_state([M.dec for M in models])
    enc0, dec0 = copy.deepcopy(M0.enc).to("meta"), copy.deepcopy(M0.dec).to("meta")

    def one(p_enc, p_dec, x, nb):
        return GlobalManifold._terms(lambda t: functional_call(enc0, p_enc, (t,)), lambda t: functional_call(dec0, p_dec, (t,)),
                                     x, nb, lam_geom, lam_curv, curv_delta)
    step_fn = vmap(one, randomness="different")

    def write_back():
        with torch.no_grad():
            for P, part in ((P_enc, "enc"), (P_dec, "dec")):
                for name, v in P.items():
                    for c, M in enumerate(models):
                        getattr(M, part).get_parameter(name).copy_(v[c])

    params = list(P_enc.values()) + list(P_dec.values())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps, pct_start=0.05)
    ar = torch.arange(batch_size, device=dev)
    for M in models:
        M.train()
    for ep in range(epochs):
        perms, losses = [torch.randperm(N, device=dev) for N in Ns], {}
        for s in range(steps):
            b = [perms[c][(s * batch_size + ar) % Ns[c]] for c in range(C)]
            x, nb = torch.stack([Xn[c][b[c]] for c in range(C)]), torch.stack([Xn[c][idx[c][b[c]]] for c in range(C)])
            loss, terms = step_fn(P_enc, P_dec, x, nb)
            opt.zero_grad(set_to_none=True); loss.sum().backward(); opt.step(); sched.step()
            for k, v in terms.items():
                losses[k] = losses.get(k, 0.0) + v.detach()
        log = ep % log_every == 0 or ep == epochs - 1
        if log:
            write_back()
        for c, M in enumerate(models):
            rec = {k: float(v[c]) / steps for k, v in losses.items()} | {"epoch": ep}
            if X_vals is not None and log:
                M.eval(); rec["val_recon_over_spacing"] = float(M.recon_error(X_vals[c]).median()); M.train()
            M.history.append(rec)
            if verbose and log:
                print({"cloud": c} | {k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()})
    write_back()
    for M, X in zip(models, Xs):
        M._finish(X)
    return models

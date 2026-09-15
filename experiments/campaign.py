"""Time-budgeted check campaign: keeps running sanity checks and small experiments on gmanifold until the budget is
used, appends one JSON line per check to results/checks/checks.jsonl, then writes results/checks/CHECKS.md.
python experiments/campaign.py --hours 8 [--quick]"""
import argparse
import itertools
import json
import math
import os
import subprocess
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gmanifold as gm
from gmanifold import transformer as tr
from gmanifold.geometry import knn

ap = argparse.ArgumentParser(); ap.add_argument("--hours", type=float, default=8.0); ap.add_argument("--quick", action="store_true"); ap.add_argument("--out", default="results/checks")
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True); T0 = time.time(); BUDGET = A.hours * 3600
EP = 20 if A.quick else 300
MODELS = ["SimpleStories/SimpleStories-5M", "SimpleStories/SimpleStories-11M", "SimpleStories/SimpleStories-30M", "SimpleStories/SimpleStories-35M",
          "SimpleStories/SimpleStories-V2-5M", "SimpleStories/SimpleStories-V2-35M", "SimpleStories/SimpleStories-1.25M"]
PREFIXES = ["Once upon a time, there was a little", "The", "One day, a big dog named Max saw a", "Lily and her mom went to the park. It was a sunny day and they saw a",
            "In the morning, the", "She opened the box and found a"]
_cache = {}


def load(name):
    if name not in _cache:
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).cuda().eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _cache.clear(); _cache[name] = (m, AutoTokenizer.from_pretrained(name)); torch.cuda.empty_cache()
    return _cache[name]


def cloud(name, prefix, locs):
    """Prefix + every real token: embedding rows and residual states at `locs`; degenerate/outlier tokens dropped."""
    model, tok = load(name)
    X, ids = tr.vocab_states(model, tok); keep = ~gm.degenerate_mask(X) & ~gm.outlier_mask(X); X, ids = X[keep], [v for v, k in zip(ids, keep.tolist()) if k]
    p = tok(prefix, add_special_tokens=False)["input_ids"]
    if tok.bos_token_id is not None and tok.bos_token_id not in tok.all_special_ids[:0]:
        pass
    seqs = torch.tensor([p + [v] for v in ids]); pos = torch.full((len(ids),), len(p))
    st = tr.collect_states(model, seqs, pos, [l for l in locs if l != "embed"]); st["embed"] = X.cpu()
    return model, tok, ids, seqs, len(p), {l: v.cuda() for l, v in st.items()}


def split(n, seed):
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)); k = n // 10
    return perm[k:], perm[:k]


def bands_and(M, Xva, kernel, T, Y):
    r = gm.support_report(M, Y, Xva, kernel, T)
    return r["samples"], {k: v for k, v in r.items() if k != "samples"}


def mid(b, key):                                            # midpoint between held-out-real and 0.5x-noise bands
    return 0.5 * (b["held-out real"][key] + b["real + 0.5x noise"][key])


# ------------------------------------------------------------------ check types
def synthetic(m, D, N, seed):
    """Curved sheet in R^D with a known volume element (curvature along the first coordinate)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    A_ = torch.linalg.qr(torch.randn(D, m + 1, device="cuda", generator=g))[0].T * 3
    st = torch.rand(N, m, device="cuda", generator=g)
    X = torch.cat([st, 0.8 * torch.sin(3 * st[:, :1])], 1) @ A_
    vol = lambda s: torch.sqrt(1 + (2.4 * torch.cos(3 * s)) ** 2)
    n_val = N // 10; Xtr, Xva, sttr = X[n_val:], X[:n_val], st[n_val:]
    M = gm.GlobalManifold(latent_dim=m, hidden=(128, 64)).fit(Xtr, epochs=EP, X_val=Xva, seed=seed)
    target = float((vol(sttr[:, 0]) ** 2).mean() / vol(sttr[:, 0]).mean()); data = float(vol(sttr[:, 0]).mean()); gap = target - data
    def mean_vol(**kw):
        x, info = M.sample(4000, seed=seed, **kw); return float(vol(sttr[knn(Xtr, 1, x)[1][:, 0]][:, 0]).mean()), info
    v_loc, info = mean_vol(alpha=0.5, radius="local"); v_glob1, _ = mean_vol(alpha=1.0); v_none, _ = mean_vol(alpha=0.5, anchor_power=0, reweight=False)
    x, _ = M.sample(2000, alpha=0.3, seed=seed); s, b = bands_and(M, Xva, gm.KernelScore(Xtr), None, x)
    r = dict(val_recon=M.history[-1]["val_recon_over_spacing"], rank=M.jacobian_rank()["rank_median"], uniform_local_err=abs(v_loc - target) / gap, uniform_global1_err=abs(v_glob1 - target) / gap,
             density_err=abs(v_none - data) / gap, ess_frac=info["ess"] / info["n_candidates"], u_samples=s["u"], u_real=b["held-out real"]["u"], u_noise=b["real + 1x noise"]["u"])
    r["pass"] = bool(r["val_recon"] < 0.5 and r["rank"] == m and r["uniform_local_err"] < 0.4 and r["uniform_global1_err"] < 0.5 and r["density_err"] < 0.4 and r["ess_frac"] > 0.2 and s["u"] > mid(b, "u") and b["real + 1x noise"]["u"] < b["held-out real"]["u"])
    return r


def determinism(name, prefix, seed):
    model, tok, ids, seqs, L, st = cloud(name, prefix, ["embed"]); X = st["embed"]; tr_i, va_i = split(len(ids), 0)
    fits = []
    for _ in range(2):
        torch.manual_seed(seed); M = gm.GlobalManifold(latent_dim=16).fit(X[tr_i], epochs=EP, seed=seed); fits.append(M)
    p = [torch.cat([q.flatten() for q in M.parameters()]) for M in fits]
    x1, _ = fits[0].sample(500, 0.3, seed=1); x2, _ = fits[1].sample(500, 0.3, seed=1)
    path = os.path.join(A.out, "tmp_model.pt"); fits[0].save(path); M2 = gm.GlobalManifold.load(path); x3, _ = M2.sample(500, 0.3, seed=1); os.remove(path)
    r = dict(param_diff=float((p[0] - p[1]).abs().max()), sample_diff=float((x1 - x2).abs().max()), saveload_diff=float((x1 - x3).abs().max()))
    r["pass"] = bool(r["param_diff"] < 1e-5 and r["sample_diff"] < 1e-4 and r["saveload_diff"] < 1e-5)
    return r


def real_alpha(name, prefix, seed, m=16, n_locs=5):
    model, tok, ids, seqs, L, st0 = cloud(name, prefix, ["embed"]); locs = tr.locations(model)
    pick = [locs[0]] + [locs[i] for i in sorted(set(int(round(k * (len(locs) - 1) / (n_locs - 1))) for k in range(1, n_locs)))]
    model, tok, ids, seqs, L, st = cloud(name, prefix, pick); tr_i, va_i = split(len(ids), seed)
    rows, ok = {}, True
    for loc in pick:
        X = st[loc]; Xtr, Xva = X[tr_i], X[va_i]
        M = gm.GlobalManifold(latent_dim=m).fit(Xtr, epochs=EP, X_val=Xva, seed=seed); kernel, T = gm.KernelScore(Xtr), gm.TangentCharts(Xtr, m)
        row = dict(val_recon=M.history[-1]["val_recon_over_spacing"], unit=M.spacing_unit, hub=M.hub_share)
        for alpha in (0.3, 0.7, 1.0, 1.5):
            x, info = M.sample(2000, alpha=alpha, seed=seed); s, b = bands_and(M, Xva, kernel, T, x)
            row[f"a{alpha}"] = dict(u=s["u"], nearest=s["nearest_real"], tangent=s["tangent"], coverage=M.coverage(x), ess=info["ess"])
        row["bands"] = {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in b.items()}
        cov_min = 0.5 if M.spacing_unit == "global" else 0.8
        row["pass"] = bool(row["a0.3"]["u"] > mid(b, "u") and row["a0.3"]["nearest"] < b["held-out real"]["nearest_real"] and row["a0.3"]["tangent"] < b["held-out real"]["tangent"]
                           and row["a0.3"]["coverage"] >= cov_min and row["a0.3"]["ess"] > 0.01 * 16000 and row["a1.5"]["u"] < b["held-out real"]["u"])
        ok &= row["pass"]; rows[loc] = row
        del M, kernel, T; torch.cuda.empty_cache()
    return dict(locations=rows, **{"pass": bool(ok)})


def propagation(name, prefix, seed, m=16):
    model, tok, ids, seqs, L, st0 = cloud(name, prefix, ["embed"]); locs = tr.locations(model)
    model, tok, ids, seqs, L, st = cloud(name, prefix, locs); tr_i, va_i = split(len(ids), seed)
    M_in = gm.GlobalManifold(latent_dim=m).fit(st["embed"][tr_i], epochs=EP, seed=seed); x, _ = M_in.sample(2000, alpha=0.3, seed=seed)
    Xva = st["embed"][va_i]; j = knn(M_in.X, 1, Xva)[1][:, 0]; g = torch.Generator(device="cuda").manual_seed(seed)
    nz = torch.randn(Xva.shape, device="cuda", generator=g); noise = Xva + nz / nz.norm(dim=1, keepdim=True) * M_in.unit(j)[:, None]
    rows, ok = {}, True
    for dst in locs[1:]:
        f = tr.make_map(model, "embed", dst, seqs[0], pos=L); Y = st[dst]; Ytr, Yva = Y[tr_i], Y[va_i]
        M_out = gm.GlobalManifold(latent_dim=m).fit(Ytr, epochs=EP, X_val=Yva, seed=seed); k_out = gm.KernelScore(Ytr)
        s, b = bands_and(M_out, Yva, k_out, None, f(x)); real_img = gm.support_report(M_out, f(Xva), kernel=k_out)["samples"]; noise_img = gm.support_report(M_out, f(noise), kernel=k_out)["samples"]
        rows[dst] = dict(u_images=s["u"], u_real_images=real_img["u"], u_noise_images=noise_img["u"], nearest_images=s["nearest_real"], nearest_real_images=real_img["nearest_real"], u_mid=mid(b, "u"))
        rows[dst]["pass"] = bool(s["u"] > mid(b, "u") and s["nearest_real"] < 1.2 * real_img["nearest_real"]); ok &= rows[dst]["pass"]
        del M_out, k_out; torch.cuda.empty_cache()
    return dict(destinations=rows, **{"pass": bool(ok)})


def next_token(name, prefix, seed, m=16):
    model, tok, ids, seqs, L, st = cloud(name, prefix, ["embed"]); X = st["embed"]; tr_i, va_i = split(len(ids), seed)
    M = gm.GlobalManifold(latent_dim=m).fit(X[tr_i], epochs=EP, seed=seed)
    @torch.no_grad()
    def lp(x):
        out = []
        for xb in x.split(512):
            with tr._Hooks(model, [], {"embed": (xb, torch.full((len(xb),), L, device="cuda"))}):
                out.append(model(input_ids=seqs[0].cuda()[None].expand(len(xb), -1)).logits[:, L].log_softmax(-1))
        return torch.cat(out)
    def js(p, q):
        mm = 0.5 * (p.exp() + q.exp()); return 0.5 * ((p.exp() * (p - mm.log())).sum(-1) + (q.exp() * (q - mm.log())).sum(-1))
    lpr = lp(X[tr_i]); r = {}
    for label, x in (("real", X[va_i]), ("samples", M.sample(2000, alpha=0.3, seed=seed)[0])):
        l = lp(x); j2 = knn(X[tr_i], 2, x)[1]; rnd = torch.randint(len(tr_i), (len(x),), generator=torch.Generator().manual_seed(1)).cuda()
        r[label] = dict(js_nearest=float(js(l, lpr[j2[:, 0]]).median()), js_random=float(js(l, lpr[rnd]).median()), top1=float((l.argmax(-1) == lpr[j2[:, 0]].argmax(-1)).float().mean()))
    r["pass"] = bool(r["samples"]["js_nearest"] < r["real"]["js_nearest"] and r["samples"]["js_nearest"] < 0.6 * r["samples"]["js_random"] and r["samples"]["top1"] > r["real"]["top1"])
    return r


def scaling(name, prefix, seed, loc="embed"):
    model, tok, ids, seqs, L, st = cloud(name, prefix, [loc]); X = st[loc]; tr_i, va_i = split(len(ids), seed)
    M = gm.GlobalManifold(latent_dim=16).fit(X[tr_i], epochs=EP, seed=seed); r = {}
    for n in (500, 2000, 8000, 32000):
        x, info = M.sample(n, alpha=0.3, seed=seed); r[str(n)] = dict(coverage=M.coverage(x), ess_frac=info["ess"] / info["n_candidates"], nearest=float(M.nearest_real(x).median()))
    covs = [r[k]["coverage"] for k in ("500", "2000", "8000", "32000")]
    r["pass"] = bool(all(covs[i] <= covs[i + 1] + 0.02 for i in range(3)) and max(r[k]["ess_frac"] for k in r if k != "pass") / max(1e-9, min(r[k]["ess_frac"] for k in r if k != "pass")) < 3)
    return r


def hyper(name, prefix, seed, loc="embed"):
    model, tok, ids, seqs, L, st = cloud(name, prefix, [loc]); X = st[loc]; tr_i, va_i = split(len(ids), seed); Xtr, Xva = X[tr_i], X[va_i]
    kernel = gm.KernelScore(Xtr); r = {}
    for label, kw, skw in (("default", {}, {}), ("K16", dict(K=16), {}), ("K64", dict(K=64), {}), ("Ks4", dict(K_s=4), {}), ("Ks16", dict(K_s=16), {}),
                           ("cand4n", {}, dict(n_candidates=8000)), ("cand16n", {}, dict(n_candidates=32000)), ("hidden256", dict(hidden=(256, 128)), {})):
        M = gm.GlobalManifold(latent_dim=16, **kw).fit(Xtr, epochs=EP, X_val=Xva, seed=seed); x, info = M.sample(2000, alpha=0.3, seed=seed, **skw)
        s = gm.support_report(M, x, kernel=kernel)["samples"]; r[label] = dict(val_recon=M.history[-1]["val_recon_over_spacing"], u=s["u"], nearest=s["nearest_real"], coverage=M.coverage(x), ess_frac=info["ess"] / info["n_candidates"])
    u0 = r["default"]["u"]; r["pass"] = bool(all(abs(v["u"] - u0) < 0.08 and v["coverage"] > 0.7 for k, v in r.items() if k not in ("pass",)))
    return r


def memory(name, prefix, seed):
    model, tok, ids, seqs, L, st = cloud(name, prefix, ["embed"]); X = st["embed"]; peaks = []
    for i in range(12):
        torch.cuda.reset_peak_memory_stats(); M = gm.GlobalManifold(latent_dim=16).fit(X, epochs=max(5, EP // 10), seed=seed + i); M.sample(2000, 0.3, seed=i); del M; torch.cuda.empty_cache()
        peaks.append(torch.cuda.max_memory_allocated() / 1e9)
    return dict(peak_first=peaks[0], peak_last=peaks[-1], allocated_after=torch.cuda.memory_allocated() / 1e9, **{"pass": bool(peaks[-1] < 1.5 * peaks[0] + 0.5)})


def stories(name, seed, n=10000, locs=("L1.attn", "L1.ffn", "L3.ffn")):
    from datasets import load_dataset
    model, tok = load(name); g = torch.Generator().manual_seed(seed); seqs, story, positions = [], [], []
    for ex in load_dataset("SimpleStories/SimpleStories", split="train", streaming=True):
        ids = tok(ex["story"], add_special_tokens=False)["input_ids"][:256]
        if len(ids) < 2:
            continue
        sid = len(set(story))
        for p in torch.randperm(len(ids), generator=g)[:8].tolist():
            seqs.append(ids); story.append(sid); positions.append(p)
        if len(positions) >= n:
            break
    story = torch.tensor(story); Lm = max(len(s) for s in seqs); ids_t = torch.full((len(seqs), Lm), tok.eos_token_id); mask = torch.zeros((len(seqs), Lm), dtype=torch.long)
    for r_, s in enumerate(seqs):
        ids_t[r_, :len(s)] = torch.tensor(s); mask[r_, :len(s)] = 1
    st = tr.collect_states(model, ids_t, torch.tensor(positions), list(locs), attention_mask=mask, batch_size=64)
    val_story = torch.randperm(int(story.max()) + 1, generator=torch.Generator().manual_seed(seed))[: (int(story.max()) + 1) // 10]; is_val = torch.isin(story, val_story)
    rows, ok = {}, True
    for loc in locs:
        X = st[loc].cuda(); Xtr, Xva = X[~is_val], X[is_val]; Xtr = Xtr[~gm.degenerate_mask(Xtr) & ~gm.outlier_mask(Xtr)]
        M = gm.GlobalManifold(latent_dim=16).fit(Xtr, epochs=max(20, EP // 3), X_val=Xva, seed=seed); kernel = gm.KernelScore(Xtr)
        x, info = M.sample(2000, alpha=0.3, seed=seed); s, b = bands_and(M, Xva, kernel, None, x)
        rows[loc] = dict(val_recon=M.history[-1]["val_recon_over_spacing"], u=s["u"], nearest=s["nearest_real"], u_mid=mid(b, "u"), nearest_noise=b["real + 0.5x noise"]["nearest_real"], ess=info["ess"],
                         pass_=bool(s["u"] > mid(b, "u") and s["nearest_real"] < b["real + 0.5x noise"]["nearest_real"]))
        ok &= rows[loc]["pass_"]; del M, kernel; torch.cuda.empty_cache()
    return dict(n=len(positions), locations=rows, **{"pass": bool(ok)})


# ------------------------------------------------------------------ queue
def jobs():
    yield "synthetic", synthetic, dict(m=2, D=64, N=5000, seed=0)
    yield "determinism", determinism, dict(name=MODELS[0], prefix=PREFIXES[0], seed=0)
    yield "real_alpha", real_alpha, dict(name=MODELS[0], prefix=PREFIXES[0], seed=0)
    yield "propagation", propagation, dict(name=MODELS[0], prefix=PREFIXES[0], seed=0)
    yield "next_token", next_token, dict(name=MODELS[0], prefix=PREFIXES[0], seed=0)
    yield "scaling", scaling, dict(name=MODELS[0], prefix=PREFIXES[0], seed=0)
    yield "hyper", hyper, dict(name=MODELS[0], prefix=PREFIXES[0], seed=0)
    yield "memory", memory, dict(name=MODELS[0], prefix=PREFIXES[0], seed=0)
    yield "stories", stories, dict(name=MODELS[0], seed=0)
    for m, D, N, seed in [(3, 64, 8000, 2), (4, 128, 12000, 3), (2, 256, 5000, 4), (2, 64, 1500, 5), (2, 64, 20000, 6), (8, 128, 20000, 7), (6, 512, 15000, 8)]:
        yield "synthetic", synthetic, dict(m=m, D=D, N=N, seed=seed)
    for i in itertools.count():                                              # endless: models x prefixes x seeds
        name, prefix, seed = MODELS[i % len(MODELS)], PREFIXES[(i // len(MODELS)) % len(PREFIXES)], i // (len(MODELS) * len(PREFIXES))
        yield "real_alpha", real_alpha, dict(name=name, prefix=prefix, seed=seed)
        if i % 2 == 0:
            yield "propagation", propagation, dict(name=name, prefix=prefix, seed=seed)
        if i % 3 == 0:
            yield "next_token", next_token, dict(name=name, prefix=prefix, seed=seed)
        if i % 4 == 1:
            yield "determinism", determinism, dict(name=name, prefix=prefix, seed=seed)
        if i % 5 == 2:
            yield "stories", stories, dict(name=name, seed=seed)
        if i % 6 == 3:
            yield "hyper", hyper, dict(name=name, prefix=prefix, seed=seed, loc="L2.ffn")
        if i % 7 == 4:
            yield "scaling", scaling, dict(name=name, prefix=prefix, seed=seed)
        if i % 8 == 5:
            yield "memory", memory, dict(name=name, prefix=prefix, seed=seed)
        if i % 9 == 6:
            yield "synthetic", synthetic, dict(m=2 + i % 5, D=64 * (1 + i % 3), N=4000 + 2000 * (i % 4), seed=100 + i)


def summarize():
    rows = [json.loads(l) for l in open(os.path.join(A.out, "checks.jsonl"))]
    by = {}
    for r in rows:
        by.setdefault(r["check"], []).append(r)
    md = [f"# gmanifold check campaign\n", f"{len(rows)} checks in {(time.time() - T0) / 3600:.1f} h on one RTX 5090; "
          f"{sum(1 for r in rows if r.get('pass'))} passed, {sum(1 for r in rows if r.get('pass') is False)} failed, {sum(1 for r in rows if r.get('error'))} errored.\n",
          "| check | runs | passed | failed | errors | models | seconds/run |", "|---|---|---|---|---|---|---|"]
    for k, v in by.items():
        models = sorted({str(r["args"].get("name", "synthetic")).split("/")[-1] for r in v})
        md.append(f"| {k} | {len(v)} | {sum(1 for r in v if r.get('pass'))} | {sum(1 for r in v if r.get('pass') is False)} | {sum(1 for r in v if r.get('error'))} | {', '.join(models)} | {sum(r['seconds'] for r in v) / len(v):.0f} |")
    fails = [r for r in rows if r.get("pass") is False or r.get("error")]
    if fails:
        md.append("\n## Failures and errors\n")
        for r in fails:
            md.append(f"* `{r['check']}` {r['args']}: " + (f"error `{r['error'][:200]}`" if r.get("error") else "failed criteria; see checks.jsonl"))
    md.append("\n## What each check asserts\n")
    md.append("* **synthetic** — curved sheet with known volume element: held-out recon < 0.5 spacings, full Jacobian rank, local-radius sampling within 0.4 of the volume-uniform target (global radius at α = 1 within 0.5), un-reweighted sampler within 0.4 of the data density, ESS > 20 % of candidates, samples above the u midpoint, noise below real.\n"
              "* **determinism** — two fits with the same seed give identical parameters (< 1e-5) and samples; save/load reproduces samples.\n"
              "* **real_alpha** — prefix + every token, 5 locations: α = 0.3 samples above the u midpoint between held-out real and 0.5× noise, closer than held-out real on nearest-real and tangent residual, coverage ≥ 0.8 (0.5 on hub-dominated clouds), ESS > 1 % of candidates; α = 1.5 below the real u.\n"
              "* **propagation** — embedding samples pushed to every location score above the u midpoint of the independent destination fit and within 1.2× the nearest-real distance of real images.\n"
              "* **next_token** — samples' next-token distributions closer to their nearest token's than real tokens are to their neighbours', < 0.6× the random-token divergence, top-1 agreement above the real baseline.\n"
              "* **scaling** — coverage non-decreasing with the number of samples (500 → 32k), ESS fraction stable within 3×.\n"
              "* **hyper** — K ∈ {16, 32, 64}, K_s ∈ {4, 8, 16}, candidates 4n–16n, a narrower net: sample u within 0.08 of the default, coverage > 0.7.\n"
              "* **memory** — 12 fit+sample cycles: peak GPU memory of the last cycle < 1.5× the first + 0.5 GB.\n"
              "* **stories** — 10k real occurrences split by story, 3 locations: α = 0.3 samples above the u midpoint and closer to real states than the 0.5× noise band (token-clustered clouds put held-out states from other stories unusually close).\n")
    open(os.path.join(A.out, "CHECKS.md"), "w").write("\n".join(md)); print("\n".join(md[:3]))


durations = {}
log = open(os.path.join(A.out, "checks.jsonl"), "a")
for i, (check, fn, kw) in enumerate(jobs()):
    elapsed = time.time() - T0; est = durations.get(check, 300)
    if elapsed + est > BUDGET:
        if elapsed > BUDGET or all(elapsed + d > BUDGET for d in durations.values()):
            break
        continue                                                            # skip jobs that would not fit; a shorter type may still fit
    t = time.time(); rec = dict(i=i, check=check, args=kw, time=time.strftime("%H:%M:%S", time.gmtime()))
    try:
        rec.update(fn(**kw))
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"; traceback.print_exc()
    rec["seconds"] = time.time() - t; durations[check] = 0.8 * durations.get(check, rec["seconds"]) + 0.2 * rec["seconds"]
    log.write(json.dumps(rec) + "\n"); log.flush(); torch.cuda.empty_cache()
    print(f"[{(time.time() - T0) / 3600:5.2f}h] {check} {kw} -> {'PASS' if rec.get('pass') else ('ERROR' if rec.get('error') else 'FAIL')} ({rec['seconds']:.0f}s)", flush=True)
    if A.quick and i >= 12:
        break
summarize()

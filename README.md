# gmanifold

**Learn a smooth low-dimensional parameterisation G: R^m → R^D of a finite set of real states X ⊂ R^D, sample
approximately uniformly (w.r.t. manifold volume) on the learned manifold near the observed support, and optionally
push the samples through a real Transformer map.** Plain PyTorch; the manifold code knows nothing about Transformers.

```
      real states X_in ⊂ R^D            (one representation location, e.g. embedding rows or residual states)
              │
              ▼  fit: E: R^D→R^m,  G: R^m→R^D     loss = recon + λ_geom·(kNN distances preserved) + λ_curv·(2nd diff)
      ┌───────────────────┐
      │  GlobalManifold   │   latent codes z_i = E(x_i),  support Ω = ∪ B(z_i, α·ρ_i)   (ρ_i = 8-th latent neighbour)
      └───────────────────┘
              │  sample(n, α):  R_i = α·median(ρ) (default) → anchor i ∝ R_i^m → z ~ U(B_i) → weight √det(JᵀJ)/q(z) → resample → x = G(z)
              ▼
      samples X_sample ≈ uniform on G(Ω), the tube around the observed states     info: ESS, multiplicity, z, anchors
              │
              ▼  map_fn = the real Transformer stretch (FFN sub-layer, attention under a fixed context, F_{k→k'})
      images  Y_sample = map_fn(X_sample)
              │
              ▼  compare with an INDEPENDENT fit  M_out = GlobalManifold().fit(X_out)   (never trained on Y_sample)
      validation: nearest-real / spacing, recon error, kernel score u (Guidotti), tangent-chart residual — vs held-out-real and noise bands
```

> **Limitations, stated up front.** This is a parameterisation of a finite set of observed states, not the true manifold.
> It builds in assumptions we consider likely: that the states near each other lie on a smooth low-dimensional surface,
> that interpolating within about one local spacing of observed states stays on it, and that a global chart of
> dimension m is adequate. The samples are uniform within a tube around the observations, not in the manifold's own
> volume; regions without nearby observations are not represented; the learned surface passes ~0.7 local spacings
> from unseen real states; and the validators encode the same interpolation assumption rather than testing it
> independently. The true shape of the set stays hidden; this is our best-supported approximation to it, with every
> claim tied to a calibration band you can inspect. Details in `DESIGN.md` §6.

## TL;DR

```python
import gmanifold as gm
M_in = gm.GlobalManifold(latent_dim=16).fit(X_in)            # X_in: [N, D] real states at the source location
X_sample, info = M_in.sample(n=5000, alpha=0.3)              # ≈ volume-uniform points on the learned manifold
Y_sample = map_fn(X_sample)                                  # the real Transformer operation of interest
M_out = gm.GlobalManifold(latent_dim=16).fit(X_out)          # independent destination fit (if X_out is known)
gm.support_report(M_out, Y_sample, X_out_heldout, gm.KernelScore(X_out), gm.TangentCharts(X_out, 16))
```

* **`alpha` is how far from the observed states you sample**, in units of one typical inter-point distance
  (the median distance from a code to its 8-th nearest code; the local-distance loss makes latent and ambient distances
  match). Every sample lies within α of some real state's code. Measured against held-out-real and noise bands on three
  model sizes (up to 45 locations), and by what the model predicts next:

  | α | verdict |
  |---|---|
  | **0.3 (default)** | on-support everywhere by all validators, coverage 0.97–0.99, highest ESS; next-token predictions like a blend of the two nearest real tokens |
  | 0.5–0.7 | still on-support at every tested location; diagnostics and next-token similarity start to loosen |
  | 1.0 | marginal (just above the half-spacing noise band) |
  | ≥ 1.5 | off-support |

  Use 0.3 for anything that must be trusted, 0.5–0.7 to deliberately move away from the observations (and check
  `support_report`), never above 1. Holds for the default global radius; with `radius="local"` only α ≤ 0.3 was safe.
* `info["ess"]` (effective sample size of the importance resampling) should be a sizeable fraction of
  `8 n` candidates; `M.coverage(X_sample)` says how much of the real cloud the samples reach.
* `radius="global"` (default) gives every anchor the same latent radius α × median ρ: a uniform-thickness neighbourhood of
  the data. Samples are uniform in volume *within that tube* (anchors are equalised, unlike the data density), not in the
  manifold's own volume: sparse regions' large cells are not filled (results summary, item 10). `radius="local"` uses α × ρ_i per anchor; because ball volume scales like ρ^m, the union is
  then dominated by the few sparsest anchors (one massive-activation outlier state in TinyLlama takes 100 % of the anchor mass),
  which destroys coverage — see the sampler-variant table in the results.
* Length unit: distances are divided by the spacing (8-th-neighbour distance) of the nearest fitted state; when a few hub
  anchors with tiny spacing are the nearest neighbour of most points (TinyLlama embeddings: 74 % of points on 5 anchors)
  the model switches to the global median spacing and says so (`spacing_unit`, `hub_share`).
* Clean X first: near-duplicate rows (e.g. the 110 untrained byte-fallback embeddings of TinyLlama) make the local spacing
  meaningless, and isolated massive-activation states (deep Llama layers, ambient norm 20× the median) attract all of the
  volume weight (ESS → 1). Drop both: `X = X[~gm.degenerate_mask(X) & ~gm.outlier_mask(X)]`. The sampler warns when ESS
  falls below 1 % of the candidates.
* The un-reweighted variant `sample(..., anchor_power=0, reweight=False)` follows the empirical data density
  instead of manifold volume (verified on a synthetic sheet with a known volume element, see `tests/`).
* Validation is always relative to **calibration bands**: held-out real states, and real states displaced by
  0.5× / 1× their local spacing. The Guidotti kernel score `u` (1 on the fitted states) is the most sensitive
  validator below one spacing; it is never used for sampling.

Long-form notes on every choice and experiment: `DESIGN.md`.

## Install and run

```bash
pip install -e .            # torch only; add [transformer] for the Hugging Face helpers
pytest tests                # synthetic-sheet tests (exact volume-uniformity check) + SimpleStories hook tests
python examples/simplestories.py --alpha 0.3 --dst L0.ffn
```

## Experiments

`experiments/sweep.py --model <hf checkpoint> --out results/<name> [--m 16 32] [--seeds 0 1] [--ablation]` runs the
sampling-centric protocol on the prefix + all-tokens clouds of any Llama-style model (dimension, fit quality, α sweep
with calibration bands, sampler/loss ablations, propagation with independent destination fits);
`experiments/report.py` turns `results/*/sweep.json` into `results/results.md` and `results/results.pdf`;
`experiments/run_all.sh` is the scheduled run used for the shipped results (tests → 5M → 35M → TinyLlama → report);
`experiments/stories.py` does the same for real-occurrence clouds (random token occurrences in SimpleStories stories).

## API

| | |
|---|---|
| `GlobalManifold(latent_dim, hidden=(512,256), K=32, K_s=8)` | `fit(X, epochs=300, lr=2e-3, lam_geom=0.1, lam_curv=1e-3, X_val=None)` → self |
| `sample(n, alpha=0.3, radius="global", n_candidates=8n, anchor_power=m, reweight=True, seed=None)` | → `(X_sample, info)` |
| `encode / decode / project / jacobian / log_volume` | the maps and the chart's volume element |
| `recon_error(Y)`, `nearest_real(Y)`, `coverage(Y)`, `jacobian_rank()` | diagnostics (distances in units of local spacing) |
| `save(path)` / `GlobalManifold.load(path)` | persistence |
| `KernelScore(X, sigma=None, lam=1e-6)(Y)` | Guidotti kernel signature u(Y) as an independent validator (arXiv:2404.00427) |
| `TangentCharts(X, m, K=max(4m,32))` | local PCA charts: `.residual(Y)` (normal distance to the nearest chart / spacing) as a second validator, `.sample(n, alpha)` as a conservative cross-check sampler |
| `support_report(M, samples, X_heldout, kernel, tangent)` | medians of all diagnostics with calibration bands |
| `intrinsic_dimension(X)`, `degenerate_mask(X)`, `outlier_mask(X)` | TwoNN / MLE estimates (to choose `latent_dim`); near-duplicate and isolated rows to drop |
| `transformer.vocab_states / collect_states / make_map / locations` | optional Hugging Face helpers for Llama-style (`model.model.layers`) and GPT-2/GPT-Neo-style (`model.transformer.h`) models; hooks verified exact |

**Architecture and hyper-parameters.** A capacity study (`experiments/capacity.py`, §7 of the results) shows the fit at
fixed m is neither capacity- nor optimisation-limited (widths 256–2048, depth 1–3, lr, batch size, λ_geom and epochs all
move held-out reconstruction by ≤ 0.006 spacings); only `latent_dim` matters, improving reconstruction smoothly up to
m = 64 while lowering ESS and coverage. Use m = 16–24 for sampling, larger m with more candidates for tighter fits.

Preprocessing is one centre and one global RMS scale (a similarity, so Euclidean geometry is untouched). The decoder is an
MLP `D → 512 → 256 → m` / `m → 256 → 512 → D` with SiLU; hidden sizes are configurable.

## What the results say (see `results/results.pdf`)

Fitted per residual location on "prefix + every real token" clouds of SimpleStories-5M / 35M (13 / 25 locations, up to
5 seeds, two prefixes) and TinyLlama-1.1B (8 locations): samples score above held-out real states (kernel u ≥ 1.03,
coverage 0.97–0.99) for α ≤ 0.7 at every location, are marginal at α = 1 and off-support at α = 1.5; their images
under the real model land on independently fitted destination manifolds like real held-out images; the volume
reweighting is exact on a synthetic sheet; the global support radius and the outlier filter are what make the sampler
robust (per-anchor radii let one massive-activation state absorb all the weight). One cloud fooled us for a while: TinyLlama's 32k-token embedding matrix looked unfittable ("6 spacings") until we found that five hub tokens with tiny spacing dominate the per-anchor length unit; in raw or global-median units it behaves like the others, and the library now detects such hub-dominated clouds (`M.spacing_unit`, `M.hub_share`).

Out of scope on purpose: anything downstream of the propagated samples; this library produces and validates them.

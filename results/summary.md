## TL;DR

Fit one decoder G: R^16 → R^D per residual location on the real token clouds of SimpleStories-5M, -35M and
TinyLlama-1.1B (prefix + every real token; also real story occurrences), then sample on the learned surface within
α × spacing of the observed states and validate against three independent checks with calibration bands.

* Samples are on-support for α ≤ 0.7 at every location of all three models (kernel score above held-out real states,
  tangent residual and nearest-real distance well inside the noise bands, coverage of the real cloud 0.97–0.99);
  α = 1 is marginal, α = 1.5 is off-support. Results are stable over seeds, latent dimension and prefixes.
* Pushed through the real model, the samples' images are scored like images of real held-out states by
  independently fitted destination manifolds.
* The sampler is uniform *within the tube around the observed states* (anchors equalised, clearly not the empirical
  token density), not in the manifold's own volume; the per-anchor rule that would fill sparse regions collapses.
* The fit is limited by the latent dimension, not by the network or the optimiser (63-fit capacity study); a VAE
  would be worse for this purpose.
* Two robustness rules matter: drop degenerate and isolated states first, and use the global support radius.
* Retracted negative: the TinyLlama embedding matrix looked "not chartable" (6 spacings) only because five hub
  tokens with tiny spacing dominate the per-anchor unit; in raw or global-median units it behaves like the small
  models (details in the last section). The library now detects such hub-dominated clouds.
* To the model itself, samples behave like blends of their nearest real tokens: their next-token distributions are
  closer to the nearest token's than real tokens are to each other (JS 0.21 vs 0.46 at α = 0.3; random-token baseline
  0.57), with top-1 agreement 32 % (random 6 %); one-spacing noise loses this (JS 0.51, agreement 8 %). Layer by layer,
  the images stay nearer to real states than held-out real images do and between the images of their two anchors.

## Summary

**Setup.** `GlobalManifold(latent_dim=16)` fitted per residual location on "prefix + every real token" clouds
(SimpleStories-5M: 13 locations, 3 seeds, m ∈ {8, 16, 32}; SimpleStories-35M: 25 locations, 5 seeds; TinyLlama-1.1B:
8 locations with 2 seeds and all 45 locations with one seed, 31,885 tokens after dropping 111 untrained byte-fallback
rows and 1 massive-activation outlier; plus a second prefix for the 5M model). Samples: 2000 per setting from 16,000 candidates, default `radius="global"`.

**What holds everywhere (all models, seeds, prefixes).**

1. *On-support regime.* Samples score above held-out real states (kernel u ≥ 1.03 at every location, coverage of the
   real cloud 0.97–0.99) for α ≤ 0.7; at α = 1.0 they still sit above the half-spacing noise band (u 0.96 vs 0.91)
   with coverage 0.93; α = 1.5 is off-support (u 0.72–0.74, coverage 0.12). Seed spread of u at α = 0.3 is ≤ 0.013.
   The independent tangent-chart residual agrees: samples 0.55–0.64 spacings from the nearest local chart for α ≤ 0.7
   (held-out real 0.82, half-spacing noise 0.95, one-spacing noise 1.24). The conservative tangent-chart sampler used
   as a cross-check gives u 0.97 and nearest-real 0.27 at α = 0.3: on-support but, by construction, hugging the anchors.
2. *Latent dimension.* m = 8 / 16 / 32 give held-out reconstruction 0.78 / 0.74 / 0.67 spacings and u 1.16 / 1.11 /
   1.06 at α = 0.3; larger m fits better but the volume weights spread more (ESS 5.5k / 3.4k / 1.5k of 16k).
3. *Volume reweighting works and is validated exactly* on a synthetic sheet with a known volume element
   (`tests/test_manifold.py`): with per-anchor radii the samples reproduce the volume-uniform target; with the global
   radius they lie between the data density and the target at small α and converge to the target as α grows.
4. *Global radius is the robust default.* With per-anchor radii (`radius="local"`) the union of balls is dominated
   by the sparsest anchors (ball volume ∝ ρ^m): on SimpleStories-5M `L5.ffn` the anchor participation ratio drops
   from 3546 to ~100 and coverage from 0.995 to 0.96; on TinyLlama a single massive-activation state took 100 % of
   the anchor mass and 99 % of the importance weight (ESS = 1), collapsing every sample onto it. Dropping isolated
   states (`outlier_mask`) and using the global radius removes the failure.
5. *Propagation.* Embedding samples pushed through the real model (`embed → L0.attn / L0.ffn / mid / last`) are
   scored by independently fitted destination manifolds like images of real held-out states (u within ±0.03 of
   the real images at every destination for α ≤ 0.5, ±0.2 at the first sub-layer where the map is near-identity
   and the samples' own overshoot carries over). Real states displaced by one spacing before propagation are
   healed by the network: u 0.71 after one sub-layer, ≥ 0.99 after six or more.
6. *Loss ablation* (5M, `embed` and `L5.ffn`, α = 0.3). Removing the local-distance term lowers u by 0.04–0.07 and
   raises the samples' reconstruction gap by 25–40 %; 3× longer training overfits (train recon 0.51 / 0.40 vs 0.67 /
   0.52, held-out recon worse, sample ESS 2.0k / 2.1k vs 4.9k / 6.8k, reconstruction gap ×3). Hidden width and the
   curvature term are neutral.

7. *Real-occurrence clouds* (20k random token occurrences from 2,500 stories, split by story; all contextual locations
   of SimpleStories-5M and 35M; §6 of the tables). Lower and token-clustered dimension (TwoNN 5–12), harder fits (held-out
   reconstruction 0.76–1.17 spacings vs 0.70–0.87 for prefix clouds) and softer calibration bands (one-spacing noise
   scores u 0.86 instead of 0.69). Samples nevertheless stay on-support at every location of both models for α ≤ 1 (u ≥ 1.01, above
   the 0.5× band), and their FFN images are scored by the independent destination fits like real held-out images
   (u 1.01–1.03 vs 1.00). Coverage is lower (0.65 at α = 0.3) only because 2000 samples are spread over 18k states.

8. *Decoder capacity study* (§7 of the tables; 63 fits on `embed`, `L2.ffn`, `L5.ffn`). At m = 16 no architecture or
   training variant changes the held-out reconstruction by more than 0.006 spacings: widths from (256, 128) to
   (2048, 1024), one to three hidden layers, learning rates 1e-3–5e-3, batches 128–2048, λ_geom 0.03–0.3 all give
   0.78–0.80 (`embed`), 0.72–0.74 (`L2.ffn`), 0.70–0.72 (`L5.ffn`); the two largest nets are worse (the 3.7M-parameter
   three-layer net under-fits in 300 epochs), and 600 epochs only lowers the training error (best validation epoch
   ≈ 300). The fit is therefore neither capacity- nor optimisation-limited. The one lever is the latent dimension:
   held-out reconstruction improves smoothly and without plateau from m = 8 to 64 (0.83 → 0.68, 0.77 → 0.57, 0.77 →
   0.49), which says the clouds have a long spectral tail rather than a sharp intrinsic dimension (local PCA needs
   21–27 components for 90–95 % of the neighbourhood variance). Larger m costs sampling efficiency: ESS at α = 0.3
   drops from 4.9k (m = 8) to 25–1200 (m = 64) and coverage from 0.99 to 0.81–0.90, while u moves from 1.15 toward
   1.03 (samples closer to the real states' own score). m = 16–24 is the practical compromise; m = 32 with more
   candidates if tighter reconstruction matters more than efficiency.

9. *Why not a VAE* (`experiments/vae_check.py`, `results/vae_check.json`; same architecture with a Gaussian encoder
   and β·KL, `embed` and `L5.ffn`). With β ≤ 1e-3 the posterior collapses to a point (σ ≈ 0.01–0.05), reconstruction
   equals ours, and decoding prior samples z ~ N(0, I) fails (u 0.05–0.16, coverage ≤ 0.15) because the aggregate
   posterior is nowhere near the prior; our Ω sampler on the posterior means works exactly as before. With β = 1e-2 the
   prior sampler becomes usable (u 1.2–1.3, coverage 0.98) at the price of reconstruction (0.79 → 0.86 at `embed`,
   0.71 → 0.80 at `L5.ffn`), of latent dimension (9 of 16 units stay active at `embed`) and of our sampler's ESS
   (415 / 126: the KL fights the local-distance term). The prior sampler also targets the aggregate-posterior (≈ data)
   density, not manifold volume, and has no notion of "near the observed support". Kept: nothing; noted as useful
   by-products: the number of active units as a data-driven m estimate, and the posterior σ as a per-point latent scale.

10. *What "volume-uniform" means on the real embedding cloud* (3940 SimpleStories-5M tokens, m = 16, 40k samples).
   Rank correlation between samples-per-real-anchor and the anchor's cell volume (∝ r_i^m): default sampler
   +0.005 (α = 0.5) / −0.004 (α = 1), data-density sampler −0.275, per-anchor-radius sampler +0.015 with 90 % of
   its samples on 1 % of the anchors and off-support (u 0.57). So the default sampler is uniform over the
   uniform-thickness tube Ω around the observed states (anchors equalised; clearly not the data density), but not
   uniform in the manifold's own volume, which would need the large cells of sparse regions to be filled — exactly
   what the per-anchor rule attempts and fails at. On the synthetic sheet the same effect appears as the global
   radius sitting between the data density and the exact volume target at small α.

11. *How the samples look to the model* (§8 of the tables, `experiments/downstream.py`). With an embedding sample at
   the last position of the prefix and the full 5M model run: the sample's next-token distribution is closer to that of
   its nearest real token (JS 0.21 at α = 0.3, 0.27 at 0.7, 0.37 at 1.5) than real tokens are to their own nearest
   neighbour (0.46) and far from a random token's (0.57); top-1 predictions agree with the nearest token's in 32 % of
   the samples (real-to-neighbour 19 %, random 6 %); entropy is slightly higher (3.4 vs 2.9 nats), as expected for a
   blend. Real tokens displaced by one spacing lose the neighbour relation (JS 0.51, top-1 agreement 8 %, entropy 3.9).
   Along the network the images of α ≤ 0.7 samples stay closer to real states than held-out real images do at every
   location (0.64–0.91 vs 0.89–0.96 spacings), keep u ≈ 1, and sit 0.5–0.9 spacings from the segment joining the
   images of their two nearest anchors (real 0.8–0.95, noise 1.1–1.3): they remain interpolations of their neighbours.

**What does not hold, and what we actually tried.** The one cloud that did not behave was the *embedding matrix* of
TinyLlama-1.1B (31,885 rows after dropping 111 shared byte-fallback rows and 1 massive-activation token; D = 2048).
What was run: `GlobalManifold` with hidden (512, 256), 150 epochs, m = 16 (two seeds) and m = 64 (one seed), 90/10
split by token, the same α sweep, plus the eight-location and all-45-location sweeps of the same model. What we saw
first: held-out reconstruction "6 spacings", samples "2.6 spacings from any real row", TwoNN ≈ 234 — and we called
the cloud not chartable. What was actually going on: five hub tokens (tight clusters of foreign-script junk tokens
such as `Архівовано`, `Webachiv`, `IABot`, spacing ≈ 0.1) are the nearest neighbour of 74 % of all held-out rows,
so "divide by the nearest anchor's spacing" divides by 0.1 for most points. In raw units the decoder reconstructs
held-out rows at 0.656 and training rows at 0.646, both ≈ the cloud's nearest-neighbour distance (0.678), the same
regime as the SimpleStories embeddings; in units of the global median spacing held-out real rows sit at 0.99
(recon 0.96), α = 0.3 samples at 0.46 (recon 0.09) and one-neighbour-distance noise at 1.40 (recon 1.38). The
library now detects hub-dominated clouds (median spacing of the anchors that points actually land on < half the
global median) and reports in global units; the TinyLlama tables were regenerated in those units. What remains
true: this cloud is high-dimensional (TwoNN 234 vs 19–24 for SimpleStories, MLE 32) and its coverage by 2000 samples
is lower (0.60 vs ≥ 0.97 elsewhere); with the corrected unit all four validators behave normally there (held-out real
0.99 / 0.95 / 1.00 / 0.98 on nearest / recon / u / tangent, one-spacing noise 1.40 / 1.39 / 0.68 / 1.40, samples at
α = 0.3 0.44 / 0.09 / 1.35 / 0.43). Not tried: m > 64, more than 150 epochs, frequent-token subsets, other preprocessing.
Use `intrinsic_dimension`, the held-out reconstruction and the reported hub share before trusting samples on a new cloud.

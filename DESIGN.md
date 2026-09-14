# gmanifold — design notes, experiments, and answers to the questions you will have

This is the long-form companion to `README.md` (how to use it) and `results/results.pdf` (the numbers). It records
what the library does, why each choice was made, every experiment that was run on the way, what went wrong, and
what the results do and do not support. Numbers refer to the runs in `results/` unless stated otherwise.

## 1. Purpose and scope

Given a finite set of real states X ⊂ R^D at one representation location of a Transformer (embedding rows, or
residual states of real token occurrences), learn a smooth parameterisation G: R^m → R^D of the set they lie on,
sample approximately uniformly (with respect to volume on the learned surface) *near the observed support*, and
optionally push the samples through a real Transformer map. The manifold code never sees the Transformer; the
optional `transformer.py` helpers only collect states and build `map_fn`s. Injectivity / LLR / EVT analyses,
layer-by-layer verification campaigns and reports live in the research version (`../manifold_sampler`), not here.

## 2. Implementation choices and their reasons

**Data preprocessing.** One centre and one global RMS scale (`X_n = (X − mean) / rms`). A similarity keeps the
Euclidean geometry intact; per-dimension whitening would distort distances and the "spacing" unit below.

**Two filters before fitting** (`degenerate_mask`, `outlier_mask`). Near-duplicate rows (nearest-neighbour distance
< 5 % of the median; e.g. SimpleStories' ~150 never-trained symbol tokens whose embeddings sit 0.005 apart, or
TinyLlama's 111 byte-fallback tokens sharing one row) make the local spacing zero and every normalised diagnostic
undefined. Isolated rows (nearest-neighbour distance > 10× the median; e.g. TinyLlama's massive-activation state of
norm 190 vs median 3–9 in deep layers) absorb all the volume weight of the sampler (ESS → 1). Both are dropped as
tokens, consistently across locations, and counted. Removing the SimpleStories cluster changes dimension estimates
by ≤ 0.4 (research-version verification).

**Encoder/decoder.** MLPs D → 512 → 256 → m and m → 256 → 512 → D with SiLU. SiLU because the decoder is
differentiated (Jacobians for the volume element); a shallow net because the capacity study (§4.6) shows nothing
larger helps. Hidden sizes are configurable.

**Loss.** L = recon + λ_geom·geom + λ_curv·curv with λ_geom = 0.1, λ_curv = 1e-3.
* recon = mean ‖G(E(x)) − x‖² (normalised units).
* geom = mean over the K = 32 Euclidean neighbours of (‖z_i − z_j‖ − ‖x_i − x_j‖)², divided by mean ‖x_i − x_j‖².
  This ties latent distances to ambient distances so that latent balls of radius α·ρ correspond to ambient
  neighbourhoods and α has a meaning. Without it the latent geometry is arbitrary and the sampler's support
  collapses (research version: support 0.24 vs 1.00) although reconstruction is unchanged.
* curv = second difference ‖G(z+δ) − 2G(z) + G(z−δ)‖²/‖δ‖² along random latent directions of length half the mean
  latent neighbour distance; a weak smoothness prior, measured to be neutral (§4.6) and kept because it is cheap.
* No KL / VAE prior (§4.7), no tangent-alignment term (the research version has one; it needs K_tan-neighbour local
  PCA at every step and was neutral for sampling).

**Optimiser.** Adam, one-cycle schedule with peak 2e-3 and 5 % warm-up, batch 512, 300 epochs. The capacity study
shows lr 1e-3–5e-3, batch 128–2048 and epochs 100–600 change held-out reconstruction by ≤ 0.006 spacings; the best
validation epoch is ≈ 300 and 600–900 epochs overfit (training error keeps falling, sample ESS halves).

**Latent support and sampler** (`sample`). Codes z_i = E(x_i); ρ_i = distance to the 8-th latent neighbour;
Ω = ∪ B(z_i, R_i). Anchor i ~ P(i) ∝ R_i^m, z uniform in the ball (direction ~ normalised Gaussian, radius R·s^{1/m}),
importance weight w = √det(J_GᵀJ_G) / q(z) with the exact mixture density q(z) = Σ_{j: z∈B_j} P(j)/R_j^m, multinomial
resampling of n from 8n candidates, x = G(z). `info` returns ESS, ball multiplicity, anchor participation ratio.
* `radius="global"` (default): R_i = α·median(ρ). `radius="local"`: R_i = α·ρ_i (the original spec). The local rule
  makes the union of balls dominated by the sparsest anchors because ball volume ∝ ρ^m: on SimpleStories-5M `L5.ffn`
  the anchor participation ratio falls from 3546 to ~100; on TinyLlama one outlier took 100 % of the anchor mass and
  99 % of the importance weight (ESS = 1) and every sample collapsed onto it. The global rule defines Ω as a
  uniform-thickness tube around the data; the sampler is then uniform *within the tube* (§4.9), robust, and
  on-support everywhere we tested.
* Volume reweighting is verified exactly on a synthetic 2-D sheet in R^64 with a known volume element
  (`tests/test_manifold.py::test_volume_uniform_sampling`): with local radii the samples reproduce E_volume[vol] to
  within the test tolerance; with the global radius they sit between the data density and the volume target at
  α = 0.5 and reach it at α = 1; the un-reweighted variant reproduces the data density.
* A warning is printed when ESS < 1 % of the candidates (the outlier signature).

**Validators** (never used for sampling).
* `KernelScore`: Guidotti's kernel signature (arXiv:2404.00427), u(x) = (1/N)Σ Λ_j K_σ(x, x_j) with
  (M + Nλ I)Λ = N·1 solved in fp64 on ≤ 8192 states, σ = median 32-NN distance, λ = 1e-6; u = 1 on fitted states,
  decays away from the cloud. The most sensitive validator below one spacing (AUC 0.93 vs 0.84 for the decoder
  error at half a spacing, research-version verification); insensitive to λ ≤ 1e-4 and to σ within 0.5–2×.
* `TangentCharts`: local PCA bases from K = max(4m, 32) > m neighbours (a centred K-neighbourhood has rank ≤ K−1);
  `.residual(Y)` is the normal distance to the nearest anchor's chart in units of that anchor's spacing;
  `.sample(n, α)` is the conservative local sampler of the spec (anchor ∝ r^m, uniform intrinsic ball) used as a
  cross-check. Bases are computed lazily for the anchors queried (a full basis would be 15 GB at D = 2048, m = 64).
* `GlobalManifold.recon_error` (decoder off-manifold error), `nearest_real`, `coverage`, `jacobian_rank`.
* `support_report` prints medians of all of these for samples next to two calibration bands: held-out real states,
  and held-out states displaced by 0.5× and 1× their local spacing in a random direction. All verdicts in this
  repository are relative to these bands, never to absolute thresholds.

**Units.** "Spacing" is the distance from a real state to its 8-th nearest real neighbour at that location
(medians: 0.78 at the SimpleStories-5M embedding, 0.97 at `L0.ffn`, 2.65 at `L5.ffn`). Every distance-like number
in the results is divided by the spacing of the nearest real state. "Held-out reconstruction 0.7" means an unseen
real state projects onto the learned surface 0.7 inter-point distances away.

**Transformer helpers.** `locations` = `embed`, `Lk.attn` (residual entering `post_attention_layernorm`),
`Lk.ffn` (block output); hooks capture or overwrite the residual at a position (in place for `Lk.attn`, because the
residual add reuses that tensor). `make_map(model, src, dst, context)` returns F_{src→dst} under a fixed context
(tokenwise fast path for `Lk.attn → Lk.ffn`). Exactness is tested against `output_hidden_states` and against
hand-computed sub-layer maps (≤ 1e-5).

## 3. Data used

* Prefix + every real token: sequences `prefix_ids + [v]` for every non-special v, state at the last position.
  Prefix "Once upon a time, there was a little" (9 tokens; length-10 sequences), second prefix "The". Split 90/10 by
  token, seed 0. SimpleStories-5M/35M: 4094 real tokens minus the degenerate cluster (3940 / 3964 kept); TinyLlama:
  32000 minus 111 byte tokens minus 1 outlier (31,885), BOS prepended because its tokenizer expects one.
* Real occurrences (`experiments/stories.py`): 20,000 random token positions (8 per story, ≤ 256 tokens) from
  2,500 SimpleStories stories, split by story (2,000 held out), duplicates and outliers dropped from the training split.

## 4. Experiments (all in `results/`)

4.1 **Dimension and fit per location** (§1 of the tables). TwoNN ≈ 18–24 at the embedding, 10–15 deeper; MLE higher
and K-dependent; held-out reconstruction 0.70–0.89 spacings at m = 16, Jacobian rank = m everywhere.

4.2 **α sweep** (§2). Samples score above held-out real states (u ≥ 1.03) at every location for α ≤ 0.7, with
coverage 0.97–0.99; α = 1 marginal (u 0.96, above the half-spacing noise band 0.91); α = 1.5 off-support. The
tangent residual agrees (samples 0.55–0.64 vs real 0.82, noise 0.95 / 1.24). Seed spread ≤ 0.013.

4.3 **Latent dimension and seeds** (§3). m = 8/16/32: reconstruction 0.78/0.74/0.67, u 1.16/1.11/1.06, ESS
5.5k/3.4k/1.5k. Five seeds on 35M and a second prefix on 5M reproduce everything.

4.4 **Sampler variants** (§4). Global vs local radius, anchor power, reweighting on/off, and the tangent-chart
cross-check (u 0.97, nearest 0.27, hugging anchors by construction).

4.5 **Propagation** (§5). Embedding samples pushed through `embed → {L0.attn, L0.ffn, mid, last}` score like images
of real held-out states under independently fitted destination manifolds (never trained on the images); real
states displaced by one spacing before propagation are healed (u 0.71 after one sub-layer, ≥ 0.99 after six).

4.6 **Capacity study** (§7, `experiments/capacity.py`, 63 fits). At m = 16: widths (256,128)…(2048,1024), one to
three hidden layers, lr, batch, λ_geom, 100–600 epochs all give held-out reconstruction within ±0.006 of the default
(0.79 / 0.73 / 0.70 at `embed` / `L2.ffn` / `L5.ffn`); the two largest nets are worse. Only m matters: 0.83 → 0.68,
0.77 → 0.57, 0.77 → 0.49 from m = 8 to 64 without plateau, at the price of ESS (4.9k → 25–1200) and coverage
(0.99 → 0.81–0.90). The clouds have a long spectral tail, not a sharp intrinsic dimension.

4.7 **VAE check** (`experiments/vae_check.py`). Same architecture with a Gaussian encoder and β·KL. β ≤ 1e-3: posterior
collapses, reconstruction equals ours, prior sampling fails (u 0.05–0.16, coverage ≤ 0.15). β = 1e-2: prior sampling
works (u 1.2, coverage 0.98) but reconstruction worsens (0.79 → 0.86, 0.71 → 0.80), 9/16 units stay active, and our
sampler's ESS collapses (415 / 126). The prior sampler targets the aggregate-posterior (≈ data) density, not volume,
and has no notion of "near the support". Not adopted.

4.8 **Real-occurrence clouds** (§6). Token-clustered, lower dimension (TwoNN 5–12), harder fits (0.76–1.27 spacings),
softer bands (1× noise u 0.86); samples on-support for α ≤ 1 at every location of both models; FFN images scored like
real images.

4.9 **Uniformity on the real embedding** (summary item 10). Rank correlation between samples-per-anchor and the
anchor's cell volume: default sampler ≈ 0 (anchors equalised), data-density sampler −0.28, per-anchor-radius sampler
+0.015 with 90 % of samples on 1 % of anchors and off-support. The default sampler is uniform within the tube
around the observed states, not in the manifold's own volume (sparse regions' large cells are not filled).

4.10 **Scale.** SimpleStories-35M (25 locations) and TinyLlama-1.1B (45 locations, D = 2048) reproduce all of the
above for contextual clouds. The TinyLlama *embedding* cloud first looked unfittable and was reported as "not
chartable by one decoder"; that was a units artifact, see §6 for the full account of what was tried and found.

## 5. Things that went wrong and were fixed

* Candidates on a latent-ball boundary failed the membership test by rounding → −∞ proposal density → NaN weights.
  Fixed with a 1e-5 relative tolerance.
* Per-anchor radii + a massive-activation outlier → ESS = 1, all samples on one point (TinyLlama). Fixed by the
  global radius default and `outlier_mask`.
* Near-duplicate embedding rows → zero spacing, NaN diagnostics. Fixed by `degenerate_mask`.
* The TinyLlama embedding "6 spacings" result: five hub tokens with spacing ≈ 0.1 are the nearest neighbour of 74 %
  of held-out rows, so per-anchor units inflated every distance 6–9×. Fixed by detecting hub-dominated clouds at fit
  time and switching to the global median spacing (`spacing_unit`, `hub_share`).
* Storing tangent bases for many locations (3.8 GB each at D = 2048) or one basis at m = 64 (15 GB) → GPU OOM.
  Fixed by computing bases lazily per queried anchor.
* Collecting 45 locations of 32k states through Python lists doubled 11.7 GB → host OOM. Fixed by preallocation.
* The synthetic torus was a wrong test case (not coverable by one chart); replaced by a curved sheet.

## 6. What the results do not support (read before quoting)

* Uniformity in the manifold's own volume beyond the tube (§4.9); "uniform" here means uniform within
  α × median spacing of the observed states.
* Any statement about regions with no observed states nearby; α ≥ 1.5 is off-support and nothing is claimed there.
* Sub-spacing fidelity: the learned surface passes ~0.7 spacings from unseen real states; the true set may be
  rougher or have structure below that scale.
* Validators as proof: they encode the same interpolation assumption (midpoints between neighbouring real states
  are scored as on-manifold by all of them; research-version hard-negative experiment).
* Per-anchor spacing units on hub-dominated clouds. TinyLlama-1.1B (31,885 rows after dropping 111 shared byte-fallback rows and 1 massive-activation token; D = 2048). What was run: `GlobalManifold` with hidden (512, 256), 150 epochs, m = 16 (two seeds) and m = 64 (one seed), 90/10 split by token, the same α sweep, plus the eight-location and all-45-location sweeps of the same model. What we saw first: held-out reconstruction "6 spacings", samples "2.6 spacings from any real row", TwoNN ≈ 234 — and we called the cloud not chartable. What was actually going on: five hub tokens (tight clusters of foreign-script junk tokens such as `Архівовано`, `Webachiv`, `IABot`, spacing ≈ 0.1) are the nearest neighbour of 74 % of all held-out rows, so "divide by the nearest anchor's spacing" divides by 0.1 for most points. In raw units the decoder reconstructs held-out rows at 0.656 and training rows at 0.646, both ≈ the cloud's nearest-neighbour distance (0.678), the same regime as the SimpleStories embeddings; in units of the global median spacing held-out real rows sit at 0.99 (recon 0.96), α = 0.3 samples at 0.46 (recon 0.09) and one-neighbour-distance noise at 1.40 (recon 1.38). The library now detects hub-dominated clouds (median spacing of the anchors that points actually land on < half the global median) and reports in global units; the TinyLlama tables were regenerated in those units. What remains true: this cloud is high-dimensional (TwoNN 234 vs 19–24 for SimpleStories, MLE 32) and the kernel score cannot separate it from one-neighbour-distance noise (u 0.98 vs 1.00), so the sampling claims there rest on nearest-real and reconstruction only. Not tried: m > 64, more than 150 epochs, frequent-token subsets, other preprocessing.
* The removed tokens: never-trained symbol tokens and massive-activation states are real vocabulary entries and real
  states; they are excluded from the fits by design and are not represented by the samples.

## 7. Reproduce

`pytest tests` (2 minutes); `bash experiments/run_all.sh` (≈ 2 GPU-hours on one RTX 5090: tests, 5M with m ∈ {8,16,32}
× 3 seeds and ablations, 35M, TinyLlama, extra seeds, second prefix, all TinyLlama locations, story clouds, report);
`python experiments/capacity.py`; `python experiments/vae_check.py`. Environment: torch 2.11 + CUDA 12.8,
transformers 5.17, one RTX 5090.

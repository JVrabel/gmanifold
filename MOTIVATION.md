# Why we sample the manifold: the injectivity programme

This note places `gmanifold` in the larger project. The library itself (fitting, sampling, validation) is documented in
`README.md` and `DESIGN.md`; this is the *why*.

## The claim we want to test

A Transformer block maps the residual stream at one location to the residual stream at the next,
F: R^D → R^D. Real states do not fill R^D; at every location they sit near a low-dimensional set, the
**information manifold** M. Our hypothesis:

> The layer maps are injective on M and preserve its topology. They change the geometry (stretch, compress,
> bend, re-scale locally) but never collapse distinct states onto one point or tear / glue parts of the manifold.

If this holds, information is never destroyed inside the network, only re-encoded; if it fails, the failure
should be localised (specific regions of M where the map folds) and we want to find those points.

## The quantity: a lower Lipschitz bound on the manifold

Injectivity with a margin is a lower Lipschitz constant restricted to M:

    L_lower = inf_{x ≠ y ∈ M} ||F(x) − F(y)|| / ||x − y||        (bi-Lipschitz ratio: L_upper / L_lower)

Globally this infimum is dominated by the worst place on the manifold, so instead of one number we look at the
**local** version: for x on M and a small displacement along the manifold (a nearby sampled point, or a
tangent direction u, giving σ_min(J_F(x) U_x) with U_x a tangent basis), the local contraction ratio r(x). Over
the whole manifold r is a random variable; its *distribution* tells us how the geometry is being reshaped, and
its **left tail** is where injectivity would break: r → 0 means two nearby states become indistinguishable.

The tail is the object of interest, and a finite sample of r never reaches the true minimum. **Extreme value
theory** is the tool for that: fit the left tail (peaks below a low threshold, generalised Pareto / Weibull-type
endpoint estimation) and estimate the lower endpoint of the distribution of r. An endpoint bounded away from zero
is evidence for injectivity with a margin; an endpoint at (or consistent with) zero, with the location of the
offending states, is the "breaking point" of injectivity. Doing this per location and per model gives the picture
of where and how much a Transformer squeezes its manifold.

## Why a manifold sampler is needed for that

The real states are a finite cloud X (one point per token, per context, per position). Estimating a tail from X
alone has three problems:

1. **It probes only where the data happen to be.** Frequent tokens and dense clusters are oversampled, sparse and
   highly stretched regions (exactly where contraction is most likely to be extreme) are underrepresented. A tail
   estimate from X is a tail of the *token distribution*, not of the manifold.
2. **Pairs must be on the manifold.** The ratio r is only meaningful for displacements along M. Random Gaussian
   perturbations leave the support; F contracts off-manifold noise for perfectly healthy reasons, so a tail measured
   on such inputs says nothing about injectivity of the layer on the data.
3. **EVT needs many independent tail samples.** The real cloud gives at most N values of r, most of them in the bulk.

So we need new points that are (a) on the learned manifold, (b) between the observed states rather than on them,
and (c) spread uniformly with respect to **manifold volume**, so that every region of M contributes to the tail in
proportion to its size and not to its token frequency. That is what `gmanifold` provides:

* `GlobalManifold.fit` learns a smooth chart G: R^m → R^D of the observed states with a geometry-preserving loss.
* `sample` draws latent candidates in a tube around the observed codes and reweights them by the decoder's volume
  element √det(JᵀJ) divided by the proposal density (importance sampling with resampling), so the accepted points are
  ≈ uniform on the surface G(Ω). Without this step the samples would follow the data density and problem 1 returns.
* The validators (`nearest_real`, `recon_error`, `KernelScore`, `TangentCharts.residual`, calibrated against held-out
  real states and against real states pushed off the manifold by noise) certify that the samples satisfy (a); this is
  problem 2.
* `transformer.make_map` applies the *real* layer maps to the samples, and an **independent** fit at the destination
  checks that the images land on the destination manifold; this is the first, coarse form of the preservation claim
  (samples in → samples out, no images falling off the support).

## From samples to the tail

With volume-uniform samples x on M at location k and the real map F = F_{k→k'}:

1. Take a tangent basis at x (from the chart, U = J_G(z) orthonormalised, or from `TangentCharts`), or a small
   displacement along M (a second sample from the same latent ball).
2. Compute the local ratio r(x) = σ_min(J_F(x) U) (or the finite-difference version ||F(x) − F(x')|| / ||x − x'||),
   and the distortion σ_max / σ_min.
3. Collect r over thousands of samples, look at the empirical distribution and its left tail, and fit the extreme
   value model to estimate the lower endpoint and its confidence interval.
4. Repeat per location, per model size, per context; compare with the same statistic on real pairs and on off-manifold
   noise as controls.

## Practical consequences (why efficiency matters)

The programme needs many clouds: every layer, several positions, many contexts, all next-token candidates, and a fit
and sample for each. That is the reason for the batched fits (`fit_many`) and the shared neighbour searches (`nn=`):
the manifold fitting is the dominant cost per cloud, the Jacobians of the sampler are second, the neighbour searches
are third but were repeated up to five times per cloud.

## Caveats to keep in mind

* G parameterises the *observed* support, not the true manifold; the tube around the codes (α) sets how far we
  interpolate, and the tail estimate inherits whatever the chart does there.
* Hub-dominated and high-dimensional clouds (TinyLlama embeddings) are covered less well by a fixed number of samples,
  so the tail there is a lower bound on what we would see with more samples.
* Extreme value estimates are only as good as the sampler's coverage of the thin, stretched regions; that is exactly
  why the volume reweighting, and its ESS check, cannot be skipped.

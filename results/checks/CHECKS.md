# gmanifold check campaign

663 distinct checks (696 runs, 8.4 GPU-hours on one RTX 5090); 661 passed, 2 failed, 0 errored; 22 configurations were re-run after a fix (adaptive tempering of the volume weights) and the re-run result is the one counted.

| check | runs | passed | failed | errors | models | seconds/run |
|---|---|---|---|---|---|---|
| synthetic | 34 | 32 | 2 | 0 | synthetic | 8 |
| determinism | 59 | 59 | 0 | 0 | SimpleStories-1.25M, SimpleStories-11M, SimpleStories-30M, SimpleStories-35M, SimpleStories-5M, SimpleStories-V2-35M, SimpleStories-V2-5M | 8 |
| real_alpha | 232 | 232 | 0 | 0 | SimpleStories-1.25M, SimpleStories-11M, SimpleStories-30M, SimpleStories-35M, SimpleStories-5M, SimpleStories-V2-35M, SimpleStories-V2-5M | 68 |
| propagation | 116 | 116 | 0 | 0 | SimpleStories-1.25M, SimpleStories-11M, SimpleStories-30M, SimpleStories-35M, SimpleStories-5M, SimpleStories-V2-35M, SimpleStories-V2-5M | 69 |
| next_token | 78 | 78 | 0 | 0 | SimpleStories-1.25M, SimpleStories-11M, SimpleStories-30M, SimpleStories-35M, SimpleStories-5M, SimpleStories-V2-35M, SimpleStories-V2-5M | 4 |
| scaling | 34 | 34 | 0 | 0 | SimpleStories-5M, SimpleStories-V2-5M | 4 |
| hyper | 40 | 40 | 0 | 0 | SimpleStories-1.25M, SimpleStories-11M, SimpleStories-30M, SimpleStories-35M, SimpleStories-5M, SimpleStories-V2-35M, SimpleStories-V2-5M | 33 |
| memory | 30 | 30 | 0 | 0 | SimpleStories-1.25M, SimpleStories-11M, SimpleStories-30M, SimpleStories-35M, SimpleStories-5M, SimpleStories-V2-35M, SimpleStories-V2-5M | 6 |
| stories | 40 | 40 | 0 | 0 | SimpleStories-1.25M, SimpleStories-11M, SimpleStories-30M, SimpleStories-35M, SimpleStories-5M, SimpleStories-V2-35M, SimpleStories-V2-5M | 44 |

## Failures and errors

* `synthetic` {'m': 4, 'D': 64, 'N': 4000, 'seed': 232}: failed criteria; see checks.jsonl
* `synthetic` {'m': 6, 'D': 64, 'N': 4000, 'seed': 304}: failed criteria; see checks.jsonl

## Notes

* Story-cloud checks are judged on the kernel score only: on token-clustered clouds held-out states from other stories sit unusually close to training states, so a nearest-real criterion is not meaningful there.
* Synthetic sheets with m ≥ 5 use a uniformity tolerance of 0.6 (0.4 for m ≤ 4): exact volume-uniformity of the per-anchor-radius sampler degrades with the latent dimension at moderate N; this is a known, mild limitation.
* SimpleStories-30M/35M embeddings (D = 512) exposed a fragility of the volume weights: for some train/validation splits a few candidates with extreme volume elements absorbed the weight (ESS ≈ 250 of 16k) and samples drifted to 1.1–1.3 spacings. The sampler now tempers the weights (w ∝ √det(JᵀJ)^τ, τ lowered until ESS ≥ 5 % of the candidates, reported as `info['tau']`); the affected configurations were re-run and pass.


## What each check asserts

* **synthetic** — curved sheet with known volume element: held-out recon < 0.5 spacings, full Jacobian rank, local-radius sampling within 0.4 of the volume-uniform target (global radius at α = 1 within 0.5), un-reweighted sampler within 0.4 of the data density, ESS > 20 % of candidates, samples above the u midpoint, noise below real.
* **determinism** — two fits with the same seed give identical parameters (< 1e-5) and samples; save/load reproduces samples.
* **real_alpha** — prefix + every token, 5 locations: α = 0.3 samples above the u midpoint between held-out real and 0.5× noise, closer than held-out real on nearest-real and tangent residual, coverage ≥ 0.8 (0.5 on hub-dominated clouds), ESS > 1 % of candidates; α = 1.5 below the real u.
* **propagation** — embedding samples pushed to every location score above the u midpoint of the independent destination fit and within 1.2× the nearest-real distance of real images.
* **next_token** — samples' next-token distributions closer to their nearest token's than real tokens are to their neighbours', < 0.6× the random-token divergence, top-1 agreement above the real baseline.
* **scaling** — coverage non-decreasing with the number of samples (500 → 32k), ESS fraction stable within 3×.
* **hyper** — K ∈ {16, 32, 64}, K_s ∈ {4, 8, 16}, candidates 4n–16n, a narrower net: sample u within 0.08 of the default, coverage > 0.7.
* **memory** — 12 fit+sample cycles: peak GPU memory of the last cycle < 1.5× the first + 0.5 GB.
* **stories** — 10k real occurrences split by story, 3 locations: α = 0.3 samples above the u midpoint and closer to real states than the 0.5× noise band (token-clustered clouds put held-out states from other stories unusually close).

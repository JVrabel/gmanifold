# gmanifold check campaign

674 checks in 8.0 h on one RTX 5090; 597 passed, 77 failed, 0 errored.

| check | runs | passed | failed | errors | models | seconds/run |
|---|---|---|---|---|---|---|
| synthetic | 34 | 24 | 10 | 0 | synthetic | 8 |
| determinism | 59 | 59 | 0 | 0 | SimpleStories-1.25M, SimpleStories-11M, SimpleStories-30M, SimpleStories-35M, SimpleStories-5M, SimpleStories-V2-35M, SimpleStories-V2-5M | 8 |
| real_alpha | 233 | 215 | 18 | 0 | SimpleStories-1.25M, SimpleStories-11M, SimpleStories-30M, SimpleStories-35M, SimpleStories-5M, SimpleStories-V2-35M, SimpleStories-V2-5M | 68 |
| propagation | 117 | 113 | 4 | 0 | SimpleStories-1.25M, SimpleStories-11M, SimpleStories-30M, SimpleStories-35M, SimpleStories-5M, SimpleStories-V2-35M, SimpleStories-V2-5M | 69 |
| next_token | 79 | 79 | 0 | 0 | SimpleStories-1.25M, SimpleStories-11M, SimpleStories-30M, SimpleStories-35M, SimpleStories-5M, SimpleStories-V2-35M, SimpleStories-V2-5M | 4 |
| scaling | 34 | 34 | 0 | 0 | SimpleStories-5M, SimpleStories-V2-5M | 4 |
| hyper | 40 | 40 | 0 | 0 | SimpleStories-1.25M, SimpleStories-11M, SimpleStories-30M, SimpleStories-35M, SimpleStories-5M, SimpleStories-V2-35M, SimpleStories-V2-5M | 33 |
| memory | 30 | 30 | 0 | 0 | SimpleStories-1.25M, SimpleStories-11M, SimpleStories-30M, SimpleStories-35M, SimpleStories-5M, SimpleStories-V2-35M, SimpleStories-V2-5M | 6 |
| stories | 48 | 3 | 45 | 0 | SimpleStories-1.25M, SimpleStories-11M, SimpleStories-30M, SimpleStories-35M, SimpleStories-5M, SimpleStories-V2-35M, SimpleStories-V2-5M | 44 |

## Failures and errors

* `stories` {'name': 'SimpleStories/SimpleStories-5M', 'seed': 0}: failed criteria; see checks.jsonl
* `synthetic` {'m': 6, 'D': 512, 'N': 15000, 'seed': 8}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-30M', 'seed': 0}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-5M', 'seed': 0}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-V2-35M', 'seed': 0}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-35M', 'seed': 0}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-11M', 'seed': 0}: failed criteria; see checks.jsonl
* `synthetic` {'m': 6, 'D': 64, 'N': 4000, 'seed': 124}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-1.25M', 'seed': 0}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-V2-5M', 'seed': 0}: failed criteria; see checks.jsonl
* `synthetic` {'m': 5, 'D': 64, 'N': 6000, 'seed': 133}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-30M', 'seed': 0}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-5M', 'seed': 1}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-V2-35M', 'seed': 1}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-35M', 'seed': 1}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-11M', 'seed': 1}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-1.25M', 'seed': 1}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-V2-5M', 'seed': 1}: failed criteria; see checks.jsonl
* `synthetic` {'m': 6, 'D': 64, 'N': 6000, 'seed': 169}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-30M', 'seed': 1}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-5M', 'seed': 1}: failed criteria; see checks.jsonl
* `synthetic` {'m': 5, 'D': 64, 'N': 8000, 'seed': 178}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-V2-35M', 'seed': 1}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-35M', 'seed': 2}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-11M', 'seed': 2}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-1.25M', 'seed': 2}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-V2-5M', 'seed': 2}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-30M', 'seed': 2}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-5M', 'seed': 2}: failed criteria; see checks.jsonl
* `synthetic` {'m': 6, 'D': 64, 'N': 8000, 'seed': 214}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-V2-35M', 'seed': 2}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-35M', 'seed': 2}: failed criteria; see checks.jsonl
* `real_alpha` {'name': 'SimpleStories/SimpleStories-30M', 'prefix': 'Once upon a time, there was a little', 'seed': 3}: failed criteria; see checks.jsonl
* `propagation` {'name': 'SimpleStories/SimpleStories-30M', 'prefix': 'Once upon a time, there was a little', 'seed': 3}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-1.25M', 'seed': 3}: failed criteria; see checks.jsonl
* `synthetic` {'m': 4, 'D': 64, 'N': 4000, 'seed': 232}: failed criteria; see checks.jsonl
* `real_alpha` {'name': 'SimpleStories/SimpleStories-30M', 'prefix': 'The', 'seed': 3}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-V2-5M', 'seed': 3}: failed criteria; see checks.jsonl
* `real_alpha` {'name': 'SimpleStories/SimpleStories-30M', 'prefix': 'One day, a big dog named Max saw a', 'seed': 3}: failed criteria; see checks.jsonl
* `propagation` {'name': 'SimpleStories/SimpleStories-30M', 'prefix': 'One day, a big dog named Max saw a', 'seed': 3}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-30M', 'seed': 3}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-5M', 'seed': 3}: failed criteria; see checks.jsonl
* `real_alpha` {'name': 'SimpleStories/SimpleStories-30M', 'prefix': 'Lily and her mom went to the park. It was a sunny day and they saw a', 'seed': 3}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-V2-35M', 'seed': 3}: failed criteria; see checks.jsonl
* `real_alpha` {'name': 'SimpleStories/SimpleStories-30M', 'prefix': 'In the morning, the', 'seed': 3}: failed criteria; see checks.jsonl
* `propagation` {'name': 'SimpleStories/SimpleStories-30M', 'prefix': 'In the morning, the', 'seed': 3}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-35M', 'seed': 3}: failed criteria; see checks.jsonl
* `real_alpha` {'name': 'SimpleStories/SimpleStories-30M', 'prefix': 'She opened the box and found a', 'seed': 3}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-1.25M', 'seed': 3}: failed criteria; see checks.jsonl
* `synthetic` {'m': 5, 'D': 64, 'N': 4000, 'seed': 268}: failed criteria; see checks.jsonl
* `real_alpha` {'name': 'SimpleStories/SimpleStories-30M', 'prefix': 'Once upon a time, there was a little', 'seed': 4}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-V2-5M', 'seed': 4}: failed criteria; see checks.jsonl
* `real_alpha` {'name': 'SimpleStories/SimpleStories-30M', 'prefix': 'The', 'seed': 4}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-30M', 'seed': 4}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-5M', 'seed': 4}: failed criteria; see checks.jsonl
* `real_alpha` {'name': 'SimpleStories/SimpleStories-30M', 'prefix': 'One day, a big dog named Max saw a', 'seed': 4}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-V2-35M', 'seed': 4}: failed criteria; see checks.jsonl
* `real_alpha` {'name': 'SimpleStories/SimpleStories-30M', 'prefix': 'Lily and her mom went to the park. It was a sunny day and they saw a', 'seed': 4}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-35M', 'seed': 4}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-11M', 'seed': 4}: failed criteria; see checks.jsonl
* `real_alpha` {'name': 'SimpleStories/SimpleStories-30M', 'prefix': 'In the morning, the', 'seed': 4}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-1.25M', 'seed': 4}: failed criteria; see checks.jsonl
* `synthetic` {'m': 6, 'D': 64, 'N': 4000, 'seed': 304}: failed criteria; see checks.jsonl
* `real_alpha` {'name': 'SimpleStories/SimpleStories-30M', 'prefix': 'She opened the box and found a', 'seed': 4}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-V2-5M', 'seed': 4}: failed criteria; see checks.jsonl
* `real_alpha` {'name': 'SimpleStories/SimpleStories-30M', 'prefix': 'Once upon a time, there was a little', 'seed': 5}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-30M', 'seed': 5}: failed criteria; see checks.jsonl
* `real_alpha` {'name': 'SimpleStories/SimpleStories-35M', 'prefix': 'Once upon a time, there was a little', 'seed': 5}: failed criteria; see checks.jsonl
* `synthetic` {'m': 5, 'D': 64, 'N': 6000, 'seed': 313}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-5M', 'seed': 5}: failed criteria; see checks.jsonl
* `real_alpha` {'name': 'SimpleStories/SimpleStories-30M', 'prefix': 'The', 'seed': 5}: failed criteria; see checks.jsonl
* `real_alpha` {'name': 'SimpleStories/SimpleStories-35M', 'prefix': 'The', 'seed': 5}: failed criteria; see checks.jsonl
* `propagation` {'name': 'SimpleStories/SimpleStories-35M', 'prefix': 'The', 'seed': 5}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-V2-35M', 'seed': 5}: failed criteria; see checks.jsonl
* `real_alpha` {'name': 'SimpleStories/SimpleStories-30M', 'prefix': 'One day, a big dog named Max saw a', 'seed': 5}: failed criteria; see checks.jsonl
* `real_alpha` {'name': 'SimpleStories/SimpleStories-35M', 'prefix': 'One day, a big dog named Max saw a', 'seed': 5}: failed criteria; see checks.jsonl
* `stories` {'name': 'SimpleStories/SimpleStories-35M', 'seed': 5}: failed criteria; see checks.jsonl

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

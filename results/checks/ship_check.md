# Ship check: importance-weight checker on real models

`experiments/ship_check.py`, one RTX 5090, ~4 min. Latent dim 16, 300 epochs, α = 0.3, 2000 samples from 16k candidates; the checker floor is ESS ≥ 5 % of the candidates.

| model / location | split | u held-out | default: ESS % | warned | suggested τ | u | nearest | auto_tau: τ | ESS % | u | nearest | coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SimpleStories-35M embed | 0 | 0.99 | 5.2 | no | 1.00 | 1.17 | 0.82 | 1.00 | 5.2 | 1.17 | 0.82 | 0.99 |
| SimpleStories-35M embed | 1 | 0.99 | 2.4 | **yes** | 0.87 | 1.18 | 0.72 | 0.87 | 5.0 | 1.21 | 0.70 | 0.99 |
| SimpleStories-35M embed | 2 | 0.99 | 4.8 | **yes** | 0.99 | 1.20 | 0.68 | 0.99 | 5.0 | 1.20 | 0.68 | 0.98 |
| SimpleStories-35M embed | 3 | 0.99 | 2.7 | **yes** | 0.87 | 1.18 | 0.73 | 0.87 | 5.0 | 1.20 | 0.70 | 0.98 |
| SimpleStories-5M L3.ffn | 0 | 1.00 | 25.0 | no | 1.00 | 1.10 | 0.61 | 1.00 | 25.0 | 1.10 | 0.61 | 0.99 |
| SimpleStories-5M L3.ffn | 1 | 1.00 | 26.7 | no | 1.00 | 1.10 | 0.60 | 1.00 | 26.7 | 1.10 | 0.60 | 0.99 |

* The checker fires exactly where the ESS of the exact weights (τ = 1) is below the floor: 3 of 4 splits of the SimpleStories-35M embedding (ESS 2.4–4.8 %), never on SimpleStories-5M `L3.ffn` (ESS 25–27 %). The printed warning and `info['warning']` agree with `check_sampling`.
* `auto_tau=True` applies the suggested τ (0.87–0.99) and lands the ESS on the floor. The samples stay on-support in both modes (kernel score u above the held-out real states, nearest real state 0.6–0.8 spacings, coverage ≥ 0.98); on these splits the degeneracy is mild, so the untempered samples are not visibly worse, but the ESS says they are built from a few hundred effective candidates.
* Behaviour is deterministic given `seed` in both modes.

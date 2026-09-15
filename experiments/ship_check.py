"""Final sanity check of the importance-weight checker on real models (run before shipping).
1. SimpleStories-35M embedding (D = 512), the cloud where the volume weights were seen to degenerate for some splits:
   sample with the default (tau = 1) and with auto_tau=True over several train/validation splits; the checker must fire
   exactly where ESS < 5 % of the candidates, and auto_tau must restore the floor with on-support samples (u ~ held-out real).
2. SimpleStories-5M L3.ffn: healthy cloud, no warning expected, samples on-support."""
import io, contextlib, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gmanifold as gm
from gmanifold import transformer as tr

def cloud(name, loc):
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).cuda().eval(); tok = AutoTokenizer.from_pretrained(name)
    X, ids = tr.vocab_states(model, tok)
    if loc != "embed":
        prefix = tok("Once upon a time, there was a little", add_special_tokens=False)["input_ids"]
        seqs = torch.tensor([prefix + [v] for v in ids]); X = tr.collect_states(model, seqs, torch.full((len(ids),), len(prefix)), [loc])[loc].cuda()
    X = X[~gm.degenerate_mask(X) & ~gm.outlier_mask(X)]; del model; torch.cuda.empty_cache(); return X

out = []
for name, loc, seeds in (("SimpleStories/SimpleStories-35M", "embed", (0, 1, 2, 3)), ("SimpleStories/SimpleStories-5M", "L3.ffn", (0, 1))):
    X = cloud(name, loc)
    for seed in seeds:
        perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed)); tr_i, va_i = perm[:-400], perm[-400:]
        M = gm.GlobalManifold(latent_dim=16).fit(X[tr_i], epochs=300, X_val=X[va_i], seed=seed); k = gm.KernelScore(X[tr_i])
        row = dict(model=name.split("/")[1], loc=loc, seed=seed, u_heldout=float(k(X[va_i]).median()))
        for mode in ("default", "auto_tau"):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                x, info = M.sample(2000, alpha=0.3, seed=seed, auto_tau=(mode == "auto_tau"))
            ok, msg = gm.check_sampling(info, verbose=False)
            row[mode] = dict(ess_frac=info["ess"] / info["n_candidates"], tau=info["tau"], tau_suggested=info["tau_suggested"], warned=info["warning"] is not None,
                             printed_warning="WARNING" in buf.getvalue(), checker_ok=ok, u=float(k(x).median()), nearest=float(M.nearest_real(x).median()), coverage=M.coverage(x))
        out.append(row); print(json.dumps(row)); 
json.dump(out, open("results/checks/ship_check.json", "w"), indent=1)

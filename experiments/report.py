"""results/*/sweep.json -> results/results.md (+ figures) -> results/results.pdf. Sampling-relevant results only."""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#8a8985"]
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "font.size": 9, "legend.frameon": False, "figure.facecolor": "white"})


def f(v, p=3):
    return "–" if v is None else (f"{v:.0f}" if isinstance(v, float) and abs(v) >= 100 else f"{v:.{p}f}" if isinstance(v, float) else str(v))


def table(rows, cols):
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(f(v) for v in r) + " |" for r in rows])


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def build():
    runs = {os.path.basename(os.path.dirname(p)): json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "*", "sweep.json")))}
    if not runs:
        return print("no sweeps found")
    figs = os.path.join(OUT, "figs"); os.makedirs(figs, exist_ok=True)
    summary = os.path.join(OUT, "summary.md")
    md = ["# gmanifold: sampling results\n"] + ([open(summary).read() + "\n"] if os.path.exists(summary) else []) + [
          "Point clouds: a fixed prefix followed by every real token, residual state at the last position, one cloud per residual location; "
          "90 % of the tokens fit the manifold, 10 % are held out. All distances are in units of the local spacing (distance to the 8-th "
          "real neighbour); `u` is the Guidotti kernel score (1 on fitted states); bands = held-out real states and real states displaced by "
          "0.5× / 1× their spacing. Samples: 2000 per setting, 8× candidates, volume reweighting with `auto_tau=True` (τ = 1 unless the ESS would fall below 5 % of the candidates, in which case the weights are tempered to √det(JᵀJ)^τ with the largest τ meeting the floor; the τ column shows where that engaged).\n"]
    md.append("## 0. Runs\n")
    md.append(table([[k, r["model"], r["D"], len(r["locations"]), r["n_states"], f"{r['seconds'] / 60:.0f} min"] for k, r in runs.items()],
                    ["run", "model", "D", "locations", "states", "time"]) + "\n")

    # 1. dimension + fit quality
    md.append("## 1. Intrinsic dimension and fit quality per location\n")
    for k, r in runs.items():
        rows = []
        fits = defaultdict(list)
        for x in r["fits"]:
            fits[(x["loc"], x["m"])].append(x)
        ms = sorted({x["m"] for x in r["fits"]})
        for d in r["dim"]:
            row = [d["loc"], d["twonn"], d["mle"]]
            for m in ms:
                ff = fits.get((d["loc"], m), [])
                row += [(f"{mean([x['val_recon'] for x in ff]):.3f} (rank {mean([x['rank'] for x in ff]):.0f})" + (" †" if ff[0].get("spacing_unit") == "global" else "")) if ff else "–"]
            rows.append(row)
        md.append(f"**{k}** ({r['model']})\n\n" + table(rows, ["location", "TwoNN", "MLE", *[f"val recon m={m}" for m in ms]]) + "\n\n"
                  + ("† hub-dominated cloud (a few anchors are the nearest neighbour of most points): distances in units of the global median spacing.\n" if any(x.get("spacing_unit") == "global" for x in r["fits"]) else ""))

    # 2. alpha sweep
    md.append("## 2. Samples versus α (mean over locations; bands in the last rows)\n")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for ci, (k, r) in enumerate(runs.items()):
        m0 = min({x["m"] for x in r["alpha"]})
        g = defaultdict(list)
        for x in r["alpha"]:
            if x["m"] == m0:
                g[x["alpha"]].append(x)
        rows = [[a, mean([x["nearest_real"] for x in v]), mean([x["recon"] for x in v]), mean([x.get("tangent") for x in v]), mean([x["u"] for x in v]), min(x["u"] for x in v),
                 mean([x["coverage"] for x in v]), mean([x["ess"] for x in v]), (mean([x["tau"] for x in v]) if all("tau" in x for x in v) else "–"), mean([x["multiplicity"] for x in v])] for a, v in sorted(g.items())]
        b = [x["bands"] for x in r["alpha"] if x["m"] == m0]
        for name in ("held-out real", "real + 0.5x noise", "real + 1x noise"):
            rows.append([name, mean([x[name]["nearest_real"] for x in b]), mean([x[name]["recon"] for x in b]), mean([x[name].get("tangent") for x in b]), mean([x[name]["u"] for x in b]), min(x[name]["u"] for x in b), "–", "–", "–", "–"])
        md.append(f"**{k}**, m = {m0}\n\n" + table(rows, ["α / band", "nearest real", "recon", "tangent resid", "u (mean)", "u (min over loc)", "coverage", "ESS", "τ (mean)", "ball multiplicity"]) + "\n")
        xs = sorted(g)
        ax[0].plot(xs, [mean([x["u"] for x in g[a]]) for a in xs], "-o", ms=3, color=C[ci % 6], label=k)
        ax[1].plot(xs, [mean([x["nearest_real"] for x in g[a]]) for a in xs], "-o", ms=3, color=C[ci % 6], label=k)
        ax[2].plot(xs, [mean([x["coverage"] for x in g[a]]) for a in xs], "-o", ms=3, color=C[ci % 6], label=k)
        for name, ls in (("real + 0.5x noise", ":"), ("real + 1x noise", "--")):
            ax[0].axhline(mean([x[name]["u"] for x in b]), ls=ls, lw=1, color=C[ci % 6])
    ax[0].set_xlabel("α"); ax[0].set_ylabel("median kernel score u of samples"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("α"); ax[1].set_ylabel("nearest real / spacing"); ax[2].set_xlabel("α"); ax[2].set_ylabel("coverage of real states")
    fig.tight_layout(); fig.savefig(os.path.join(figs, "alpha.png")); plt.close(fig)
    md.append("![alpha](figs/alpha.png)\n\n*Dotted / dashed lines: kernel score of real states displaced by 0.5× / 1× spacing (the off-support bands).*\n")

    # 3. latent dim / seeds
    for k, r in runs.items():
        ms = sorted({x["m"] for x in r["alpha"]}); seeds = sorted({x["seed"] for x in r["alpha"]})
        if len(ms) > 1 or len(seeds) > 1:
            md.append(f"## 3. Latent dimension and seed sensitivity ({k})\n")
            rows = []
            for m in ms:
                for a in (0.3, 0.5, 1.0):
                    v = [x for x in r["alpha"] if x["m"] == m and x["alpha"] == a]
                    if v:
                        rows.append([m, a, mean([x["u"] for x in v]), f"{min(x['u'] for x in v):.3f}–{max(x['u'] for x in v):.3f}", mean([x["nearest_real"] for x in v]), mean([x["coverage"] for x in v]), mean([x["ess"] for x in v]),
                                     f"{min(x['val_recon'] for x in r['fits'] if x['m'] == m):.3f}–{max(x['val_recon'] for x in r['fits'] if x['m'] == m):.3f}"])
            md.append(table(rows, ["m", "α", "u mean", "u range (loc × seed)", "nearest real", "coverage", "ESS", "val recon range"]) + "\n")

    # 4. ablations
    for k, r in runs.items():
        if r["ablation"]:
            md.append(f"## 4. Sampler variants and loss ablation ({k}, α = 0.3)\n")
            rows = [[x["loc"], x["variant"], x.get("anchors"), x["ess"], x["coverage"], x["nearest_real"], x["recon"], x["u"]] for x in r["ablation"] if x["kind"] == "sampler"]
            md.append("Sampler variants (same decoder): support radius rule (global = α × median ρ for all anchors, local = α × ρ_i), anchor "
                      "probability ∝ R^p and whether the volume reweighting is applied. `anchors` = participation ratio of the anchor distribution.\n\n"
                      + table(rows, ["location", "variant", "anchors", "ESS", "coverage", "nearest real", "recon", "u"]) + "\n")
            rows = [[x["loc"], x["variant"], x.get("val_recon"), x.get("train_recon"), x["ess"], x["coverage"], x["nearest_real"], x["recon"], x["u"]] for x in r["ablation"] if x["kind"] == "loss"]
            md.append("Loss / architecture variants (each refitted, then sampled at α = 0.3).\n\n"
                      + table(rows, ["location", "variant", "val recon", "train recon", "ESS", "coverage", "nearest real", "recon", "u"]) + "\n")

    # 5. propagation
    md.append("## 5. Propagation through the real model: images vs independent destination fits\n")
    md.append("Samples at the embedding are pushed through the real Transformer stretch `embed → dst` (fixed prefix context) and scored by the "
              "destination manifold fitted only on real destination states. Controls: images of held-out real states, and of real states displaced "
              "by 0.5× / 1× spacing *before* propagation.\n")
    for k, r in runs.items():
        for p in r["propagation"]:
            rows = [[name, v["nearest_real"], v["recon"], v.get("tangent"), v["u"]] for name, v in p["rows"].items()]
            rows += [[f"band: {name}", v["nearest_real"], v["recon"], v.get("tangent"), v["u"]] for name, v in p["bands"].items()]
            md.append(f"**{k}: embed → {p['dst']}**\n\n" + table(rows, ["set", "nearest real", "recon", "tangent resid", "u"]) + "\n")
    # 6. real-occurrence clouds
    st = {os.path.basename(os.path.dirname(p)): json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "*", "stories.json")))}
    if st:
        md.append("## 6. Real-occurrence clouds (random token occurrences in real stories, split by story)\n")
        md.append("Same fit and sampling protocol on residual states of real token occurrences (20k occurrences, 8 per story, "
                  "≤ 256 tokens per story); duplicates and isolated outliers dropped from the training split. Propagation uses the "
                  "tokenwise FFN maps `Lk.attn → Lk.ffn`.\n")
        for k, r in st.items():
            md.append(f"**{k}** ({r['model']}: {r['n']} occurrences from {r['n_stories']} stories, {r['n_val']} held out)\n")
            rows = [[d["loc"], d["n_train"], d["twonn"], d["mle"], next(x["val_recon"] for x in r["fits"] if x["loc"] == d["loc"])] for d in r["dim"]]
            md.append(table(rows, ["location", "train states", "TwoNN", "MLE", "val recon"]) + "\n")
            g = defaultdict(list)
            for x in r["alpha"]:
                g[x["alpha"]].append(x)
            rows = [[a, mean([x["nearest_real"] for x in v]), mean([x["recon"] for x in v]), mean([x["u"] for x in v]), min(x["u"] for x in v), mean([x["coverage"] for x in v]), mean([x["ess"] for x in v])] for a, v in sorted(g.items())]
            b = [x["bands"] for x in r["alpha"]]
            for name in ("held-out real", "real + 0.5x noise", "real + 1x noise"):
                rows.append([name, mean([x[name]["nearest_real"] for x in b]), mean([x[name]["recon"] for x in b]), mean([x[name]["u"] for x in b]), min(x[name]["u"] for x in b), "–", "–"])
            md.append(table(rows, ["α / band", "nearest real", "recon", "u (mean)", "u (min over loc)", "coverage", "ESS"]) + "\n")
            for p_ in r["propagation"][:4]:
                rows = [[name, v["nearest_real"], v["recon"], v["u"]] for name, v in p_["rows"].items()] + [[f"band: {n_}", v["nearest_real"], v["recon"], v["u"]] for n_, v in p_["bands"].items()]
                md.append(f"{p_['src']} → {p_['dst']}\n\n" + table(rows, ["set", "nearest real", "recon", "u"]) + "\n")
    # 7. decoder capacity study
    cap = os.path.join(OUT, "capacity", "capacity.json")
    if os.path.exists(cap):
        rows_ = json.load(open(cap))
        md.append("## 7. Decoder capacity and hyper-parameters (SimpleStories-5M, three locations)\n")
        md.append("Every setting scored by held-out reconstruction / spacing (final and best epoch on the validation split) and by the "
                  "samples at α = 0.3. `default` = hidden (512, 256), 300 epochs, lr 2e-3, batch 512, λ_geom 0.1.\n")
        def tag(r):
            extra = {k: v for k, v in r.items() if k in ("lr", "batch_size", "lam_geom")}
            return "default" if (r["hidden"] == [512, 256] and r["epochs"] == 300 and not extra) else f"hidden {tuple(r['hidden'])}, {r['epochs']} ep" + (f", {extra}" if extra else "")
        for loc in dict.fromkeys(r["loc"] for r in rows_):
            rr = [r for r in rows_ if r["loc"] == loc]
            seen, uniq = set(), []
            for r in rr:
                key = (r["m"], tag(r))
                if key not in seen:
                    seen.add(key); uniq.append(r)
            md.append(f"**{loc}** — latent dimension (default architecture):\n\n" + table([[r["m"], r["val_recon_best"], r["train_recon"], r["u"], r["nearest"], r["coverage"], r["ess"]] for r in uniq if tag(r) == "default"],
                      ["m", "val recon (best)", "train recon", "u", "nearest real", "coverage", "ESS"]) + "\n")
            md.append(f"**{loc}** — architecture / training variants at m = 16 (sorted by held-out reconstruction):\n\n"
                      + table([[tag(r), f"{r['params'] / 1e6:.2f}M", r["val_recon_final"], f"{r['val_recon_best']:.3f} @ {r['best_epoch']}", r["train_recon"], r["u"], r["coverage"], r["ess"]]
                               for r in sorted([r for r in uniq if r["m"] == 16], key=lambda r: r["val_recon_best"])],
                              ["variant", "params", "val recon", "best (epoch)", "train recon", "u", "coverage", "ESS"]) + "\n")
    # 8. how the samples look to the model
    for p in sorted(glob.glob(os.path.join(OUT, "*", "downstream.json"))):
        d = json.load(open(p)); k = os.path.basename(os.path.dirname(p))
        md.append(f"## 8. How the samples look to the model ({k}, {d['model']})\n")
        md.append("Embedding samples are placed at the last position of the prefix and the *full* model is run. (a) Next-token distributions: "
                  "Jensen–Shannon divergence to the distribution of the sample's nearest real token, second-nearest, and a random token; "
                  "entropy; top-1 agreement. (b) Layer-by-layer trajectory of the images against every location's own manifold and bands, with "
                  "`perp` = distance from the segment joining the images of the two nearest real anchors (in spacings) and the fraction of images "
                  "whose projection falls between those two anchor images.\n")
        md.append(table([[r["set"], r["js_nearest"], r["js_second"], r["js_random"], r["entropy"], r["top1_agree_nearest"], r["top1_agree_random"]] for r in d["next_token"]],
                        ["set", "JS to nearest token", "JS to 2nd nearest", "JS to random token", "entropy (real: %.2f)" % d["next_token"][0]["entropy_real"], "top-1 = nearest", "top-1 = random"]) + "\n")
        ex = next((r["examples"] for r in d["next_token"] if "examples" in r), None)
        if ex:
            md.append("Examples (α = first sweep value): the two nearest real tokens of a sample, the sample's top-3 next-token predictions, and the nearest token's top-3.\n\n"
                      + table([[e["nearest"], e["second"], " ".join(e["sample_top3"]), " ".join(e["nearest_top3"])] for e in ex], ["nearest", "2nd nearest", "sample predicts", "nearest token predicts"]) + "\n")
        names = list(d["trajectory"][0]["rows"])
        for key, label in (("u", "kernel score u"), ("nearest_real", "nearest real / spacing"), ("perp_over_spacing", "perp. distance from the anchor-image segment / spacing")):
            rows = [[t["loc"]] + [t["rows"][n][key] for n in names] + ([t["bands"]["real + 1x noise"].get(key if key != "perp_over_spacing" else "nearest_real")] if key != "perp_over_spacing" else []) for t in d["trajectory"]]
            md.append(f"{label} of the images, per location:\n\n" + table(rows, ["location"] + names + (["band: real + 1x noise"] if key != "perp_over_spacing" else [])) + "\n")
        if os.path.exists(os.path.join(os.path.dirname(p), "layers.png")):
            md.append(f"![layers]({k}/layers.png)\n\n*PCA of the real states (grey) and of the images of samples at the smallest (blue) and largest (orange) α, at four locations.*\n")
    open(os.path.join(OUT, "results.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(OUT, "results.md"))
    try:
        import markdown
        from weasyprint import CSS, HTML
        css = ("@page { size: A4; margin: 14mm; @bottom-right { content: counter(page) ' / ' counter(pages); font-size: 7pt; color: #777; } }"
               "body { font-family: 'DejaVu Sans', sans-serif; font-size: 8.5pt; line-height: 1.35; } h1 { font-size: 15pt; } h2 { font-size: 11.5pt; margin-top: 12pt; }"
               "table { border-collapse: collapse; font-size: 7pt; margin: 4pt 0 8pt 0; } th, td { border-bottom: 0.4pt solid #ccc; padding: 1.5pt 4pt; text-align: right; }"
               "th:first-child, td:first-child { text-align: left; } th { background: #f1f1ee; } img { max-width: 100%; max-height: 80mm; display: block; margin: 4pt auto; }"
               "code { font-family: 'DejaVu Sans Mono', monospace; font-size: 7.5pt; }")
        html = markdown.markdown("\n".join(md), extensions=["tables"])
        HTML(string=html, base_url=os.path.abspath(OUT) + "/").write_pdf(os.path.join(OUT, "results.pdf"), stylesheets=[CSS(string=css)])
        print("wrote", os.path.join(OUT, "results.pdf"))
    except Exception as e:
        print("PDF not written:", e)


if __name__ == "__main__":
    build()

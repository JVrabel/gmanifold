"""Re-run the campaign configurations that failed, with the current library, and append the records (recheck=True)."""
import json, os, sys, time
sys.argv = ["campaign.py", "--out", "results/checks"]
src = open("experiments/campaign.py").read().split("# ------------------------------------------------------------------ queue")[0]
ns = {}; exec(compile(src, "campaign", "exec"), ns)
rows = [json.loads(l) for l in open("results/checks/checks.jsonl")]
todo = [r for r in rows if r.get("pass") is False and r["check"] in ("real_alpha", "propagation") and not r.get("recheck")]
log = open("results/checks/checks.jsonl", "a")
for r in todo:
    t = time.time(); rec = dict(i=r["i"], check=r["check"], args=r["args"], recheck=True, time=time.strftime("%H:%M:%S", time.gmtime()))
    try:
        rec.update(ns[r["check"]](**r["args"]))
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["seconds"] = time.time() - t; log.write(json.dumps(rec) + "\n"); log.flush()
    print(f"recheck {r['check']} {r['args']} -> {'PASS' if rec.get('pass') else 'FAIL'} ({rec['seconds']:.0f}s)", flush=True)
print("RECHECK_DONE")

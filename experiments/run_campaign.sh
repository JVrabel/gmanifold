#!/bin/bash
# 8-hour check campaign, then commit and push the summary (GH_TOKEN must be in the environment for the push).
cd /workspace/projects/gmanifold && source /venv/main/bin/activate
python experiments/campaign.py --hours ${1:-8} --out results/checks
git add results/checks/CHECKS.md results/checks/checks.jsonl experiments/campaign.py experiments/run_campaign.sh
git -c commit.gpgsign=false commit -q -m "Check campaign: $(grep -m1 'checks in' results/checks/CHECKS.md)" && git push -q "https://x-access-token:${GH_TOKEN}@github.com/JVrabel/gmanifold.git" main && echo PUSHED
echo "=== CAMPAIGN_DONE $(date -u)"

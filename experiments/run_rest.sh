#!/bin/bash
cd /workspace/projects/gmanifold && source /venv/main/bin/activate
run() { echo "=== $(date -u +%H:%M:%S) START $*"; "$@" && echo "=== $(date -u +%H:%M:%S) OK $*" || echo "=== $(date -u +%H:%M:%S) FAIL $*"; }
run python -m pytest tests -q
run python experiments/sweep.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out results/TinyLlama --m 16 --seeds 0 1 --epochs 150 --locs embed L0.attn L0.ffn L3.ffn L7.ffn L11.ffn L15.ffn L21.ffn
run python experiments/sweep.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out results/TinyLlama_m64 --m 64 --seeds 0 --epochs 150 --no-propagation --locs embed L0.ffn L3.ffn L11.ffn L21.ffn
run python experiments/sweep.py --model SimpleStories/SimpleStories-35M  --out results/35M_seeds --m 16 --seeds 2 3 4 --no-propagation
run python experiments/sweep.py --model SimpleStories/SimpleStories-5M   --out results/5M_prefix2 --prefix "The" --m 16 --seeds 0 1
run python experiments/sweep.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out results/TinyLlama_all --m 16 --seeds 0 --epochs 150 --no-propagation
run python experiments/stories.py --model SimpleStories/SimpleStories-5M  --out results/5M_stories  --n 20000 --epochs 150
run python experiments/stories.py --model SimpleStories/SimpleStories-35M --out results/35M_stories --n 20000 --epochs 150
run python experiments/report.py
echo "=== REST_DONE $(date -u)"

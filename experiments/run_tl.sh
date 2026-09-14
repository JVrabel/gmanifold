#!/bin/bash
cd /workspace/projects/gmanifold && source /venv/main/bin/activate
run() { echo "=== $(date -u +%H:%M:%S) START $*"; "$@" && echo "=== $(date -u +%H:%M:%S) OK $*" || echo "=== $(date -u +%H:%M:%S) FAIL $*"; }
run python experiments/sweep.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out results/TinyLlama --m 16 --seeds 0 1 --epochs 150 --locs embed L0.attn L0.ffn L3.ffn L7.ffn L11.ffn L15.ffn L21.ffn
run python experiments/sweep.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out results/TinyLlama_m64 --m 64 --seeds 0 --epochs 150 --no-propagation --locs embed L0.ffn L3.ffn L11.ffn L21.ffn
run python experiments/sweep.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out results/TinyLlama_all --m 16 --seeds 0 --epochs 150 --no-propagation
run python experiments/report.py
echo "=== TL_DONE $(date -u)"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m sysone.envs.tabletop --out data/tabletop --train ${TRAIN:-3000} --val 300 --test 500
python -m sysone.train_sft --data data/tabletop/train.jsonl --out runs/sft --epochs ${EPOCHS:-1}
python -m sysone.eval --ckpt runs/sft --out runs/sft/eval.json
python -m sysone.eval --ckpt runs/sft --fit-temp data/tabletop/val.jsonl --save-calibrated runs/sft_ts --out runs/sft_ts_eval.json
python -m sysone.train_rlcd --init runs/sft --out runs/rlcd --steps ${RLCD_STEPS:-300}
python -m sysone.eval --ckpt runs/rlcd --out runs/rlcd/eval.json
python -m sysone.baseline_generate --limit 100 --out runs/baseline_generate.json

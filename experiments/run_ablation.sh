#!/usr/bin/env bash
# Ablation: does hysteresis memory add capacity over plain causal attention?
#
# Three arms per sequence length:
#   mem            -> memory ON,  ffn 2048   (7,612,138 params)
#   nomem          -> memory OFF, ffn 2048   (5,774,824 params)  -- smaller
#   nomem-matched  -> memory OFF, ffn 2944   (7,613,416 params)  -- param-matched
#
# The matched arm is the one that matters: if `mem` beats it, the win comes from
# the architecture, not from 1.8M extra weights.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="--amp --tf32 --out-dir results"
BASE="--hidden-dim 256 --num-layers 4 --lr 1e-3 --warmup 200 --min-lr-ratio 0.1 --steps-per-epoch 200"

echo "### GATE: plain decoder must solve L=65 (validated: 99.94% tok, 98.05% seq)."
echo "###   Copy switches on around epoch 9-12; a 15-epoch budget or lr=3e-4"
echo "###   cuts it off before that and makes every arm look dead."
python experiments/train_prototype.py --task copy --seq-len 65 --epochs 40 \
  --batch-size 64 --eval-batch-size 32 --no-memory --ffn-dim 2944 \
  $BASE $COMMON --out-dir results_gate
python - <<'PY'
import json, sys
s = json.load(open("results_gate/summary_copy_L65_nomem-matched.json"))
print(f"GATE token acc {s['token_acc']*100:.2f}% | pos rank {s['pos_diag']['rank']} "
      f"| near-dupes {s['pos_diag']['near_dupes']}")
if s['token_acc'] < 0.95:
    sys.exit("GATE FAILED: fix positional encoding before running the suite")
print("GATE PASSED")
PY

echo "### 0. Control: pure noise (loss must stay at ln(1000)=6.908)"
python experiments/train_prototype.py --task noise --seq-len 65 \
  --epochs 5 --batch-size 32 $BASE $COMMON

for L in 65 257 513 1025; do
  case $L in
    65)   BS=64; EBS=32; EP=40;  CK="";             ;;
    257)  BS=32; EBS=16; EP=60;  CK="";             ;;
    513)  BS=16; EBS=8;  EP=80;  CK="";             ;;
    1025) BS=8;  EBS=4;  EP=100; CK="--checkpoint"; ;;
  esac
  for ARM in mem nomem nomem-matched; do
    case $ARM in
      mem)           FLAG="";                          ;;
      nomem)         FLAG="--no-memory";               ;;
      nomem-matched) FLAG="--no-memory --ffn-dim 2944";;
    esac
    echo ""
    echo "### copy L=$L arm=$ARM bs=$BS epochs=$EP"
    python experiments/train_prototype.py \
      --task copy --seq-len $L --epochs $EP \
      --batch-size $BS --eval-batch-size $EBS \
      $BASE $FLAG $CK $COMMON
  done
done

echo ""
echo "### Summary"
python - <<'PY'
import glob, json
rows = [json.load(open(f)) for f in sorted(glob.glob("results/summary_*.json"))]
print(f"{'tag':30} {'params':>11} {'floor':>7} {'train':>7} {'copy':>8} "
      f"{'tok%':>7} {'seq%':>7} {'min':>6} {'<floor':>7}")
for s in rows:
    print(f"{s['tag']:30} {s['params']:>11,} {s['floor']:>7.3f} "
          f"{s['final_train_loss']:>7.3f} {s['copy_loss']:>8.4f} "
          f"{s['token_acc']*100:>7.2f} {s['seq_acc']*100:>7.2f} "
          f"{s['minutes']:>6.1f} {str(s['below_floor']):>7}")

print("\nMemory vs param-matched baseline (exact-sequence accuracy):")
by = {s['tag']: s for s in rows}
for L in (65, 257, 513, 1025):
    a, b = by.get(f"copy_L{L}_mem"), by.get(f"copy_L{L}_nomem-matched")
    if a and b:
        d = (a['seq_acc'] - b['seq_acc']) * 100
        verdict = "memory helps" if d > 2 else ("memory hurts" if d < -2 else "no difference")
        print(f"  L={L:>5}: mem {a['seq_acc']*100:6.2f}% vs matched {b['seq_acc']*100:6.2f}% "
              f"-> {d:+6.2f} pp  ({verdict})")
PY
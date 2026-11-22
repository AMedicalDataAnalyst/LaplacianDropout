#!/usr/bin/env bash
# Night queue 2026-11-22: does band dropout still help on an ImageNet-pretrained
# backbone? (the cheap version of the ImageNet gate). Designed once; do not edit
# while it runs. One cell = finetune_pretrained.py on (dataset, init, lr, seed),
# four arms each. Cells are ordered so a time cut only drops the least important.
#
#   P1  pretrained, lr 3e-4, seeds 0-2          (the pre-registered test)
#   P2  pretrained, lr 3e-4, seeds 3-4          (tighter SE)
#   P3  pretrained, lr 1e-3, seeds 0-2          (is the answer LR-specific?)
#   P4  from scratch, lr 3e-3, 50 epochs, seeds 0-1, imagewoof + dtd
#       (same datasets without pretraining; imagenette-from-scratch is Phase 2)
#
# Nothing starts unless its estimated time fits before END (default 06:50).
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
OUT=results/phase3_pretrained
LOG=logs/phase3/night_1122.log
END_EPOCH=$(date -d "${END:-tomorrow 06:50}" +%s)
mkdir -p "$OUT" logs/phase3

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

epochs_for() {  # ~2000 optimiser steps for the pretrained cells
  case "$1:$2" in
    dtd:pretrained) echo 50 ;;
    *:pretrained)   echo 20 ;;
    *:scratch)      echo 50 ;;
  esac
}

cell() {  # dataset init lr seed est_minutes
  local d=$1 init=$2 lr=$3 seed=$4 est=$5
  local tag="${d}__${init}__lr${lr}__s${seed}"
  local out="$OUT/$tag.json"
  [[ -f "$out" ]] && { say "skip $tag (exists)"; return; }
  local left=$(( (END_EPOCH - $(date +%s)) / 60 ))
  if (( left < est )); then say "GUARD $tag needs ~${est} min, ${left} left: not started"; return; fi
  say "start $tag"
  if $PY -u finetune_pretrained.py --dataset "$d" --init "$init" --lr "$lr" --seed "$seed" \
        --epochs "$(epochs_for "$d" "$init")" --out "$out" > "logs/phase3/$tag.txt" 2>&1; then
    say "done $tag"
  else
    say "FAILED $tag (see logs/phase3/$tag.txt)"
  fi
}

say "QUEUE START, end guard $(date -d @$END_EPOCH '+%F %T')"
for s in 0 1 2; do for d in dtd imagewoof imagenette; do cell $d pretrained 3e-4 $s 28; done; done
say "P1 DONE"
for s in 3 4;   do for d in dtd imagewoof imagenette; do cell $d pretrained 3e-4 $s 28; done; done
say "P2 DONE"
for s in 0 1 2; do for d in dtd imagewoof imagenette; do cell $d pretrained 1e-3 $s 28; done; done
say "P3 DONE"
for s in 0 1;   do for d in dtd imagewoof; do cell $d scratch 3e-3 $s 50; done; done
say "P4 DONE"
$PY report_pretrained.py >> "$LOG" 2>&1
say "NIGHT DONE"

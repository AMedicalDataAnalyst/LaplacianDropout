#!/usr/bin/env bash
# Polls until the de-risk's combo eval JSON appears, then fires the
# reviewer-defense run sequence. Run in background:
#   bash queue_after_derisk.sh > queue_after_derisk.log 2>&1 &
# Times out after 24h to avoid silently waiting forever if de-risk dies.
set -u
cd "$(dirname "$0")"

TRIGGER=cifar_results/cifar_eval_cifar10_band_drop_all_p_augmix_seed0.json
TIMEOUT=86400   # 24h
INTERVAL=120    # 2 min between checks
WAITED=0

echo "$(date -Iseconds)  waiting for de-risk to land $TRIGGER (timeout ${TIMEOUT}s)"

while [[ ! -f "$TRIGGER" ]]; do
  sleep "$INTERVAL"
  WAITED=$((WAITED + INTERVAL))
  if [[ $WAITED -ge $TIMEOUT ]]; then
    echo "$(date -Iseconds)  TIMEOUT after ${WAITED}s without $TRIGGER. Not firing reviewer-defense."
    exit 1
  fi
done

echo "$(date -Iseconds)  de-risk trigger seen. Sleeping 60s as grace then firing reviewer-defense."
sleep 60

bash run_reviewer_defense.sh
echo "$(date -Iseconds)  queue_after_derisk done. exit=$?"

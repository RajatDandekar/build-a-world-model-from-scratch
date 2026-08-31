#!/bin/zsh
# Pull logs, probe jsons, and viz npz from the Modal results volume (not the
# heavy checkpoints). Usage: ./sync_results.sh
set -e
cd "$(dirname "$0")"
MODAL="/Users/raj/.venv/bin/modal"
export MODAL_PROFILE=rajatworkspace
mkdir -p results
for run in $($MODAL volume ls ijepa-l6-results 2>/dev/null | grep -vE "probes|viz|smoke" ); do
  mkdir -p "results/$run"
  $MODAL volume get --force ijepa-l6-results "$run/log.jsonl" "results/$run/log.jsonl" 2>/dev/null || true
  $MODAL volume get --force ijepa-l6-results "$run/meta.json" "results/$run/meta.json" 2>/dev/null || true
done
$MODAL volume get --force ijepa-l6-results probes results/ 2>/dev/null || true
$MODAL volume get --force ijepa-l6-results viz results/ 2>/dev/null || true
echo "synced:"
find results -type f | sort

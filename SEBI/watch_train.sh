#!/usr/bin/env bash
# watch_train.sh — filter mlx_lm.lora stdout for meaningful events only
# Emits one line per event: checkpoints, val losses, errors, completion.
# Designed to be piped to Monitor so each event becomes a notification.
#
# Usage: ./watch_train.sh /path/to/train_log.txt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${1:-$SCRIPT_DIR/SEBI_CPT/train_log_exp.txt}"

if [ ! -f "$LOG" ]; then
  echo "ERROR: log file $LOG does not exist yet"
  exit 1
fi

# Tail the log, emit only on:
# - Val loss lines (every steps_per_eval)
# - Checkpoint save lines (every steps_per_save)
# - Final weights save
# - Tracebacks / errors / OOM / Killed
# - Loading messages (start signals)
tail -f "$LOG" 2>&1 | grep --line-buffered -E \
  "Val loss|Saved adapter weights|Saved final weights|Loading pretrained model|Loading datasets|Traceback|Error|OOM|Killed|OutOfMemory|Test loss|Test ppl"

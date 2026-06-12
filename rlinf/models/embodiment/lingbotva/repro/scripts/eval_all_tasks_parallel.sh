#!/bin/bash
# Dispatch all 10 tasks × 5 episodes across 8 GPUs in parallel.
# Usage: bash eval_all_tasks_parallel.sh [out_dir]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOG_ROOT="${LOG_ROOT:-/workspace/rl_logs}"
source "$SCRIPT_DIR/env.sh"

OUT_DIR="${1:-/workspace/rl_logs/eval_all_tasks_combined}"
WORKER="$SCRIPT_DIR/eval_worker.py"
NUM_GPUS=8
LOGDIR="$OUT_DIR/worker_logs"
mkdir -p "$OUT_DIR" "$LOGDIR"

# Build job list: (task_id, ep_idx) for all 10 tasks × 5 episodes
TASKS=(0 1 2 3 4 5 6 7 8 9)
EPS=(0 1 2 3 4)

pids=()
gpu_idx=0

for task_id in "${TASKS[@]}"; do
    for ep_idx in "${EPS[@]}"; do
        logfile="$LOGDIR/t${task_id}_ep${ep_idx}.log"
        echo "[dispatch] GPU$gpu_idx  task=$task_id ep=$ep_idx  log=$logfile"
        CUDA_VISIBLE_DEVICES=$gpu_idx "$PY" "$WORKER" "$task_id" "$ep_idx" "$OUT_DIR" \
            > "$logfile" 2>&1 &
        pids+=($!)
        gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

        # Wait for a wave of 8 before launching more
        if [ ${#pids[@]} -ge $NUM_GPUS ]; then
            echo "[dispatch] waiting for wave of ${#pids[@]} workers..."
            for pid in "${pids[@]}"; do
                wait "$pid" || echo "[dispatch] worker PID=$pid exited non-zero"
            done
            pids=()
            echo "[dispatch] wave done, continuing..."
        fi
    done
done

# Wait for any remaining jobs
if [ ${#pids[@]} -gt 0 ]; then
    echo "[dispatch] waiting for final ${#pids[@]} workers..."
    for pid in "${pids[@]}"; do
        wait "$pid" || echo "[dispatch] worker PID=$pid exited non-zero"
    done
fi

echo ""
echo "============================================================"
echo "All workers done. Aggregating results..."
echo "============================================================"

# Print per-task success counts
"$PY" - "$OUT_DIR" <<'PYEOF'
import sys, json, pathlib

out_dir = pathlib.Path(sys.argv[1])
task_names = [
    "alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce", "ketchup",
    "tomato_sauce", "butter", "milk", "choc_pudding", "orange_juice",
]
total_ok = total = 0
for tid, name in enumerate(task_names):
    td = out_dir / f"task{tid}_{name}"
    ok = 0
    for ep in range(5):
        jf = td / f"ep{ep}.json"
        if jf.exists():
            d = json.loads(jf.read_text())
            ok += int(d.get("success", False))
            total += 1
    total_ok += ok
    print(f"  task{tid:2d} {name:<20s}  {ok}/5")
print(f"\n  TOTAL: {total_ok}/{total}  ({100*total_ok/max(total,1):.1f}%)")
PYEOF

# Regenerate HTML gallery
echo ""
echo "[dispatch] regenerating HTML gallery..."
"$PY" - "$OUT_DIR" <<'PYEOF'
import json, pathlib

out_dir = pathlib.Path(sys.argv[1])
task_names = [
    "alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce", "ketchup",
    "tomato_sauce", "butter", "milk", "choc_pudding", "orange_juice",
]

import sys

rows = []
for tid, name in enumerate(task_names):
    td = out_dir / f"task{tid}_{name}"
    for ep in range(5):
        vid = td / f"ep{ep}.mp4"
        jf  = td / f"ep{ep}.json"
        if not vid.exists():
            continue
        ok = False
        if jf.exists():
            ok = json.loads(jf.read_text()).get("success", False)
        label = "SUCCESS" if ok else "FAIL"
        color  = "#5f5" if ok else "#f55"
        rel    = vid.relative_to(out_dir)
        rows.append(f"""
        <div class="card">
          <video controls loop muted playsinline src="{rel}" width="256"></video>
          <div class="lbl" style="color:{color}">t{tid} {name} ep{ep+1}/5 — {label}</div>
        </div>""")

html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>LingBot-VA GRPO step-170 eval</title>
<style>
  body {{background:#111;color:#ddd;font-family:monospace}}
  h1 {{margin:16px}}
  .grid {{display:flex;flex-wrap:wrap;gap:8px;padding:12px}}
  .card {{background:#222;padding:6px;border-radius:4px}}
  .lbl  {{font-size:11px;margin-top:4px;text-align:center}}
</style></head><body>
<h1>LingBot-VA GRPO step-170 — all tasks eval (obs + imagination)</h1>
<div class="grid">{''.join(rows)}
</div></body></html>"""

(out_dir / "index.html").write_text(html)
print(f"[gallery] wrote {out_dir}/index.html  ({len(rows)} videos)")
PYEOF

echo "[eval_all_tasks_parallel] done"

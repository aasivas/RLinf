#!/bin/bash
# Eval GRPO step-90 checkpoint across all 10 tasks x 5 episodes on 8 GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOG_ROOT="${LOG_ROOT:-/workspace/rl_logs}"
source "$SCRIPT_DIR/env.sh"

CKPT="/workspace/rl_logs/grpo_exact_train_logs/libero_object_grpo_lingbotva/checkpoints/global_step_90/actor/model_state_dict/full_weights.pt"
OUT_DIR="${1:-/workspace/rl_logs/eval_step90}"
WORKER="$SCRIPT_DIR/eval_worker.py"
NUM_GPUS=8
LOGDIR="$OUT_DIR/worker_logs"
mkdir -p "$OUT_DIR" "$LOGDIR"

TASKS=(0 1 2 3 4 5 6 7 8 9)
EPS=(0 1 2 3 4)

pids=()
gpu_idx=0

for task_id in "${TASKS[@]}"; do
    for ep_idx in "${EPS[@]}"; do
        logfile="$LOGDIR/t${task_id}_ep${ep_idx}.log"
        echo "[dispatch] GPU$gpu_idx  task=$task_id ep=$ep_idx  log=$logfile"
        CUDA_VISIBLE_DEVICES=$gpu_idx "$PY" "$WORKER" "$task_id" "$ep_idx" "$OUT_DIR" "$CKPT" \
            > "$logfile" 2>&1 &
        pids+=($!)
        gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

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

"$PY" - "$OUT_DIR" <<'PYEOF'
import sys, json, pathlib

out_dir = pathlib.Path(sys.argv[1])
task_names = [
    "alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce", "ketchup",
    "tomato_sauce", "butter", "milk", "choc_pudding", "orange_juice",
]

rows = []
summary_rows = []
total_ok = total = 0

for tid, name in enumerate(task_names):
    td = out_dir / f"task{tid}_{name}"
    task_ok = 0
    for ep in range(5):
        vid = td / f"ep{ep}.mp4"
        jf  = td / f"ep{ep}.json"
        if not vid.exists():
            continue
        ok = False
        if jf.exists():
            ok = json.loads(jf.read_text()).get("success", False)
        task_ok += int(ok)
        total_ok += int(ok)
        total += 1
        label = "SUCCESS" if ok else "FAIL"
        color  = "#5f5" if ok else "#f55"
        rel    = vid.relative_to(out_dir)
        rows.append(f"""
        <div class="card">
          <video controls loop muted playsinline src="{rel}" width="256"></video>
          <div class="lbl" style="color:{color}">t{tid} {name} ep{ep+1}/5 — {label}</div>
        </div>""")
    summary_rows.append(f"<tr><td>t{tid}</td><td>{name}</td><td style='color:{'#5f5' if task_ok>=3 else '#fa0' if task_ok>=2 else '#f55'}'>{task_ok}/5</td></tr>")

sr_pct = 100*total_ok/max(total,1)
html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>LingBot-VA GRPO step-90 eval — LIBERO-Object</title>
<style>
  body {{background:#111;color:#ddd;font-family:monospace;margin:0;padding:12px}}
  h1 {{margin:8px 0 4px}}
  .nav {{margin-bottom:14px;font-size:13px}}
  .nav a {{color:#7af;text-decoration:none;border:1px solid #7af;padding:3px 10px;border-radius:3px;margin-right:6px}}
  .nav a:hover {{background:#7af;color:#111}}
  .summary {{margin-bottom:12px;font-size:13px}}
  table {{border-collapse:collapse;margin-bottom:16px}}
  td,th {{padding:3px 10px;border:1px solid #333}}
  th {{background:#222}}
  .grid {{display:flex;flex-wrap:wrap;gap:8px}}
  .card {{background:#222;padding:6px;border-radius:4px}}
  .lbl  {{font-size:11px;margin-top:4px;text-align:center}}
</style></head><body>
<h1>LingBot-VA GRPO step-90 (peak checkpoint) — LIBERO-Object eval</h1>
<div class="nav">
  <a href="../eval_sft/index.html">&#8592; SFT baseline (step 0)</a>
  <a href="../eval_all_tasks_combined/index.html">GRPO step-170 &#8594;</a>
</div>
<div class="summary">
  <b>Overall SR: {total_ok}/{total} ({sr_pct:.1f}%)</b> &nbsp;|&nbsp; top row = observation, bottom row = imagination/diffusion
  <table style="margin-top:8px">
    <tr><th>id</th><th>task</th><th>SR (5 eps)</th></tr>
    {''.join(summary_rows)}
  </table>
</div>
<div class="grid">{''.join(rows)}
</div></body></html>"""

(out_dir / "index.html").write_text(html)
print(f"[gallery] wrote {out_dir}/index.html  ({len(rows)} videos, SR={total_ok}/{total} = {sr_pct:.1f}%)")
PYEOF

echo "[eval_step90_parallel] done"

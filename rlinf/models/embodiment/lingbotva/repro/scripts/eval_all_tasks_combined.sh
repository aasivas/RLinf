#!/bin/bash
# Run eval_all_tasks_combined.py: all 10 tasks x 5 episodes, combined videos.
set -u
export LOG_ROOT="${LOG_ROOT:-/workspace/rl_logs}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

CKPT="/workspace/rl_logs/grpo_exact_train_logs/libero_object_grpo_lingbotva/checkpoints/global_step_170/actor/model_state_dict/full_weights.pt"

export LINGBOT_VA_MODEL_PATH="$BASE_MODEL"
export LINGBOT_VA_TRANSFORMER_STATE_DICT_PATH="$CKPT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /workspace/rl_logs/eval_all_tasks_combined

CUDA_VISIBLE_DEVICES=0 "$PY" \
    "$(dirname "${BASH_SOURCE[0]}")/eval_all_tasks_combined.py" \
    2>&1 | tee /workspace/rl_logs/eval_all_tasks_combined/run.log

echo "[eval_all_tasks_combined] done"

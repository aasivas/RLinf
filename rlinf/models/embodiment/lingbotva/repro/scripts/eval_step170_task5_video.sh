#!/bin/bash
# Evaluate step-170 checkpoint on task 5, saving both the imaginated
# video (decoded from diffusion latents) and the LIBERO observation video.
# One episode is run (single env, single pass).
#
# Output goes to $LOG_ROOT/eval_step170_task5_video/:
#   imagination_env0.mp4  — decoded diffusion frames for env 0
#   observations_env0.mp4 — agentview + wrist concatenated per step
#   eval_results.json     — success/fail outcome
#
# Usage: bash eval_step170_task5_video.sh

set -u
export LOG_ROOT="${LOG_ROOT:-/workspace/rl_logs}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

CKPT="/workspace/rl_logs/grpo_exact_train_logs/libero_object_grpo_lingbotva/checkpoints/global_step_170/actor/model_state_dict/full_weights.pt"
OUT="$LOG_ROOT/eval_step170_task5_video"
mkdir -p "$OUT"

export LINGBOT_VA_MODEL_PATH="$BASE_MODEL"
export LINGBOT_VA_TRANSFORMER_STATE_DICT_PATH="$CKPT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 "$PY" \
    "$(dirname "${BASH_SOURCE[0]}")/eval_step170_task5_video.py" \
    --config-path "$REPO_PATH/examples/embodiment/config" \
    --config-name libero_object_eval_lingbotva \
    env.eval.total_num_envs=1 \
    algorithm.eval_rollout_epoch=1 \
    "env.eval.task_id_filter=[5]" \
    "+env.eval.eval_reset_start_idx=0" \
    env.eval.video_cfg.save_video=False \
    "actor.model.lingbotva.save_root=$OUT/runtime" \
    "runner.logger.log_path=$OUT" \
    2>&1 | tee "$OUT/eval.log"

echo "[eval_step170_task5_video] done -> $OUT"

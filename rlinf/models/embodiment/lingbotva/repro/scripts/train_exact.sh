#!/bin/bash
# ============================================================================
# THE WINNING RUN: GRPO RL with the (near-)exact gradient on LingBot-VA.
# ============================================================================
# All-10-task Libero-Object GRPO from the SFT checkpoint.
# Based on the 2-task winning config (SFT 40% -> step 30 70.0%, z=2.45) but
# expanded to all 10 tasks to test generalisation beyond {2,6}.
#
# Why this config works where earlier ones were flat:
#   recompute_kv_replay=true  -> exact KV-cache-replay recompute (proximal
#                                ratio ~0.91, vs the biased no-replay ~0.6).
#   ignore_terminations=true  -> every env runs the full 240 steps -> the
#                                rollout buffer is a non-ragged [chunks x envs]
#                                grid -> per-rank FSDP collectives are
#                                symmetric by construction -> no distributed
#                                deadlock (the ragged-buffer failure, see
#                                NEXT_SESSION.md "gotchas").
#
# Disaggregated placement (actor 0-3 / rollout 4-7) + ring weight sync is
# REQUIRED on this container: collocated CUDA-IPC weight sync hits pidfd_getfd
# EPERM (ptrace_scope locked). See ENV_PREP.md.
#
# Usage:  train_exact.sh [MAX_STEPS] [RESUME_DIR]
#   MAX_STEPS   default 30
#   RESUME_DIR  optional global_step_<N> dir to resume from (e.g. to extend a run)
#
# ~25-30 min/step (ignore_terminations keeps envs stepping after success).
# Checkpoints (every 10 steps) under $LOG_ROOT/grpo_exact_train_logs/.../checkpoints/.
# Disk management: a background loop keeps dcp_checkpoint (29 GB resume-state)
# only for the two most recent saves; older saves retain only full_weights.pt
# (9.5 GB eval-weights). Peak disk usage: ~2×38 GB for full + N×9.5 GB for
# eval-weights-only saves.
# In-loop deterministic eval (val_check_interval=10) runs 4 envs at noise=0
# every 10 steps as a lightweight progress signal; use eval_tasks.sh for
# rigorous large-N evals after training.
# ============================================================================
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

MAX_STEPS="${1:-30}"
RESUME_DIR="${2:-}"
# Extra hydra overrides forwarded verbatim, e.g. '+env.train.task_sr_override=[...]'
EXTRA_OVERRIDE_ARGS=("${@:3}")
export LINGBOT_VA_MODEL_PATH="$BASE_SFT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # OK here: disaggregated => ring sync, no IPC
export TORCH_NCCL_BLOCKING_WAIT=1                          # surface NCCL desync instead of silent hang

RESUME_ARG=()
[ -n "$RESUME_DIR" ] && RESUME_ARG=("+runner.resume_dir=$RESUME_DIR")

# ---------------------------------------------------------------------------
# Background checkpoint cleanup: keep dcp_checkpoint only for the last 2 saves.
# Older global_step_N dirs retain actor/model_state_dict/full_weights.pt (eval
# weights) but have their dcp_checkpoint (resume state) removed to save ~29 GB
# per pruned step.
# ---------------------------------------------------------------------------
_cleanup_old_checkpoints() {
  local ckpt_base="$1"
  while true; do
    sleep 120
    [ -d "$ckpt_base" ] || continue
    mapfile -t steps < <(
      find "$ckpt_base" -maxdepth 1 -type d -name "global_step_*" \
        | sort -V
    )
    n="${#steps[@]}"
    (( n > 2 )) || continue
    for (( i=0; i < n-2; i++ )); do
      dcp="${steps[$i]}/actor/dcp_checkpoint"
      if [ -d "$dcp" ]; then
        echo "[cleanup] Pruning dcp_checkpoint: $dcp"
        rm -rf "$dcp"
      fi
    done
  done
}

CKPT_BASE="$LOG_ROOT/grpo_exact_train_logs/libero_object_grpo_lingbotva/checkpoints"
_cleanup_old_checkpoints "$CKPT_BASE" &
CLEANUP_PID=$!
trap 'kill "$CLEANUP_PID" 2>/dev/null || true' EXIT

"$PY" examples/embodiment/train_embodied_agent.py \
  --config-path "$REPO_PATH/examples/embodiment/config" \
  --config-name libero_object_grpo_lingbotva \
  runner.max_steps="$MAX_STEPS" runner.val_check_interval=-1 runner.save_interval=10 \
  "${RESUME_ARG[@]}" \
  actor.model.lingbotva.enable_offload=true rollout.model.lingbotva.enable_offload=true \
  env.train.total_num_envs=32 algorithm.group_size=8 env.train.group_size=8 \
  env.train.ignore_terminations=true \
  algorithm.rollout_epoch=1 actor.global_batch_size=32 actor.micro_batch_size=8 \
  actor.model.lingbotva.recompute_kv_replay=true actor.optim.lr=3.0e-6 \
  actor.model.lingbotva.rl_noise_level=1.0 rollout.model.lingbotva.rl_noise_level=1.0 \
  actor.model.lingbotva.kv_replay_max_history=24 actor.model.lingbotva.kv_replay_max_frames=96 \
  env.train.max_steps_per_rollout_epoch=240 env.train.max_episode_steps=240 \
  +env.train.adaptive_task_sampling=true \
  env.eval.total_num_envs=40 \
  +weight_syncer.use_ring_sync=true \
  runner.logger.log_path="$LOG_ROOT/grpo_exact_train_logs" \
  "${EXTRA_OVERRIDE_ARGS[@]}"

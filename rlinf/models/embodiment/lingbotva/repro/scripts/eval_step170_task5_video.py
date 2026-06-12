"""Evaluate step-170 on task 5 and save imagination + observation videos.

Two videos are saved per env per episode:
  imagination_env<N>.mp4  — the model's imaginated future frames, decoded from
                            the diffusion latents at each chunk.  Concatenated
                            left-right: agentview | eye-in-hand (matches the
                            two-camera latent layout the model uses).
  observations_env<N>.mp4 — raw LIBERO agentview frames collected step-by-step.

The imagination capture works by wrapping _infer_batch_impl so it also returns
the final video latents (shape [B, 48, FC, h, w]) alongside the actions.  After
each chunk we decode those latents with the server's VAE (using the same
denormalize path as wan_va_server.decode_one_video) and append the pixel frames.
"""

from __future__ import annotations

import json
import time
import types
from pathlib import Path

import hydra
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from rlinf.envs.libero.libero_env import LiberoEnv
from rlinf.models import get_model


# ---------------------------------------------------------------------------
# Imagination capture: wrap _infer_batch_impl to leak the video latents
# ---------------------------------------------------------------------------

def _patch_backend_for_latent_capture(backend):
    """Monkey-patch backend._infer_batch_impl to also store the video latents.

    After each call the decoded latent is appended to
    ``backend._captured_latents``, a list[Tensor] in CPU float32.
    """
    original_impl = backend._infer_batch_impl

    def _patched_impl(obs_batch, *, frame_st_id=0):
        server = backend._server
        obs_sequences = backend._normalize_obs_sequences(obs_batch)
        batch_size = len(obs_sequences)

        if frame_st_id == 0 and server.init_latent is None:
            server.init_latent = backend._encode_obs_batch(obs_sequences)

        latents = torch.randn(
            batch_size,
            48,
            server.job_config.frame_chunk_size,
            server.latent_height,
            server.latent_width,
            device=server.device,
            dtype=server.dtype,
        )
        actions = torch.randn(
            batch_size,
            server.job_config.action_dim,
            server.job_config.frame_chunk_size,
            server.action_per_frame,
            1,
            device=server.device,
            dtype=server.dtype,
        )

        server.scheduler.set_timesteps(server.job_config.num_inference_steps)
        server.action_scheduler.set_timesteps(
            server.job_config.action_num_inference_steps
        )
        timesteps = torch.nn.functional.pad(
            server.scheduler.timesteps, (0, 1), mode="constant", value=0
        )
        if server.job_config.video_exec_step != -1:
            timesteps = timesteps[: server.job_config.video_exec_step]
        action_timesteps = torch.nn.functional.pad(
            server.action_scheduler.timesteps, (0, 1), mode="constant", value=0
        )

        with torch.no_grad():
            for step_idx, timestep in enumerate(timesteps):
                last_step = step_idx == len(timesteps) - 1
                latent_cond = (
                    server.init_latent[:, :, 0:1] if frame_st_id == 0 else None
                )
                input_dict = backend._prepare_batch_input(
                    latent_model_input=latents,
                    action_model_input=None,
                    latent_t=float(timestep),
                    action_t=float(timestep),
                    latent_cond=latent_cond,
                    action_cond=None,
                    frame_st_id=frame_st_id,
                )
                video_noise_pred = server.transformer(
                    input_dict["latent_res_lst"],
                    update_cache=1 if last_step else 0,
                    cache_name=server.cache_name,
                    action_mode=False,
                )
                if not last_step or server.job_config.video_exec_step != -1:
                    video_noise_pred = backend._data_seq_to_patch(
                        server.job_config.patch_size,
                        video_noise_pred,
                        server.job_config.frame_chunk_size,
                        server.latent_height,
                        server.latent_width,
                        batch_size=backend._cfg_batch_size(batch_size),
                    )
                    if server.job_config.guidance_scale > 1:
                        video_noise_pred = (
                            video_noise_pred[batch_size:]
                            + server.job_config.guidance_scale
                            * (video_noise_pred[:batch_size] - video_noise_pred[batch_size:])
                        )
                    else:
                        video_noise_pred = video_noise_pred[:batch_size]
                    latents = server.scheduler.step(
                        video_noise_pred, timestep, latents, return_dict=False
                    )
                if latent_cond is not None:
                    latents[:, :, 0:1] = latent_cond

            for step_idx, timestep in enumerate(action_timesteps):
                last_step = step_idx == len(action_timesteps) - 1
                action_cond = (
                    torch.zeros(
                        [
                            batch_size,
                            server.job_config.action_dim,
                            1,
                            server.action_per_frame,
                            1,
                        ],
                        device=server.device,
                        dtype=server.dtype,
                    )
                    if frame_st_id == 0
                    else None
                )
                input_dict = backend._prepare_batch_input(
                    latent_model_input=None,
                    action_model_input=actions,
                    latent_t=float(timestep),
                    action_t=float(timestep),
                    latent_cond=None,
                    action_cond=action_cond,
                    frame_st_id=frame_st_id,
                )
                action_noise_pred = server.transformer(
                    input_dict["action_res_lst"],
                    update_cache=1 if last_step else 0,
                    cache_name=server.cache_name,
                    action_mode=True,
                )
                if not last_step:
                    action_noise_pred = (
                        action_noise_pred.unflatten(
                            1,
                            (
                                server.job_config.frame_chunk_size,
                                server.action_per_frame,
                            ),
                        )
                        .permute(0, 3, 1, 2)
                        .unsqueeze(-1)
                    )
                    if server.job_config.action_guidance_scale > 1:
                        action_noise_pred = (
                            action_noise_pred[batch_size:]
                            + server.job_config.action_guidance_scale
                            * (action_noise_pred[:batch_size] - action_noise_pred[batch_size:])
                        )
                    else:
                        action_noise_pred = action_noise_pred[:batch_size]
                    actions = server.action_scheduler.step(
                        action_noise_pred, timestep, actions, return_dict=False
                    )
                if action_cond is not None:
                    actions[:, :, 0:1] = action_cond

        actions[:, ~server.action_mask] *= 0
        # Store latents for decoding later (CPU, fp32 to save GPU memory).
        backend._captured_latents.append(latents.detach().cpu().float())
        torch.cuda.empty_cache()
        return backend._postprocess_action_batch(actions)

    backend._captured_latents = []
    backend._infer_batch_impl = _patched_impl


def _decode_imagination_latents(backend, out_dir: Path, num_envs: int, fps: int = 10):
    """Decode captured latents and write per-env imagination videos."""
    server = backend._server

    latents_mean = (
        torch.tensor(server.vae.config.latents_mean)
        .view(1, server.vae.config.z_dim, 1, 1, 1)
    )
    latents_std_inv = 1.0 / torch.tensor(server.vae.config.latents_std).view(
        1, server.vae.config.z_dim, 1, 1, 1
    )

    vae = server.vae
    vae_device = next(vae.parameters()).device

    # captured_latents: list of [B, 48, FC, h, w] tensors, one per chunk.
    # Concatenate along the time dim and split per env.
    if not backend._captured_latents:
        print("[video] No imagination latents captured — skipping.", flush=True)
        return

    all_latents = torch.cat(backend._captured_latents, dim=2)  # [B, 48, T_total, h, w]

    num_cams = len(server.job_config.obs_cam_keys)
    W_total = all_latents.shape[-1]
    W_per_cam = W_total // num_cams

    for env_idx in range(num_envs):
        lat = all_latents[env_idx : env_idx + 1].to(vae_device).to(vae.dtype)
        # Denormalize: cast mean/std to vae.dtype so arithmetic stays in bf16,
        # matching wan_va_server.decode_one_video which does .to(latents.dtype).
        lm = latents_mean.to(vae_device).to(vae.dtype)
        ls = latents_std_inv.to(vae_device).to(vae.dtype)
        lat = lat / ls + lm
        with torch.no_grad():
            video = vae.decode(lat, return_dict=False)[0]  # [1, C, T, H, W]
        # Convert to uint8 [T, H, W_total, 3]
        video = video[0].permute(1, 2, 3, 0)  # [T, H, W_total, C]
        video = ((video.float().clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).cpu().numpy()

        out_path = out_dir / f"imagination_env{env_idx}.mp4"
        imageio.mimwrite(str(out_path), video, fps=fps, codec="libx264", quality=8)
        print(f"[video] saved imagination video: {out_path} ({len(video)} frames)", flush=True)


# ---------------------------------------------------------------------------
# Observation video helpers
# ---------------------------------------------------------------------------

def _collect_obs_frame(raw_obs: dict) -> np.ndarray | None:
    """Extract agentview + wrist and concatenate horizontally (H, 2W, 3)."""
    agentview = raw_obs.get("agentview_image")
    wrist = raw_obs.get("robot0_eye_in_hand_image")
    if agentview is None:
        return None
    if wrist is None:
        return np.asarray(agentview)
    # Both are (H, W, 3) uint8; concatenate side-by-side.
    return np.concatenate([np.asarray(agentview), np.asarray(wrist)], axis=1)


def _save_obs_video(frames: list[np.ndarray], out_path: Path, fps: int = 10):
    if not frames:
        print(f"[video] No observation frames to save for {out_path}", flush=True)
        return
    imageio.mimwrite(str(out_path), frames, fps=fps, codec="libx264", quality=8)
    print(f"[video] saved observation video: {out_path} ({len(frames)} frames)", flush=True)


# ---------------------------------------------------------------------------
# Environment / model helpers (copied from eval_lingbotva.py)
# ---------------------------------------------------------------------------

def _build_libero_env(cfg: DictConfig) -> LiberoEnv:
    eval_cfg = cfg.env.eval
    return LiberoEnv(
        cfg=eval_cfg,
        num_envs=int(eval_cfg.total_num_envs),
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )


def _chunk_step_with_obs(env: LiberoEnv, chunk_actions: np.ndarray):
    num_envs, chunk_size, _ = chunk_actions.shape
    raw_obs_history: list[list[dict]] = [[] for _ in range(num_envs)]
    obs_list = []
    rewards = []
    terms = []
    truncs = []
    for i in range(chunk_size):
        action = chunk_actions[:, i]
        wrapped, r, t, tr, _ = env.step(action, auto_reset=False)
        for env_idx in range(num_envs):
            raw_obs_history[env_idx].append(env.current_raw_obs[env_idx])
        obs_list.append(wrapped)
        rewards.append(r)
        terms.append(t)
        truncs.append(tr)
    rewards = torch.stack(rewards, dim=1)
    terms = torch.stack(terms, dim=1)
    truncs = torch.stack(truncs, dim=1)
    past_dones = (terms | truncs).any(dim=1)
    if past_dones.any() and env.auto_reset:
        obs_list[-1], _ = env._handle_auto_reset(
            past_dones.cpu().numpy(), obs_list[-1], {}
        )
    return obs_list[-1], rewards, terms, truncs, raw_obs_history, past_dones


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def _evaluate_with_video(
    model: torch.nn.Module,
    env: LiberoEnv,
    out_dir: Path,
    fps: int = 10,
) -> dict:
    num_envs = env.num_envs
    obs_frames: list[list[np.ndarray]] = [[] for _ in range(num_envs)]

    print("[eval] resetting env...", flush=True)
    obs, _ = env.reset()

    # Collect first frame from each env.
    for env_idx in range(num_envs):
        if env.current_raw_obs is not None and env.current_raw_obs[env_idx] is not None:
            frame = _collect_obs_frame(env.current_raw_obs[env_idx])
            if frame is not None:
                obs_frames[env_idx].append(frame)

    enable_kv_replay = getattr(model, "enable_kv_cache_replay", False)
    counted = [False] * num_envs
    successes = 0
    failures = 0
    total_episodes_done = 0
    task_stats: dict[int, dict[str, int]] = {}
    cur_episode_task_ids = list(env.task_ids.copy())
    chunk_idx = 0
    max_steps = num_envs * 260

    total_steps = 0
    while not all(counted) and total_steps < max_steps:
        t0 = time.time()
        action_tensor, _ = model.predict_action_batch(obs, mode="eval")
        infer_sec = time.time() - t0

        t1 = time.time()
        chunk_actions = action_tensor.detach().cpu().numpy()
        (
            final_obs,
            rewards,
            chunk_terminations,
            chunk_truncations,
            raw_obs_history,
            past_dones,
        ) = _chunk_step_with_obs(env, chunk_actions)
        step_sec = time.time() - t1

        # Accumulate observation frames.
        for env_idx in range(num_envs):
            if not counted[env_idx]:
                for raw_obs in raw_obs_history[env_idx]:
                    frame = _collect_obs_frame(raw_obs)
                    if frame is not None:
                        obs_frames[env_idx].append(frame)

        successes_this_chunk = chunk_terminations.any(dim=1)
        for env_idx in range(past_dones.shape[0]):
            if past_dones[env_idx].item():
                if not counted[env_idx]:
                    counted[env_idx] = True
                    task_id = int(cur_episode_task_ids[env_idx])
                    stats = task_stats.setdefault(task_id, {"success": 0, "total": 0})
                    stats["total"] += 1
                    if successes_this_chunk[env_idx].item():
                        stats["success"] += 1
                        successes += 1
                    else:
                        failures += 1
                    total_episodes_done += 1
                cur_episode_task_ids[env_idx] = int(env.task_ids[env_idx])
                if hasattr(model, "reset_episode"):
                    model.reset_episode(env_idx)
            elif enable_kv_replay and not counted[env_idx]:
                state = model._episode_states.get(env_idx)
                if state is not None and state.prev_model_action is not None:
                    model.record_chunk_observations(
                        env_idx=env_idx,
                        chunk_obs_list=raw_obs_history[env_idx],
                        prev_model_action=state.prev_model_action,
                    )

        obs = final_obs
        total_steps += chunk_actions.shape[1]
        chunk_idx += 1
        print(
            f"  chunk={chunk_idx:3d} step={total_steps:4d} "
            f"infer={infer_sec:6.2f}s env={step_sec:5.2f}s "
            f"episodes={total_episodes_done} succ={successes} fail={failures}",
            flush=True,
        )

    # Save observation videos.
    for env_idx in range(num_envs):
        _save_obs_video(
            obs_frames[env_idx],
            out_dir / f"observations_env{env_idx}.mp4",
            fps=fps,
        )

    # Decode and save imagination videos.
    backend = model._ensure_backend() if hasattr(model, "_ensure_backend") else None
    if backend is not None:
        _decode_imagination_latents(backend, out_dir, num_envs=num_envs, fps=fps)

    total = successes + failures
    return {
        "success_rate": successes / total if total > 0 else 0.0,
        "successes": successes,
        "failures": failures,
        "task_stats": {
            tid: {
                **stats,
                "rate": stats["success"] / stats["total"] if stats["total"] else 0.0,
            }
            for tid, stats in task_stats.items()
        },
    }


# ---------------------------------------------------------------------------
# Hydra entry
# ---------------------------------------------------------------------------

@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="libero_object_eval_lingbotva",
)
def main(cfg: DictConfig) -> None:
    if str(cfg.actor.model.model_type) != "lingbotva":
        raise ValueError(
            f"eval_step170_task5_video.py expects actor.model.model_type=lingbotva, "
            f"got {cfg.actor.model.model_type!r}."
        )

    cfg.runner.only_eval = True
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    out_dir = Path(cfg.runner.logger.log_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = _build_libero_env(cfg)
    model = get_model(cfg.actor.model)
    if model is None:
        raise RuntimeError(f"Failed to build model of type {cfg.actor.model.model_type}")

    # Initialise the eval backend eagerly (no inference needed) then patch it
    # so _infer_batch_impl captures video latents before the eval loop starts.
    print("[eval] initialising backend...", flush=True)
    backend = model._ensure_backend()
    _patch_backend_for_latent_capture(backend)
    print("[eval] backend ready, patch applied.", flush=True)

    t0 = time.time()
    metrics = _evaluate_with_video(model, env, out_dir)
    metrics["elapsed_sec"] = time.time() - t0
    metrics["num_envs"] = env.num_envs

    (out_dir / "eval_results.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"[eval] outputs saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

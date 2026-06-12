"""
Evaluate step-170 checkpoint on all 10 LIBERO-Object tasks, 5 episodes each.

For each episode, saves a combined MP4:
  top row    — actual LIBERO observations (agentview | wrist), per step
  bottom row — model imagination decoded from diffusion latents, per chunk

Text labels are overlaid in the top-left corner of each row.

Outputs (under OUT_DIR):
  task{T}_{short_name}/ep{N}.mp4   — combined video, T=0..9, N=0..4
  index.html                        — gallery page
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_NAMES = [
    "alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce", "ketchup",
    "tomato_sauce", "butter", "milk", "choc_pudding", "orange_juice",
]

FONT_PATH = "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf"
FONT_SIZE = 11

OUT_DIR = Path("/workspace/rl_logs/eval_all_tasks_combined")
FPS = 10

# ---------------------------------------------------------------------------
# Text overlay
# ---------------------------------------------------------------------------

_font = ImageFont.truetype(FONT_PATH, FONT_SIZE)


def _add_label(frame: np.ndarray, text: str, color=(220, 220, 80)) -> np.ndarray:
    """Draw text with a dark outline into the top-left of the frame (in-place copy)."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    x, y = 3, 2
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=_font)
    draw.text((x, y), text, fill=color, font=_font)
    return np.array(img)


# ---------------------------------------------------------------------------
# Imagination capture patch
# ---------------------------------------------------------------------------

def _patch_backend(backend):
    """Wrap _infer_batch_impl to stash the final video latents each chunk."""
    original = backend._infer_batch_impl

    def _patched(obs_batch, *, frame_st_id=0):
        server = backend._server
        obs_sequences = backend._normalize_obs_sequences(obs_batch)
        batch_size = len(obs_sequences)

        if frame_st_id == 0 and server.init_latent is None:
            server.init_latent = backend._encode_obs_batch(obs_sequences)

        latents = torch.randn(
            batch_size, 48,
            server.job_config.frame_chunk_size,
            server.latent_height, server.latent_width,
            device=server.device, dtype=server.dtype,
        )
        actions = torch.randn(
            batch_size, server.job_config.action_dim,
            server.job_config.frame_chunk_size,
            server.action_per_frame, 1,
            device=server.device, dtype=server.dtype,
        )

        server.scheduler.set_timesteps(server.job_config.num_inference_steps)
        server.action_scheduler.set_timesteps(server.job_config.action_num_inference_steps)
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
                    latent_model_input=latents, action_model_input=None,
                    latent_t=float(timestep), action_t=float(timestep),
                    latent_cond=latent_cond, action_cond=None,
                    frame_st_id=frame_st_id,
                )
                video_noise_pred = server.transformer(
                    input_dict["latent_res_lst"],
                    update_cache=1 if last_step else 0,
                    cache_name=server.cache_name, action_mode=False,
                )
                if not last_step or server.job_config.video_exec_step != -1:
                    video_noise_pred = backend._data_seq_to_patch(
                        server.job_config.patch_size, video_noise_pred,
                        server.job_config.frame_chunk_size,
                        server.latent_height, server.latent_width,
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
                        [batch_size, server.job_config.action_dim, 1,
                         server.action_per_frame, 1],
                        device=server.device, dtype=server.dtype,
                    ) if frame_st_id == 0 else None
                )
                input_dict = backend._prepare_batch_input(
                    latent_model_input=None, action_model_input=actions,
                    latent_t=float(timestep), action_t=float(timestep),
                    latent_cond=None, action_cond=action_cond,
                    frame_st_id=frame_st_id,
                )
                action_noise_pred = server.transformer(
                    input_dict["action_res_lst"],
                    update_cache=1 if last_step else 0,
                    cache_name=server.cache_name, action_mode=True,
                )
                if not last_step:
                    action_noise_pred = (
                        action_noise_pred.unflatten(
                            1, (server.job_config.frame_chunk_size, server.action_per_frame)
                        ).permute(0, 3, 1, 2).unsqueeze(-1)
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
        backend._captured_latents.append(latents.detach().cpu().float())
        torch.cuda.empty_cache()
        return backend._postprocess_action_batch(actions)

    backend._captured_latents = []
    backend._infer_batch_impl = _patched


# ---------------------------------------------------------------------------
# Imagination decode
# ---------------------------------------------------------------------------

def _decode_latents(backend, num_envs: int) -> list[np.ndarray]:
    """Decode captured latents → list of per-env uint8 pixel arrays [T, H, W, 3]."""
    if not backend._captured_latents:
        return [np.zeros((1, 128, 256, 3), dtype=np.uint8)] * num_envs

    server = backend._server
    vae = server.vae
    vae_device = next(vae.parameters()).device

    latents_mean = (
        torch.tensor(server.vae.config.latents_mean)
        .view(1, server.vae.config.z_dim, 1, 1, 1)
    )
    latents_std_inv = (
        1.0 / torch.tensor(server.vae.config.latents_std)
        .view(1, server.vae.config.z_dim, 1, 1, 1)
    )

    # all_latents: [B, 48, T_total, h, w]
    all_latents = torch.cat(backend._captured_latents, dim=2)

    results = []
    for env_idx in range(num_envs):
        lat = all_latents[env_idx : env_idx + 1].to(vae_device).to(vae.dtype)
        lm = latents_mean.to(vae_device).to(vae.dtype)
        ls = latents_std_inv.to(vae_device).to(vae.dtype)
        lat = lat / ls + lm
        with torch.no_grad():
            decoded = vae.decode(lat, return_dict=False)[0]  # [1, C, T, H, W]
        frames = decoded[0].permute(1, 2, 3, 0)  # [T, H, W, C]
        frames = ((frames.float().clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).cpu().numpy()
        results.append(frames)
    return results


# ---------------------------------------------------------------------------
# Observation frame collection
# ---------------------------------------------------------------------------

def _obs_frame(raw_obs: dict) -> np.ndarray | None:
    """Return agentview | wrist side-by-side [H, 2W, 3]."""
    av = raw_obs.get("agentview_image")
    wr = raw_obs.get("robot0_eye_in_hand_image")
    if av is None:
        return None
    av = np.asarray(av)
    if wr is None:
        return av
    return np.concatenate([av, np.asarray(wr)], axis=1)


# ---------------------------------------------------------------------------
# Combined video builder
# ---------------------------------------------------------------------------

def _build_combined(
    obs_frames: list[np.ndarray],
    imag_frames: np.ndarray,
    obs_label: str,
    imag_label: str,
) -> list[np.ndarray]:
    """Stack obs (top) and imagination (bottom), aligned by time."""
    N = len(obs_frames)
    M = len(imag_frames)

    if M > 0:
        indices = np.round(np.linspace(0, M - 1, N)).astype(int)
        imag_aligned = imag_frames[indices]
    else:
        h, w = obs_frames[0].shape[:2]
        imag_aligned = np.zeros((N, h, w, 3), dtype=np.uint8)

    combined = []
    for obs_f, imag_f in zip(obs_frames, imag_aligned):
        # Ensure both rows are the same width (pad or crop if needed).
        w = max(obs_f.shape[1], imag_f.shape[1])
        if obs_f.shape[1] < w:
            obs_f = np.pad(obs_f, ((0, 0), (0, w - obs_f.shape[1]), (0, 0)))
        if imag_f.shape[1] < w:
            imag_f = np.pad(imag_f, ((0, 0), (0, w - imag_f.shape[1]), (0, 0)))

        obs_f = _add_label(obs_f, obs_label, color=(100, 220, 100))
        imag_f = _add_label(imag_f, imag_label, color=(100, 180, 255))

        # thin divider between rows
        divider = np.full((2, w, 3), 60, dtype=np.uint8)
        combined.append(np.concatenate([obs_f, divider, imag_f], axis=0))

    return combined


# ---------------------------------------------------------------------------
# Env step helper
# ---------------------------------------------------------------------------

def _chunk_step_with_obs(env, chunk_actions: np.ndarray):
    from rlinf.envs.libero.libero_env import LiberoEnv  # type: ignore
    num_envs, chunk_size, _ = chunk_actions.shape
    raw_obs_history: list[list[dict]] = [[] for _ in range(num_envs)]
    obs_list, rewards, terms, truncs = [], [], [], []
    for i in range(chunk_size):
        wrapped, r, t, tr, _ = env.step(chunk_actions[:, i], auto_reset=False)
        for ei in range(num_envs):
            raw_obs_history[ei].append(env.current_raw_obs[ei])
        obs_list.append(wrapped)
        rewards.append(r)
        terms.append(t)
        truncs.append(tr)
    rewards = torch.stack(rewards, dim=1)
    terms = torch.stack(terms, dim=1)
    truncs = torch.stack(truncs, dim=1)
    past_dones = (terms | truncs).any(dim=1)
    return obs_list[-1], rewards, terms, truncs, raw_obs_history, past_dones


# ---------------------------------------------------------------------------
# Per-task eval
# ---------------------------------------------------------------------------

def _run_single_episode(model, eval_cfg, task_id: int, ep_idx: int, num_eps: int):
    """Run exactly one episode (1 env) and return (obs_frames, imag_frames, success)."""
    from rlinf.envs.libero.libero_env import LiberoEnv
    from omegaconf import open_dict

    with open_dict(eval_cfg):
        eval_cfg.total_num_envs = 1
        eval_cfg.task_id_filter = [task_id]
        # Advance the ordered reset pointer so each episode sees a different init state.
        eval_cfg.eval_reset_start_idx = ep_idx

    env = LiberoEnv(cfg=eval_cfg, num_envs=1, seed_offset=0,
                    total_num_processes=1, worker_info=None)

    backend = model._ensure_backend()
    backend._captured_latents.clear()
    if hasattr(model, "reset_episode"):
        model.reset_episode(0)

    obs_frames: list[np.ndarray] = []
    obs, _ = env.reset()
    if env.current_raw_obs and env.current_raw_obs[0]:
        f = _obs_frame(env.current_raw_obs[0])
        if f is not None:
            obs_frames.append(f)

    enable_kv = getattr(model, "enable_kv_cache_replay", False)
    done = False
    success = False
    chunk_idx = 0
    total_steps = 0

    while not done and total_steps < 260:
        action_tensor, _ = model.predict_action_batch(obs, mode="eval")
        chunk_actions = action_tensor.detach().cpu().numpy()
        final_obs, _, chunk_terms, _, raw_obs_history, past_dones = (
            _chunk_step_with_obs(env, chunk_actions)
        )
        successes_this_chunk = chunk_terms.any(dim=1)

        for raw_obs in raw_obs_history[0]:
            f = _obs_frame(raw_obs)
            if f is not None:
                obs_frames.append(f)

        if past_dones[0].item():
            done = True
            success = bool(successes_this_chunk[0].item())
            if hasattr(model, "reset_episode"):
                model.reset_episode(0)
        elif enable_kv:
            state = model._episode_states.get(0)
            if state is not None and state.prev_model_action is not None:
                model.record_chunk_observations(
                    env_idx=0,
                    chunk_obs_list=raw_obs_history[0],
                    prev_model_action=state.prev_model_action,
                )

        obs = final_obs
        total_steps += chunk_actions.shape[1]
        chunk_idx += 1
        print(
            f"  [t{task_id} ep{ep_idx}] chunk={chunk_idx:3d} "
            f"step={total_steps:4d} {'DONE' if done else '...'}",
            flush=True,
        )

    # Decode imagination for the single env.
    imag_list = _decode_latents(backend, num_envs=1)
    imag_frames = imag_list[0]
    backend._captured_latents.clear()

    env.env.close()
    return obs_frames, imag_frames, success


def _eval_task(model, task_id: int, num_eps: int, task_dir: Path):
    from omegaconf import open_dict

    print(f"\n{'='*60}", flush=True)
    print(f"[task {task_id}] {TASK_NAMES[task_id]}  ({num_eps} episodes)", flush=True)

    cfg = _GLOBAL_CFG
    # Work on a mutable copy of the eval cfg so changes don't bleed across tasks.
    from omegaconf import OmegaConf
    eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env.eval, resolve=True))

    results = []
    for ep_idx in range(num_eps):
        print(f"\n  [t{task_id}] --- episode {ep_idx+1}/{num_eps} ---", flush=True)
        obs_frames, imag_frames, success = _run_single_episode(
            model, eval_cfg, task_id=task_id, ep_idx=ep_idx, num_eps=num_eps
        )

        label_obs = (
            f"OBS | t{task_id}: {TASK_NAMES[task_id]} | ep{ep_idx+1}/{num_eps}"
            f" | {'SUCCESS' if success else 'FAIL'}"
        )
        label_imag = (
            f"IMAGINATION | t{task_id}: {TASK_NAMES[task_id]} | ep{ep_idx+1}/{num_eps}"
        )
        combined = _build_combined(obs_frames, imag_frames, label_obs, label_imag)
        out_path = task_dir / f"ep{ep_idx}.mp4"
        imageio.mimwrite(str(out_path), combined, fps=FPS, codec="libx264", quality=8)
        print(
            f"  [t{task_id}] ep{ep_idx}: {'OK' if success else 'FAIL'}"
            f" -> {out_path} ({len(combined)} frames)",
            flush=True,
        )
        results.append({"ep": ep_idx, "success": success, "frames": len(combined)})

    # Write per-task HTML stub so the gallery updates as tasks complete.
    _write_html({task_id: results, **{
        t: json.loads((OUT_DIR / f"task{t}_{TASK_NAMES[t]}" / "results.json").read_text())
        for t in range(task_id)
        if (OUT_DIR / f"task{t}_{TASK_NAMES[t]}" / "results.json").exists()
    }}, OUT_DIR)

    return results


# ---------------------------------------------------------------------------
# HTML gallery
# ---------------------------------------------------------------------------

def _write_html(all_results: dict[int, list[dict]], out_dir: Path):
    rows = []
    for task_id in range(10):
        name = TASK_NAMES[task_id]
        task_dir_rel = f"task{task_id}_{name}"
        eps = all_results.get(task_id, [])
        cells = ""
        for ep_data in eps:
            ei = ep_data["ep"]
            ok = ep_data["success"]
            badge = (
                '<span style="color:#6f6;font-weight:bold">✓ OK</span>'
                if ok else
                '<span style="color:#f66;font-weight:bold">✗ FAIL</span>'
            )
            cells += f"""
            <td style="padding:4px;text-align:center">
              <div style="font-size:11px;margin-bottom:2px">ep{ei+1} {badge}</div>
              <video controls width="260" style="display:block;border:1px solid #444">
                <source src="{task_dir_rel}/ep{ei}.mp4" type="video/mp4">
              </video>
            </td>"""
        rows.append(f"""
        <tr>
          <td style="padding:6px 10px;white-space:nowrap;font-weight:bold;
                     color:#adf;border-right:1px solid #333;vertical-align:top">
            t{task_id}<br><span style="font-size:11px;font-weight:normal;color:#888">{name}</span>
          </td>
          {cells}
        </tr>""")

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>LingBot-VA Step 170 — All Tasks</title>
<style>
  body {{ font-family: monospace; background: #111; color: #eee; padding: 20px; }}
  h1 {{ color: #7cf; margin-bottom: 4px; }}
  p.sub {{ color: #888; font-size: 13px; margin-top: 0; }}
  table {{ border-collapse: collapse; }}
  tr:nth-child(even) {{ background: #181818; }}
  tr:hover {{ background: #1e2a1e; }}
  th {{ color: #7cf; padding: 4px 8px; border-bottom: 1px solid #333; }}
  video {{ background: #000; }}
</style>
</head>
<body>
<h1>LingBot-VA — Step 170 · All 10 Tasks · 5 Episodes Each</h1>
<p class="sub">Top row: actual LIBERO observation (agentview | wrist).
Bottom row: model imagination decoded from diffusion latents.</p>
<table>
  <thead>
    <tr>
      <th style="text-align:left">Task</th>
      {''.join(f'<th>ep{i+1}</th>' for i in range(5))}
    </tr>
  </thead>
  <tbody>
    {''.join(rows)}
  </tbody>
</table>
</body>
</html>"""
    (out_dir / "index.html").write_text(html)
    print(f"[html] gallery written -> {out_dir / 'index.html'}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_GLOBAL_CFG: DictConfig | None = None


def main():
    global _GLOBAL_CFG

    import sys
    import os

    repo_path = os.environ.get("REPO_PATH", "/workspace/RLinf")
    sys.path.insert(0, repo_path)
    sys.path.insert(0, os.environ.get("LINGBOT_VA_REPO_PATH", "/workspace/lingbot-va"))
    sys.path.insert(0, os.environ.get("LIBERO_PATH", "/workspace/LIBERO"))

    # Bootstrap Hydra config the same way the eval scripts do.
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    config_dir = str(Path(repo_path) / "examples/embodiment/config")
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(
            config_name="libero_object_eval_lingbotva",
            overrides=[
                f"env.eval.total_num_envs=5",
                "algorithm.eval_rollout_epoch=1",
                "env.eval.video_cfg.save_video=False",
                f"actor.model.lingbotva.save_root={OUT_DIR}/runtime",
                f"runner.logger.log_path={OUT_DIR}",
            ],
        )
    _GLOBAL_CFG = cfg
    cfg.runner.only_eval = True

    from rlinf.models import get_model
    print("[init] loading model...", flush=True)
    model = get_model(cfg.actor.model)
    if model is None:
        raise RuntimeError("Failed to build model")

    # Eager backend init + patch.
    print("[init] initialising backend...", flush=True)
    backend = model._ensure_backend()
    _patch_backend(backend)
    print("[init] ready.", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[int, list[dict]] = {}

    for task_id in range(10):
        name = TASK_NAMES[task_id]
        task_dir = OUT_DIR / f"task{task_id}_{name}"
        task_dir.mkdir(parents=True, exist_ok=True)
        results = _eval_task(model, task_id=task_id, num_eps=5, task_dir=task_dir)
        all_results[task_id] = results
        (task_dir / "results.json").write_text(json.dumps(results, indent=2))

    _write_html(all_results, OUT_DIR)
    (OUT_DIR / "all_results.json").write_text(json.dumps(all_results, indent=2))
    print("\n[done] all tasks complete.", flush=True)


if __name__ == "__main__":
    main()

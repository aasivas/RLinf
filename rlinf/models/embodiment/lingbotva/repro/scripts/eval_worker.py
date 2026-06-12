"""Single-episode worker for parallel video eval.

Usage:
    CUDA_VISIBLE_DEVICES=<gpu> python eval_worker.py <task_id> <ep_idx> <out_dir>

Loads the step-170 checkpoint, runs one episode for the given task/ep, and
writes <out_dir>/task<T>_<name>/ep<N>.mp4 (combined obs+imagination video).
Also writes ep<N>.json with {"success": bool}.

Exit code 0 on success, 1 on failure.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf, open_dict

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

REPO_PATH = os.environ.get("REPO_PATH", "/workspace/RLinf")
LINGBOT_VA_REPO_PATH = os.environ.get("LINGBOT_VA_REPO_PATH", "/workspace/lingbot-va")
LIBERO_PATH = os.environ.get("LIBERO_PATH", "/workspace/LIBERO")
CKPT_RL = "/workspace/rl_logs/grpo_exact_train_logs/libero_object_grpo_lingbotva/checkpoints/global_step_170/actor/model_state_dict/full_weights.pt"
CKPT_SFT = "sft"  # sentinel: use base_sft model path, no state-dict override

for p in [REPO_PATH, LINGBOT_VA_REPO_PATH, LIBERO_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

TASK_NAMES = [
    "alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce", "ketchup",
    "tomato_sauce", "butter", "milk", "choc_pudding", "orange_juice",
]
FONT = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", 11)
FPS = 10

# ---------------------------------------------------------------------------
# Text overlay
# ---------------------------------------------------------------------------

def _label(frame: np.ndarray, text: str, color=(220, 220, 80)) -> np.ndarray:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        d.text((3 + dx, 2 + dy), text, fill=(0, 0, 0), font=FONT)
    d.text((3, 2), text, fill=color, font=FONT)
    return np.array(img)


# ---------------------------------------------------------------------------
# Imagination capture
# ---------------------------------------------------------------------------

def _patch_backend(backend):
    original = backend._infer_batch_impl

    def _patched(obs_batch, *, frame_st_id=0):
        server = backend._server
        obs_seqs = backend._normalize_obs_sequences(obs_batch)
        B = len(obs_seqs)
        if frame_st_id == 0 and server.init_latent is None:
            server.init_latent = backend._encode_obs_batch(obs_seqs)

        latents = torch.randn(B, 48, server.job_config.frame_chunk_size,
                              server.latent_height, server.latent_width,
                              device=server.device, dtype=server.dtype)
        actions = torch.randn(B, server.job_config.action_dim,
                              server.job_config.frame_chunk_size,
                              server.action_per_frame, 1,
                              device=server.device, dtype=server.dtype)

        server.scheduler.set_timesteps(server.job_config.num_inference_steps)
        server.action_scheduler.set_timesteps(server.job_config.action_num_inference_steps)
        ts = torch.nn.functional.pad(server.scheduler.timesteps, (0, 1), value=0)
        if server.job_config.video_exec_step != -1:
            ts = ts[:server.job_config.video_exec_step]
        ats = torch.nn.functional.pad(server.action_scheduler.timesteps, (0, 1), value=0)

        with torch.no_grad():
            for i, t in enumerate(ts):
                last = i == len(ts) - 1
                lc = server.init_latent[:, :, 0:1] if frame_st_id == 0 else None
                inp = backend._prepare_batch_input(
                    latent_model_input=latents, action_model_input=None,
                    latent_t=float(t), action_t=float(t),
                    latent_cond=lc, action_cond=None, frame_st_id=frame_st_id)
                vpred = server.transformer(inp["latent_res_lst"],
                                           update_cache=1 if last else 0,
                                           cache_name=server.cache_name, action_mode=False)
                if not last or server.job_config.video_exec_step != -1:
                    vpred = backend._data_seq_to_patch(
                        server.job_config.patch_size, vpred,
                        server.job_config.frame_chunk_size,
                        server.latent_height, server.latent_width,
                        batch_size=backend._cfg_batch_size(B))
                    gs = server.job_config.guidance_scale
                    vpred = vpred[B:] + gs * (vpred[:B] - vpred[B:]) if gs > 1 else vpred[:B]
                    latents = server.scheduler.step(vpred, t, latents, return_dict=False)
                if lc is not None:
                    latents[:, :, 0:1] = lc

            for i, t in enumerate(ats):
                last = i == len(ats) - 1
                ac = (torch.zeros([B, server.job_config.action_dim, 1,
                                   server.action_per_frame, 1],
                                  device=server.device, dtype=server.dtype)
                      if frame_st_id == 0 else None)
                inp = backend._prepare_batch_input(
                    latent_model_input=None, action_model_input=actions,
                    latent_t=float(t), action_t=float(t),
                    latent_cond=None, action_cond=ac, frame_st_id=frame_st_id)
                apred = server.transformer(inp["action_res_lst"],
                                           update_cache=1 if last else 0,
                                           cache_name=server.cache_name, action_mode=True)
                if not last:
                    apred = (apred.unflatten(1, (server.job_config.frame_chunk_size,
                                                  server.action_per_frame))
                             .permute(0, 3, 1, 2).unsqueeze(-1))
                    ags = server.job_config.action_guidance_scale
                    apred = apred[B:] + ags * (apred[:B] - apred[B:]) if ags > 1 else apred[:B]
                    actions = server.action_scheduler.step(apred, t, actions, return_dict=False)
                if ac is not None:
                    actions[:, :, 0:1] = ac

        actions[:, ~server.action_mask] *= 0
        backend._captured_latents.append(latents.detach().cpu().float())
        torch.cuda.empty_cache()
        return backend._postprocess_action_batch(actions)

    backend._captured_latents = []
    backend._infer_batch_impl = _patched


def _decode_imagination(backend) -> np.ndarray:
    """Decode captured latents → uint8 [T, H, W, 3]."""
    if not backend._captured_latents:
        return np.zeros((1, 128, 256, 3), dtype=np.uint8)
    server = backend._server
    vae = server.vae
    dev = next(vae.parameters()).device
    lm = torch.tensor(server.vae.config.latents_mean).view(1, server.vae.config.z_dim, 1, 1, 1)
    ls = 1.0 / torch.tensor(server.vae.config.latents_std).view(1, server.vae.config.z_dim, 1, 1, 1)
    all_lat = torch.cat(backend._captured_latents, dim=2)  # [1, 48, T, h, w]
    lat = all_lat[0:1].to(dev).to(vae.dtype)
    lat = lat / ls.to(dev, vae.dtype) + lm.to(dev, vae.dtype)
    with torch.no_grad():
        decoded = vae.decode(lat, return_dict=False)[0]  # [1, C, T, H, W]
    frames = decoded[0].permute(1, 2, 3, 0)
    frames = ((frames.float().clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).cpu().numpy()
    backend._captured_latents.clear()
    return frames


# ---------------------------------------------------------------------------
# Env step helpers
# ---------------------------------------------------------------------------

def _obs_frame(raw_obs: dict) -> np.ndarray | None:
    av = raw_obs.get("agentview_image")
    wr = raw_obs.get("robot0_eye_in_hand_image")
    if av is None:
        return None
    av = np.ascontiguousarray(np.asarray(av)[::-1])
    if wr is not None:
        wr = np.ascontiguousarray(np.asarray(wr)[::-1])
        return np.concatenate([av, wr], axis=1)
    return av


def _chunk_step(env, chunk_actions):
    num_envs, chunk_size, _ = chunk_actions.shape
    raw_hist = [[] for _ in range(num_envs)]
    obs_list, R, T, Tr = [], [], [], []
    for i in range(chunk_size):
        w, r, t, tr, _ = env.step(chunk_actions[:, i], auto_reset=False)
        for ei in range(num_envs):
            raw_hist[ei].append(env.current_raw_obs[ei])
        obs_list.append(w); R.append(r); T.append(t); Tr.append(tr)
    terms = torch.stack(T, dim=1)
    truncs = torch.stack(Tr, dim=1)
    past_dones = (terms | truncs).any(dim=1)
    return obs_list[-1], terms, raw_hist, past_dones


# ---------------------------------------------------------------------------
# Combined video
# ---------------------------------------------------------------------------

def _combine(obs_frames, imag_frames, obs_label, imag_label) -> list[np.ndarray]:
    N = len(obs_frames)
    M = len(imag_frames)
    if M > 0:
        idxs = np.round(np.linspace(0, M - 1, N)).astype(int)
        imag = imag_frames[idxs]
    else:
        h, w = obs_frames[0].shape[:2]
        imag = np.zeros((N, h, w, 3), dtype=np.uint8)

    out = []
    for of, imf in zip(obs_frames, imag):
        W = max(of.shape[1], imf.shape[1])
        if of.shape[1] < W:
            of = np.pad(of, ((0, 0), (0, W - of.shape[1]), (0, 0)))
        if imf.shape[1] < W:
            imf = np.pad(imf, ((0, 0), (0, W - imf.shape[1]), (0, 0)))
        div = np.full((2, W, 3), 60, dtype=np.uint8)
        out.append(np.concatenate([
            _label(of, obs_label, color=(100, 220, 100)),
            div,
            _label(imf, imag_label, color=(100, 180, 255)),
        ], axis=0))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(task_id: int, ep_idx: int, out_dir: Path, ckpt: str = CKPT_RL):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from rlinf.envs.libero.libero_env import LiberoEnv
    from rlinf.models import get_model

    name = TASK_NAMES[task_id]
    tag = f"t{task_id}({name}) ep{ep_idx}"
    print(f"[worker] {tag} starting on {os.environ.get('CUDA_VISIBLE_DEVICES','?')}", flush=True)

    GlobalHydra.instance().clear()
    config_dir = str(Path(REPO_PATH) / "examples/embodiment/config")
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(
            config_name="libero_object_eval_lingbotva",
            overrides=[
                "env.eval.total_num_envs=1",
                "algorithm.eval_rollout_epoch=1",
                "env.eval.video_cfg.save_video=False",
                f"actor.model.lingbotva.save_root={out_dir}/runtime_{task_id}_{ep_idx}",
                f"runner.logger.log_path={out_dir}",
            ],
        )

    use_sft = (ckpt == CKPT_SFT)
    if use_sft:
        # SFT baseline: load transformer from base_sft directory, no RL override.
        os.environ.setdefault("LINGBOT_VA_MODEL_PATH", "/root/ckpts/base_sft")
        os.environ.pop("LINGBOT_VA_TRANSFORMER_STATE_DICT_PATH", None)
    else:
        os.environ.setdefault("LINGBOT_VA_MODEL_PATH",
                              os.environ.get("BASE_MODEL", "/root/ckpts/base"))
        os.environ.setdefault("LINGBOT_VA_TRANSFORMER_STATE_DICT_PATH", ckpt)

    # Build env.
    eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env.eval, resolve=True))
    with open_dict(eval_cfg):
        eval_cfg.total_num_envs = 1
        eval_cfg.task_id_filter = [task_id]
        eval_cfg.eval_reset_start_idx = ep_idx

    env = LiberoEnv(cfg=eval_cfg, num_envs=1, seed_offset=0,
                    total_num_processes=1, worker_info=None)

    # Build model + patch backend.
    model = get_model(cfg.actor.model)
    backend = model._ensure_backend()
    _patch_backend(backend)
    if hasattr(model, "reset_episode"):
        model.reset_episode(0)

    # Run episode.
    obs_frames: list[np.ndarray] = []
    obs, _ = env.reset()
    if env.current_raw_obs and env.current_raw_obs[0]:
        f = _obs_frame(env.current_raw_obs[0])
        if f is not None:
            obs_frames.append(f)

    enable_kv = getattr(model, "enable_kv_cache_replay", False)
    done = success = False
    steps = chunk_idx = 0

    while not done and steps < 260:
        action_tensor, _ = model.predict_action_batch(obs, mode="eval")
        chunk_actions = action_tensor.detach().cpu().numpy()
        final_obs, chunk_terms, raw_hist, past_dones = _chunk_step(env, chunk_actions)

        for raw_obs in raw_hist[0]:
            f = _obs_frame(raw_obs)
            if f is not None:
                obs_frames.append(f)

        if past_dones[0].item():
            done = True
            success = bool(chunk_terms.any(dim=1)[0].item())
            if hasattr(model, "reset_episode"):
                model.reset_episode(0)
        elif enable_kv:
            state = model._episode_states.get(0)
            if state is not None and state.prev_model_action is not None:
                model.record_chunk_observations(
                    env_idx=0,
                    chunk_obs_list=raw_hist[0],
                    prev_model_action=state.prev_model_action,
                )

        obs = final_obs
        steps += chunk_actions.shape[1]
        chunk_idx += 1
        print(f"  [{tag}] chunk={chunk_idx:2d} step={steps:3d} {'DONE' if done else '...'}", flush=True)

    env.env.close()

    # Decode imagination and save combined video.
    print(f"  [{tag}] decoding imagination...", flush=True)
    imag_frames = _decode_imagination(backend)

    result_str = "SUCCESS" if success else "FAIL"
    num_eps = 5
    ckpt_tag = "sft" if ckpt == CKPT_SFT else os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(ckpt))))
    obs_label  = f"OBS | {ckpt_tag} | t{task_id}: {name} | ep{ep_idx+1}/{num_eps} | {result_str}"
    imag_label = f"IMAGINATION | {ckpt_tag} | t{task_id}: {name} | ep{ep_idx+1}/{num_eps}"

    combined = _combine(obs_frames, imag_frames, obs_label, imag_label)
    task_dir = out_dir / f"task{task_id}_{name}"
    task_dir.mkdir(parents=True, exist_ok=True)
    vid_path = task_dir / f"ep{ep_idx}.mp4"
    imageio.mimwrite(str(vid_path), combined, fps=FPS, codec="libx264", quality=8)
    print(f"  [{tag}] {result_str} -> {vid_path} ({len(combined)} frames)", flush=True)

    (task_dir / f"ep{ep_idx}.json").write_text(json.dumps({"success": success, "frames": len(combined)}))
    return success


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        print("Usage: eval_worker.py <task_id> <ep_idx> <out_dir> [ckpt_path|sft]")
        sys.exit(1)
    task_id = int(sys.argv[1])
    ep_idx  = int(sys.argv[2])
    out_dir = Path(sys.argv[3])
    ckpt    = sys.argv[4] if len(sys.argv) == 5 else CKPT_RL
    ok = main(task_id, ep_idx, out_dir, ckpt=ckpt)
    sys.exit(0 if ok is not None else 1)

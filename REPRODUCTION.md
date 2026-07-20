# Tracing and Profiling Reproduction Guide

This guide outlines the steps to start the standalone trace server, run an end-to-end (E2E) PPO training run on LIBERO-Spatial using the OpenPI pi0.5 model, and capture trace profiling events.

---

## 1. Setup the Environment

Ensure you are in the workspace root and your virtual environment is active:
```bash
source /workspace/venv/bin/activate
export EMBODIED_PATH=/workspace/examples/embodiment
export PYTHONPATH=/workspace/libero:$PYTHONPATH
export MUJOCO_GL=egl
export ROBOT_PLATFORM=nvidia
```

---

## 2. Start the Standalone Trace Server

The trace server acts as the centralized HTTP collector. Launch it in the background or a separate shell:
```bash
/workspace/venv/bin/python3 toolkits/start_trace_server.py \
    --host 127.0.0.1 \
    --port 8888 \
    --file /workspace/trace_events.jsonl
```
*Note: This command will write all incoming trace events directly to `/workspace/trace_events.jsonl`.*

---

## 3. Launch E2E Training Run with Tracing Enabled

Execute the validation training run using `train_embodied_agent.py`. The tracing parameters `--trace-server-ip` and `--trace-server-port` will tell the driver and the auto-initialized Ray workers where to flush their captured trace spans.

Run the following command:
```bash
python3 examples/embodiment/train_embodied_agent.py \
    --config-name libero_spatial_ppo_openpi_pi05 \
    --trace-server-ip 127.0.0.1 \
    --trace-server-port 8888 \
    algorithm.rollout_epoch=2 \
    env.train.total_num_envs=32 \
    actor.micro_batch_size=64 \
    actor.global_batch_size=256 \
    actor.model.model_path=/workspace/models/RLinf-Pi05-LIBERO-SFT \
    rollout.model.model_path=/workspace/models/RLinf-Pi05-LIBERO-SFT \
    runner.max_steps=3
```

---

## 4. Verification and Visualization

### Line Count Check
Once training successfully exits after completing the 3 steps, verify the output file line count:
```bash
wc -l /workspace/trace_events.jsonl
```
*Expected: A successful clean run generates exactly **2,513** trace lines (each JSON line represents one trace event).*

### Verify Process Registrations
Check that all expected processes (`driver`, `EnvGroup`, `RolloutGroup`, and `ActorGroup`) successfully registered their Chrome Trace metadata:
```bash
grep '"ph": "M"' /workspace/trace_events.jsonl
```

### Perfetto UI Visualization
1. Download `/workspace/trace_events.jsonl` to your local machine.
2. Open [Perfetto UI](https://ui.perfetto.dev/) or `chrome://tracing` in a Chrome-based browser.
3. Click **Open trace file** and select the JSONL file to view interactive timelines showing exactly where time was spent during the execution of actor steps, environment interactions, and rollout generation.

# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from omegaconf import DictConfig
from rlinf.scheduler import Worker, Channel, Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.metric_utils import compute_split_num

class DummyActorWorker(Worker):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.placement = HybridComponentPlacement(cfg, Cluster())
        self.stage_num = cfg.rollout.pipeline_stage_num

    def init_worker(self):
        return None

    def set_global_step(self, step):
        pass

    def sync_model_to_rollout(self):
        return None

    def load_checkpoint(self, path):
        return None

    def save_checkpoint(self, path, step):
        return None

    async def recv_rollout_trajectories(self, input_channel: Channel):
        env_world_size = self.placement.get_world_size("env")
        actor_world_size = self._world_size
        send_num = env_world_size * self.stage_num
        recv_num = actor_world_size
        split_num = compute_split_num(send_num, recv_num)

        for _ in range(split_num):
            await input_channel.get(async_op=True).async_wait()

    def compute_advantages_and_returns(self):
        return {"dummy_adv": 0.0}

    def run_training(self):
        return {"dummy_loss": 0.0}

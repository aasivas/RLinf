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


class DummyRewardWorker(Worker):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.placement = HybridComponentPlacement(cfg, Cluster())

    def init_worker(self):
        return None

    async def compute_rewards(self, input_channel: Channel, output_channel: Channel):
        env_group_name = self.cfg.env.group_name
        stage_num = self.cfg.rollout.pipeline_stage_num
        train_batch_size = self.cfg.env.train.total_num_envs // stage_num
        local_num_train_envs = train_batch_size // self._world_size
        total_last_run_count = 0

        while True:
            # Recv input
            merged_data = await self.recv_from(
                group_name=env_group_name,
                channel=input_channel,
                tag="train_reward_obs",
                async_op=True,
                batch_size=train_batch_size,
            ).async_wait()

            last_run = merged_data.get("last_run", None)
            last_run_count = (
                int(last_run.sum().item()) if last_run is not None else 0
            )

            # Send output (rewards)
            rewards = torch.zeros(merged_data["obs"].shape[0], 1)
            self.send_to(
                group_name=env_group_name,
                channel=output_channel,
                data=rewards,
                tag="train_reward_obs",
                async_op=True,
            )

            total_last_run_count += last_run_count
            if total_last_run_count >= local_num_train_envs:
                break

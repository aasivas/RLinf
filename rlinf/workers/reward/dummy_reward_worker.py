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
from rlinf.utils.comm_mapping import CommMapper

class DummyRewardWorker(Worker):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.placement = HybridComponentPlacement(cfg, Cluster())
        self.total_num_train_envs = cfg.env.train.total_num_envs

    def init_worker(self):
        return None

    async def compute_rewards(self, input_channel: Channel, output_channel: Channel):
        env_world_size = self.placement.get_world_size("env")
        src_ranks = CommMapper.get_src_ranks(self.total_num_train_envs, env_world_size, self._world_size, self._rank)
        dst_ranks = CommMapper.get_dst_ranks(self.total_num_train_envs, self._world_size, env_world_size, self._rank)

        local_num_train_envs = sum(size for _, size in src_ranks)
        total_last_run_count = 0
        
        while True:
            # Recv input
            last_run_count = 0
            for src_rank, expected_size in src_ranks:
                data = await input_channel.get(
                    key=CommMapper.build_channel_key(src_rank, self._rank, extra="train_reward_input"),
                    async_op=True
                ).async_wait()
                last_run = data.get("last_run", None)
                last_run_count += int(last_run.sum().item()) if last_run is not None else 0
            
            # Send output (rewards)
            for dst_rank, size in dst_ranks:
                rewards = torch.zeros(size)
                output_channel.put(
                    rewards,
                    key=CommMapper.build_channel_key(self._rank, dst_rank, extra="train_reward_output"),
                    async_op=True
                )
            
            total_last_run_count += last_run_count
            if total_last_run_count >= local_num_train_envs:
                break

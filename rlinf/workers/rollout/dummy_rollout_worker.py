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
from rlinf.utils.comm_mapping import CommMapper
from rlinf.data.embodied_io_struct import RolloutResult
from rlinf.utils.placement import HybridComponentPlacement

class DummyRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.placement = HybridComponentPlacement(cfg, Cluster())
        
        self.num_action_chunks = cfg.actor.model.num_action_chunks
        self.action_dim = cfg.actor.model.action_dim
        
        self.n_train_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.num_action_chunks
        )
        self.n_eval_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.num_action_chunks
        )
        
        self.rollout_epoch = cfg.algorithm.get("rollout_epoch", 1)
        self.eval_rollout_epoch = cfg.algorithm.get("eval_rollout_epoch", 1)
        
        self.total_num_train_envs = cfg.env.train.total_num_envs
        self.total_num_eval_envs = cfg.env.eval.total_num_envs

    def init_worker(self):
        return None

    def set_global_step(self, step):
        pass

    def sync_model_from_actor(self):
        return None

    async def generate(self, input_channel: Channel, output_channel: Channel):
        env_world_size = self.placement.get_world_size("env")
        src_ranks = CommMapper.get_src_ranks(self.total_num_train_envs, env_world_size, self._world_size, self._rank)
        dst_ranks = CommMapper.get_dst_ranks(self.total_num_train_envs, self._world_size, env_world_size, self._rank)

        for _epoch in range(self.rollout_epoch):
            # Recv bootstrap obs
            for src_rank, expected_size in src_ranks:
                await input_channel.get(
                    key=CommMapper.build_channel_key(src_rank, self._rank, extra="train_obs"),
                    async_op=True
                ).async_wait()
                
            for _step in range(self.n_train_chunk_steps):
                # Send rollout results
                for dst_rank, size in dst_ranks:
                    actions = torch.randn(size, self.num_action_chunks, self.action_dim)
                    rollout_result = RolloutResult(
                        actions=actions,
                        forward_inputs={"action": actions},
                        bootstrap_values=torch.zeros(size, 1),
                        prev_logprobs=torch.zeros(size, self.num_action_chunks),
                        prev_values=torch.zeros(size, self.num_action_chunks),
                        versions=torch.zeros(size, dtype=torch.long)
                    )
                    output_channel.put(
                        rollout_result,
                        key=CommMapper.build_channel_key(self._rank, dst_rank, extra="train_rollout_results"),
                        async_op=True
                    )
                
                # Recv obs
                for src_rank, expected_size in src_ranks:
                    await input_channel.get(
                        key=CommMapper.build_channel_key(src_rank, self._rank, extra="train_obs"),
                        async_op=True
                    ).async_wait()

            # Final bootstrap rollout result
            for dst_rank, size in dst_ranks:
                rollout_result = RolloutResult(
                    bootstrap_values=torch.zeros(size, 1),
                    prev_values=torch.zeros(size, self.num_action_chunks)
                )
                output_channel.put(
                    rollout_result,
                    key=CommMapper.build_channel_key(self._rank, dst_rank, extra="train_rollout_results"),
                    async_op=True
                )

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        env_world_size = self.placement.get_world_size("env")
        src_ranks = CommMapper.get_src_ranks(self.total_num_eval_envs, env_world_size, self._world_size, self._rank)
        dst_ranks = CommMapper.get_dst_ranks(self.total_num_eval_envs, self._world_size, env_world_size, self._rank)

        for _epoch in range(self.eval_rollout_epoch):
            # Recv bootstrap obs
            for src_rank, expected_size in src_ranks:
                await input_channel.get(
                    key=CommMapper.build_channel_key(src_rank, self._rank, extra="eval_obs"),
                    async_op=True
                ).async_wait()
                
            for _step in range(self.n_eval_chunk_steps):
                # Send actions
                for dst_rank, size in dst_ranks:
                    actions = torch.randn(size, self.num_action_chunks, self.action_dim)
                    output_channel.put(
                        actions,
                        key=CommMapper.build_channel_key(self._rank, dst_rank, extra="eval_actions"),
                        async_op=True
                    )
                
                is_last_step = _step == self.n_eval_chunk_steps - 1
                if not is_last_step or self.cfg.env.eval.auto_reset:
                    for src_rank, expected_size in src_ranks:
                        await input_channel.get(
                            key=CommMapper.build_channel_key(src_rank, self._rank, extra="eval_obs"),
                            async_op=True
                        ).async_wait()

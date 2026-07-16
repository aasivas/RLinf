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
from rlinf.data.embodied_io_struct import RolloutResult
from rlinf.utils.placement import HybridComponentPlacement


class DummyRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.placement = HybridComponentPlacement(cfg, Cluster())
        self.num_action_chunks = cfg.actor.model.get("num_action_chunks", 1)
        self.action_dim = cfg.actor.model.get("action_dim", 1)

        self.n_train_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch // self.num_action_chunks
        )
        self.n_eval_chunk_steps = (
            (self.cfg.env.eval.max_steps_per_rollout_epoch // self.num_action_chunks)
            if "eval" in self.cfg.env
            else 0
        )

        self.rollout_epoch = cfg.env.train.rollout_epoch
        self.eval_rollout_epoch = (
            cfg.env.eval.get("rollout_epoch", 1) if "eval" in cfg.env else 1
        )
        self.env_decoupled_mode = cfg.env.get("decoupled_mode", False)

    def init_worker(self):
        return None

    def set_global_step(self, step):
        pass

    def sync_model_from_actor(self):
        return None

    def _split_rollout_result(
        self, rollout_result: RolloutResult, sizes: list[int]
    ) -> list[RolloutResult]:
        def _split_optional_tensor(
            tensor: torch.Tensor | None,
        ) -> tuple[torch.Tensor | None, ...]:
            if tensor is None:
                return tuple(None for _ in sizes)
            return tuple(torch.split(tensor, sizes, dim=0))

        split_actions = _split_optional_tensor(rollout_result.actions)
        split_prev_logprobs = _split_optional_tensor(rollout_result.prev_logprobs)
        split_prev_values = _split_optional_tensor(rollout_result.prev_values)
        split_bootstrap_values = _split_optional_tensor(rollout_result.bootstrap_values)
        split_intervene_flags = _split_optional_tensor(rollout_result.intervene_flags)
        split_versions = _split_optional_tensor(rollout_result.versions)
        split_forward_inputs = (
            [{} for _ in sizes]
            if not rollout_result.forward_inputs
            else [
                {
                    key: torch.split(value, sizes, dim=0)[idx]
                    for key, value in rollout_result.forward_inputs.items()
                    if value is not None
                }
                for idx in range(len(sizes))
            ]
        )
        return [
            RolloutResult(
                actions=split_actions[idx],
                prev_logprobs=split_prev_logprobs[idx],
                prev_values=split_prev_values[idx],
                bootstrap_values=split_bootstrap_values[idx],
                intervene_flags=split_intervene_flags[idx],
                forward_inputs=split_forward_inputs[idx],
                versions=split_versions[idx],
            )
            for idx in range(len(sizes))
        ]

    async def generate_one_epoch(self, input_channel: Channel, output_channel: Channel):
        env_group_name = self.cfg.env.group_name
        stage_num = self.cfg.rollout.pipeline_stage_num
        train_batch_size = self.cfg.env.train.total_num_envs // stage_num

        for _ in range(self.n_train_chunk_steps):
            for stage_id in range(stage_num):
                await self.recv_from(
                    group_name=env_group_name,
                    channel=input_channel,
                    tag="train_rollout_results",
                    route_key=stage_id if not self.env_decoupled_mode else None,
                    async_op=True,
                    batch_size=train_batch_size,
                ).async_wait()

                actions = torch.randn(
                    train_batch_size // self._world_size,
                    self.num_action_chunks,
                    self.action_dim,
                )
                rollout_result = RolloutResult(
                    actions=actions,
                    forward_inputs={"action": actions},
                    bootstrap_values=torch.zeros(
                        train_batch_size // self._world_size, 1
                    ),
                    prev_logprobs=torch.zeros(
                        train_batch_size // self._world_size, self.num_action_chunks
                    ),
                    prev_values=torch.zeros(
                        train_batch_size // self._world_size, self.num_action_chunks
                    ),
                    versions=torch.zeros(
                        train_batch_size // self._world_size, dtype=torch.long
                    ),
                )
                self.send_to(
                    group_name=env_group_name,
                    channel=output_channel,
                    data=rollout_result,
                    tag="train_rollout_results",
                    route_key=stage_id if not self.env_decoupled_mode else None,
                    async_op=True,
                    batch_size=train_batch_size,
                    split_fn=self._split_rollout_result,
                )

        for stage_id in range(stage_num):
            await self.recv_from(
                group_name=env_group_name,
                channel=input_channel,
                tag="train_rollout_results",
                route_key=stage_id if not self.env_decoupled_mode else None,
                async_op=True,
                batch_size=train_batch_size,
            ).async_wait()

            rollout_result = RolloutResult(
                bootstrap_values=torch.zeros(
                    train_batch_size // self._world_size, 1
                ),
                prev_values=torch.zeros(
                    train_batch_size // self._world_size, self.num_action_chunks
                ),
            )
            self.send_to(
                group_name=env_group_name,
                channel=output_channel,
                data=rollout_result,
                tag="train_rollout_results",
                route_key=stage_id if not self.env_decoupled_mode else None,
                async_op=True,
                batch_size=train_batch_size,
                split_fn=self._split_rollout_result,
            )

    async def generate(self, input_channel: Channel, output_channel: Channel):
        for _ in range(self.rollout_epoch):
            await self.generate_one_epoch(input_channel, output_channel)

    async def evaluate_one_epoch(self, input_channel: Channel, output_channel: Channel):
        env_group_name = self.cfg.env.group_name
        stage_num = self.cfg.rollout.pipeline_stage_num
        eval_batch_size = self.cfg.env.eval.total_num_envs // stage_num

        for _step in range(self.n_eval_chunk_steps):
            for stage_id in range(stage_num):
                await self.recv_from(
                    group_name=env_group_name,
                    channel=input_channel,
                    tag="rollout_results",
                    mode="eval",
                    route_key=stage_id if not self.env_decoupled_mode else None,
                    async_op=True,
                    batch_size=eval_batch_size,
                ).async_wait()

                actions = torch.randn(
                    eval_batch_size // self._world_size,
                    self.num_action_chunks,
                    self.action_dim,
                )
                self.send_to(
                    group_name=env_group_name,
                    channel=output_channel,
                    data=actions,
                    tag="rollout_results",
                    mode="eval",
                    route_key=stage_id if not self.env_decoupled_mode else None,
                    async_op=True,
                    batch_size=eval_batch_size,
                )

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.env_decoupled_mode:
            raise NotImplementedError(
                "DummyRolloutWorker does not support eval decoupled mode yet"
            )
        else:
            for _ in range(self.eval_rollout_epoch):
                await self.evaluate_one_epoch(input_channel, output_channel)

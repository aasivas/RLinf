# Copyright 2025 The RLinf Authors.
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

def register_reward(name: str, reward_class: type):
    assert name not in reward_registry, f"Reward {name} already registered"
    reward_registry[name] = reward_class


def get_rule_based_reward_class(name: str):
    assert name in reward_registry, f"Reward {name} not found"
    val = reward_registry[name]
    if isinstance(val, str):
        import importlib
        module_path, class_name = val.rsplit(".", 1)
        module = importlib.import_module(module_path)
        val = getattr(module, class_name)
        reward_registry[name] = val
    return val


reward_registry = {
    "math": "rlinf.algorithms.rewards.math.MathReward",
    "vqa": "rlinf.algorithms.rewards.vqa.VQAReward",
    "code_offline": "rlinf.algorithms.rewards.code.CodeRewardOffline",
    "searchr1": "rlinf.algorithms.rewards.searchr1.SearchR1Reward",
    "rstar2": "rlinf.algorithms.rewards.rstar2.Rstar2Reward",
}

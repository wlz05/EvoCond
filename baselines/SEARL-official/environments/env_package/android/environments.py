# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict

class BaseEnv(ABC):
    @abstractmethod
    def step(self, llm_raw_response) -> Tuple[Dict, float, bool, Dict]:
        '\n        action is llm raw response\n        Args:\n            action: Action to take, assume it\'s raw llm action\n\n        Returns:\n            obs, reward, done, info\n\n        obs: {\n            \'obs_str\': "This is the obs template, you see <image> and <image>, you heard <audio> and <audio>",\n            \'multi_modal_data\':{\n                \'<image>\':[list of images],\n                \'<audio>\':[list of audios],\n            }\n            # num of <image> and <audio> in the obs_str should match len(multi_modal_data[\'<image>\']) and len(multi_modal_data[\'<audio>\'])\n        }\n        info: {\n            "metrics": {\n                \'success\': success,\n                \'action_is_effective\': action_is_effective,\n                \'action_is_valid\': action_is_valid,\n            } # metrics you want to log in wandb\n            "llm_raw_response": llm_raw_response,\n            "llm_response": llm_response, # for update\n        }\n        '
        pass

    @abstractmethod
    def close(self):
        """Close the environment."""
        pass

    @abstractmethod
    def reset(self, seed=None) -> Tuple[Dict, Dict]:
        '\n        Reset the environment.\n        NOTE: the environment should be same for the same seed\n        Args:\n            seed: Seed for the environment\n\n        Returns:\n            obs,info\n\n            format should be same as step\n        '
        pass

    @abstractmethod
    def system_prompt(self) -> str:
        '\n        Get the system prompt for the environment.\n\n        Returns:\n            System prompt string.\n        '
        pass

    def compute_reward(self) -> float:
        '\n        give final reward\n        Currently the reward calculation in rollout manager will be: sum(step_rewards)+env.compute_reward()\n        In most cases you can set this to 0.0 since the step rewards are already accumulated, but if you want to add some extra reward for the final step, define it here.\n        '
        return 0.0

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com

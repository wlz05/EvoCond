# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from typing import List, Tuple, Dict, Union, Any
import torch
import numpy as np
import os
from environments.prompts import *
from collections import defaultdict
from PIL import Image

def to_numpy(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        pass
    elif isinstance(data, (int, float, bool, Tuple, List)):
        data = np.array(data)
    else:
        raise ValueError(f"Unsupported type: {type(data)})")
    return data

class EnvironmentManagerBase:
    def __init__(self, envs, projection_f, env_name=None):
        '\n        Initialize the environment manager.\n\n        Parameters:\n        - envs: The environment instance, usually a vectorized environment containing multiple sub-environments.\n        - projection_f: A function that maps text actions to environment actions.\n        - env_name (str): The name of the environment.\n        '
        self.envs = envs
        self.projection_f = projection_f
        self.env_name = env_name

    def reset(self) -> Dict[str, Any]:
        "\n        Reset all environments and return the initial observations.\n\n        Returns:\n        - next_observations (Dict):\n          - 'text' (None or List[str]): The textual observation.\n          - 'image' (np.ndarray or torch.Tensor): The image observation as either a NumPy array or a PyTorch tensor.\n          - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).\n        "
        obs, infos = self.envs.reset()
        return {'text': None, 'image': obs, 'anchor': None}, infos

    def step(self, text_actions: List[str]):
        "\n        Execute text actions and return the next state, rewards, done flags, and additional information.\n\n        Parameters:\n        - text_actions (List[str]): A list of text actions to execute.\n\n        Returns:\n        - next_observations (Dict):\n          - 'text' (None or List[str]): The textual observation.\n          - 'image' (np.ndarray or torch.Tensor): The image observation as either a NumPy array or a PyTorch tensor.\n          - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).\n        - rewards (np.ndarry or torch.Tensor): The rewards returned by the environment.\n        - dones (np.ndarray or torch.Tensor): Done flags indicating which environments have completed.\n        - infos (List[Dict]): Additional environment information.\n\n        Exceptions:\n        - NotImplementedError: If an observation key is not in ('text', 'image').\n        "
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_observations = {
            'text': None,  # Implement this if needed
            'image': next_obs,
            'anchor': None
            # For GiGPO only. anchor observation without any histories, hint, etc. Implement this if needed
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, ) -> List[str]:
        '\n        This function builds the text observation for the agent.\n\n        Returns:\n        - postprocess_text_obs (List[str]): A list of processed text observations.\n        '
        pass

    def close(self) -> None:
        '\n        Close the environment and release resources.\n        '
        self.envs.close()

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        "\n        Evaluate if the episodes are successful or not.\n        (Default) implementation is to check info['won'] of the last step.\n\n        Returns:\n        - success (np.ndarray or torch.Tensor): 1 if the episode is successful, 0 otherwise.\n        "
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        batch_size = len(total_batch_list)

        success = defaultdict(list)

        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)

        assert len(success['success_rate']) == batch_size

        return {key: np.array(value) for key, value in success.items()}

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                return

    def save_image(self, image, step):
        '\n        Save an image to a file.\n\n        Parameters:\n        - image (np.ndarray or torch.Tensor): The image to save.\n        - path (str): The path to save the image.\n        '
        path = os.path.join(os.path.dirname(__file__), os.path.join("images", self.env_name))
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f"step{step}.png")
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported type: {type(image)})")

        if len(image.shape) == 4:
            image = image[0]
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.max() <= 1.0:
            image = (image * 255)

        image = image.astype(np.uint8)

        image = Image.fromarray(image)
        image.save(path)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com

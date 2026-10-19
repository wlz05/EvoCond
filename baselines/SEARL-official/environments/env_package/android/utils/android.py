# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from typing import List
from abc import ABC, abstractmethod

class BaseModelCaller(ABC):
    '\n    Base class for multimodal model calling.\n\n    Users must subclass this and implement the `call()` method\n    to define how to send prompt and images to their model or API.\n    '
    @abstractmethod
    def call(self, images: List[str], prompt: str) -> str:
        "\n        Send prompt and images to model and return its response.\n\n        Args:\n            images (List[str]): List of base64-encoded image strings (without MIME header).\n            prompt (str): Instructional text to guide the model's behavior.\n\n        Returns:\n            str: Model-generated response.\n        "
        pass

def call_model(
    images: List[str],
    prompt: str,
    caller: BaseModelCaller
) -> str:
    '\n    Args:\n        image_base64_list (List[str]):\n            A list of image strings encoded in base64 (without the MIME prefix).\n            Each item should be a single image in base64 format.\n        prompt_content (str):\n            A natural language instruction or question that guides the model\'s response,\n            such as "Describe the image" or "What is this UI?".\n\n    Returns:\n        str:\n            The textual output returned by the model, typically a natural language answer.\n    '
    return caller.call(images, prompt)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com

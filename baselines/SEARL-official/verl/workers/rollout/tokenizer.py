# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'\nThe base tokenizer class, required for any hybrid engine based rollout or inference with vLLM.\n'

from abc import ABC, abstractmethod

import numpy as np
import torch

__all__ = ["HybridEngineBaseTokenizer"]

class HybridEngineBaseTokenizer(ABC):
    """the tokenizer property and function name should align with HF's to meet vllm requirement"""

    @property
    @abstractmethod
    def vocab_size(self):
        '\n        `int`: Size of the base vocabulary (without the added tokens).\n        '
        pass

    @property
    @abstractmethod
    def pad_token_id(self):
        '\n        `Optional[int]`: Id of the padding token in the vocabulary. Returns `None` if the token has not been set.\n        '
        pass

    @property
    @abstractmethod
    def eos_token_id(self):
        '\n        `Optional[int]`: Id of the end of sentence token in the vocabulary. Returns `None` if the token has not been\n        set.\n        '
        pass

    @property
    @abstractmethod
    def all_special_ids(self) -> list[int]:
        "\n        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.\n        "
        pass

    @property
    @abstractmethod
    def all_special_tokens(self) -> list[str]:
        "\n        `List[str]`: A list of the unique special tokens (`'<unk>'`, `'<cls>'`, ..., etc.).\n\n        Convert tokens of `tokenizers.AddedToken` type to string.\n        "
        pass

    @abstractmethod
    def encode(self, text):
        '\n        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.\n\n        Args:\n            text (`str`, `List[str]` or `List[int]`):\n                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the\n                `tokenize` method) or a list of integers.\n\n            text_pair (`str`, `List[str]` or `List[int]`, *optional*):\n                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using\n                the `tokenize` method) or a list of integers.\n        '
        pass

    @abstractmethod
    def decode(
        self,
        token_ids: int | list[int] | np.ndarray | torch.Tensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        '\n        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special\n        tokens and clean up tokenization spaces.\n\n        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.\n\n        Args:\n            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor]`):\n                List of tokenized input ids. Can be obtained using the `__call__` method.\n            skip_special_tokens (`bool`, *optional*, defaults to `False`):\n                Whether or not to remove special tokens in the decoding.\n            clean_up_tokenization_spaces (`bool`, *optional*):\n                Whether or not to clean up the tokenization spaces. If `None`, will default to\n                `self.clean_up_tokenization_spaces`.\n            kwargs (additional keyword arguments, *optional*):\n                Will be passed to the underlying model specific decode method.\n\n        Returns:\n            `str`: The decoded sentence.\n        '
        pass

    @abstractmethod
    def convert_ids_to_tokens(self, ids: int | list[int], skip_special_tokens: bool = False) -> str | list[str]:
        '\n        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and\n        added tokens.\n\n        Args:\n            ids (`int` or `List[int]`):\n                The token id (or token ids) to convert to tokens.\n            skip_special_tokens (`bool`, *optional*, defaults to `False`):\n                Whether or not to remove special tokens in the decoding.\n\n        Returns:\n            `str` or `List[str]`: The decoded token(s).\n        '
        pass

    @abstractmethod
    def get_added_vocab(self) -> dict[str, int]:
        '\n        Returns the added tokens in the vocabulary as a dictionary of token to index. Results might be different from\n        the fast call because for now we always add the tokens even if they are already in the vocabulary. This is\n        something we should change.\n\n        Returns:\n            `Dict[str, int]`: The added tokens.\n        '
        pass

    @abstractmethod
    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        '\n        Converts a sequence of tokens in a single string. The most simple way to do it is `" ".join(tokens)` but we\n        often want to remove sub-word tokenization artifacts at the same time.\n\n        Args:\n            tokens (`List[str]`): The token to join in a string.\n\n        Returns:\n            `str`: The joined tokens.\n        '
        pass

    @property
    def is_fast(self):
        return False

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com

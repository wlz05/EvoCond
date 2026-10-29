# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

# Copyright 2025 Amazon.com Inc and/or its affiliates
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
'\nDataset class that enables dynamic data generation strategies between iterations of training.\nThis class extends RLHFDataset and uses an AbstractDataGen instance to generate data.\n\nThis is especially useful in settings where proposer model generates new tasks based\non rollout data.\n'

import logging
from abc import ABC, abstractmethod
from typing import Optional

import datasets
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl import DataProto
from verl.utils.dataset import RLHFDataset
from verl.utils.imports import load_extern_type

logger = logging.getLogger(__name__)

class AbstractDataGenerator(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def generate(self, dataset: Dataset) -> datasets.Dataset:
        '\n        Generate method must be implemented by subclasses.\n        Args:\n            dataset: The dataset to generate from.\n        Returns:\n            Processed data or result as implemented by the subclass.\n        '
        pass

class MockDataGenerator(AbstractDataGenerator):
    '\n    A noop data gen class that only reappends the first datapoint.\n    This class is useful as a placeholder and testing.\n    '

    def __init__(self, config: DictConfig = None):
        super().__init__(config)

    def generate(self, dataset: Dataset) -> datasets.Dataset:
        print("MockDataGenerator: No operation performed on the dataset.")
        return dataset.dataframe.select([0])

class DynamicGenDataset(RLHFDataset):
    '\n    A dataset class that uses a data generation strategy to process data.\n    This class extends RLHFDataset and uses an AbstractDataGen instance to generate data.\n    '

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        super().__init__(data_files, tokenizer, config, processor)
        self.datagen: AbstractDataGenerator = config.datagen
        assert "datagen" in config and config.datagen.get("path", None) is not None, (
            f"datagen path is not set in config: {config}"
        )
        # Dynamically load the custom datagen class
        datagen_cls = load_extern_type(config.datagen.path, config.datagen.name)

        # Verify that the custom datagen class inherits from AbstractDataGenerator
        abs_cls = AbstractDataGenerator
        if not issubclass(datagen_cls, abs_cls):
            raise TypeError(
                f"The custom datagen class '{config.datagen.name}' from '{config.datagen.path}'"
                + " must inherit from {abs_cls}"
            )

        self.data_generator = datagen_cls(config.datagen)
        self.on_batch_end()

    def append_dataframe(self, new_dataframe: datasets.Dataset):
        new_dataframe = self.maybe_filter_out_long_prompts(new_dataframe)
        self.dataframe = datasets.concatenate_datasets([self.dataframe, new_dataframe])

        logger.info(f"new dataset len: {len(self.dataframe)}")

    def on_batch_end(self, batch: DataProto) -> None:
        '\n        Generate data using the provided data generation strategy.\n        Note: This method is intended to change the dataset after each training batch.\n        '
        new_data = self.data_generator.generate(self)
        self.append_dataframe(new_data)

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com

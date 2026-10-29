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

'\nThis module is used to merge huggingface model and test verl checkpoints from FSDP and Megatron backends.\n\nTo merge FSDP checkpoints:\n```sh\npython -m verl.model_merger merge     --backend fsdp     --local_dir /mnt/shared-storage-user/ai4good2-share/fengxinshun/saves/mgpo_hazard_rnn_g0.5_all_span/global_step_87/actor     --target_dir /mnt/shared-storage-user/ai4good2-share/fengxinshun/saves/mgpo_hazard_rnn_g0.5_all_span\n```\n\nTo merge Megatron checkpoints:\n```sh\npython -m verl.model_merger merge     --backend megatron     --tie-word-embedding     --local_dir checkpoints/verl_megatron_gsm8k_examples/qwen2_5_0b5_megatron_saveload/global_step_1/actor     --target_dir /path/to/merged_hf_model\n```\n\nor use distribtued merge for large models like dpskv3 671B\n\n```sh\ntorchrun --nproc_per_node 1 --nnodes 8 --node_rank ${RANK} -m verl.model_merger merge    --backend megatron     --local_dir ./checkpoints/global_step_1/actor     --target_dir /path/to/merged_hf_model\n```\n\n\nFor more details, please refer to documentation:\nhttps://verl.readthedocs.io/en/latest/advance/checkpoint.html#convert-fsdp-and-megatron-checkpoints-to-huggingface-format-model\n'

from .base import generate_config_from_args, parse_args

def main():
    args = parse_args()
    config = generate_config_from_args(args)
    print(f"config: {config}")

    if config.backend == "fsdp":
        from .merge import FSDPModelMerger

        merger = FSDPModelMerger(config)
    elif config.backend == "megatron":
        from .parallel import MegatronModelMerger

        merger = MegatronModelMerger(config)
    else:
        raise NotImplementedError(f"Unknown backend: {config.backend}")

    merger.merge_and_save()
    merger.cleanup()

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com

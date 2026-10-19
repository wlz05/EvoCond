# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import re
import torch
import string
import random
import warnings
from abc import ABC
from verl import DataProto
from envs.tool_manager import TOOL_MANAGER_REGISTRY
from verl.utils.model import compute_position_id_with_mask
from verl.utils.functionals import tokenize_and_postprocess_data
from typing import List, Tuple
from PIL import Image
import sys
from copy import deepcopy

class MMEnv(ABC): # Serves as the base environment for multimodal environments
    '\n    Abstract base class for multimodal reinforcement learning environments in the SEARL framework.\n    \n    This class extends the basic environment functionality to support multimodal inputs,\n    particularly images, alongside text. It provides the core functionality for environment \n    interaction, tool management, reward computation, and data processing in multimodal settings.\n    \n    Attributes:\n        tool_manager: Manages tool execution and interaction with multimodal language models\n        max_prompt_length (int): Maximum allowed length for prompts\n        use_verify_tool (bool): Flag indicating whether to use verification tools\n        use_process_reward (bool): Flag indicating whether to use process rewards\n        chat_template (str): Chat template for formatting conversations\n        processor: Processor for handling multimodal data (image/text)\n    '
    def __init__(self, config, centralized_actor=None):
        tool_manager_name = config.get('tool_manager', 'qwen3')
        # Check if a tool manager is specified
        # If not specified, use adaptive mode
        if not tool_manager_name:
            tool_manager_name = "adaptive"
        if tool_manager_name.startswith('centralized_'):
            if centralized_actor is None:
                raise ValueError(f"\u4f7f\u7528\u96c6\u4e2d\u5f0f\u5de5\u5177\u7ba1\u7406\u5668 '{tool_manager_name}' \u9700\u8981\u63d0\u4f9b centralized_actor \u53c2\u6570")
            self.tool_manager = TOOL_MANAGER_REGISTRY[tool_manager_name](
                verl_config=config,
                centralized_actor_handle=centralized_actor
            )
        else:
            # Distributed mode, keep the original logic unchanged
            if tool_manager_name == 'adaptive':
                model_type = config.get('model_type')
                if 'qwen3' in model_type:
                    tool_manager_name = 'qwen3'
                elif 'qwen2' in model_type:
                    tool_manager_name = 'qwen2_5'
                elif 'qwen2_5_vl' in model_type:
                    tool_manager_name = 'qwen2_5_vl'
                elif 'llama' in model_type:
                    tool_manager_name = 'llama3'
                else:
                    tool_manager_name = model_type
                    raise ValueError(f"'{tool_manager_name}' \u9700\u8981\u8fdb\u884c\u9002\u914d\uff0c\u8bf7\u6dfb\u52a0\u4e00\u4e2a\u5bf9\u5e94\u7684tool_manager")

            self.tool_manager = TOOL_MANAGER_REGISTRY[tool_manager_name](verl_config=config)
        # breakpoint()
        # self.processor = self.tool_manager.processor
        # breakpoint()
        self.max_prompt_length = config.get('max_prompt_length', 2048)
        self.use_verify_tool = False
        self.use_process_reward = config.get('use_process_reward', False)
        self.chat_template = config.get('chat_template', None)

    def verify_tool(self, data_source, solution_str, ground_truth, extra_info):
        # If you need a tool to evaluate the generated response, you need to modify the following code
        # the data would be stored in data[i].non_tensor_batch['reward_model']['ground_truth']['verified_results']
        # print('verify tool start')
        raise NotImplementedError

    def _process_data(self, data_item, tokenizer):
        # process the data_item to the token and decode them
        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        # prompt_str = tokenizer.decode(valid_prompt_ids)
        # response_str = tokenizer.decode(valid_response_ids)
        # response_str = response_str.replace("<|image_pad|>", "")
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
        data_source = data_item.non_tensor_batch['data_source']
        extra_info = data_item.non_tensor_batch.get('extra_info', None)

        return {
            'prompt_str': prompt_str,
            'response_str': response_str,
            'ground_truth': ground_truth,
            'data_source': data_source,
            'extra_info': extra_info
        }
    def _replace_all_image_pads(self, original_str, count, processor):

        "\n        \u5c06\u4e00\u4e2a\u5b57\u7b26\u4e32\u4e2d\u6240\u6709\u7684 '<image_pad>' \u66ff\u6362\u4e3a\u6307\u5b9a\u6570\u91cf\u7684 '<image_pad>'\u3002\n\n        \u53c2\u6570:\n        original_str (str): \u539f\u59cb\u8f93\u5165\u5b57\u7b26\u4e32\u3002\n        count (int): \u5e0c\u671b\u66ff\u6362\u6210\u7684 '<image pad>' \u7684\u6570\u91cf\u3002\n\n        \u8fd4\u56de:\n        str: \u66ff\u6362\u540e\u7684\u65b0\u5b57\u7b26\u4e32\u3002\n        "

        replacement = ' '.join(['<|image_pad|>'] * count)

        new_str = original_str.replace('<|image_pad|>', replacement)
        return new_str

    def get_step_reward(self, responses, format_score=0.1):

        step_reward = [1] * len(responses)
        return step_reward

    def _mm_process(self, mm_output, image_result, temp_next_obs, processor):
        '\u5904\u7406\u591a\u6a21\u6001\u8f93\u51fa\uff0c\u751f\u6210\u591a\u6a21\u6001\u6570\u636e\u548c\u66f4\u65b0\u540e\u7684\u89c2\u5bdf\u3002'
        if not mm_output:
            return None, None, temp_next_obs
        mm_data = processor.image_processor(image_result, return_tensors='pt')
        temp_multi_modal_data = {"pixel_values": mm_data["pixel_values"], "image_grid_thw": mm_data["image_grid_thw"]}
        temp_image_data = image_result
        temp_next_obs = self._replace_all_image_pads(temp_next_obs, mm_data["image_grid_thw"][0].prod() // (processor.image_processor.merge_size**2),processor)
        return temp_multi_modal_data, temp_image_data, temp_next_obs

    def step(self, responses, processor, image_data: List[List[Image.Image]]):
        '\u6267\u884c\u73af\u5883\u6b65\u9aa4\uff0c\u6839\u636e\u6a21\u578b\u54cd\u5e94\u548c\u56fe\u50cf\u6570\u636e\u66f4\u65b0\u73af\u5883\u72b6\u6001\u3002\n        \n        Args:\n            responses: \u6a21\u578b\u751f\u6210\u7684\u54cd\u5e94\u5217\u8868\n            processor: \u6a21\u578b\u5904\u7406\u5668\uff0c\u7528\u4e8e\u5904\u7406\u56fe\u50cf\u6570\u636e\n            image_data: \u8f93\u5165\u7684\u56fe\u50cf\u6570\u636e\u5217\u8868\uff0c\u6bcf\u4e2a\u5143\u7d20\u4e3a\u4e00\u4e2a\u56fe\u50cf\u6279\u6b21\n            \n        Returns:\n            \u5305\u542b\u66f4\u65b0\u540e\u7684\u89c2\u5bdf\u3001\u5b8c\u6210\u6807\u5fd7\u3001\u6709\u6548\u64cd\u4f5c\u3001\u662f\u5426\u4e3a\u5de5\u5177\u8c03\u7528\u3001\u65b0\u56fe\u50cf\u6570\u636e\u3001\u6709\u6548\u5de5\u5177\u8c03\u7528\u548c\u539f\u59cb\u63d0\u793a\u7684\u5143\u7ec4\n        '
        tokenizer = processor.tokenizer
        print("start to the env step", file=sys.stderr, flush=True)
        cur_actions, tool_results = self.tool_manager.execute_actions(responses=responses, image_data=image_data)
        next_obs, dones, valid_action, is_tool, new_image = [], [], [], [], []
        raw_prompt = []
        multi_modal_data = []
        valid_tool = []

        for action, tool_result in zip(cur_actions, tool_results):
            # print(f" action: {action} tool_result: {tool_result}\n\n")
            tool_result = tool_result[0]
            image_result = tool_result[1]
            mm_output = True if image_result is not None else False
            temp_valid_tool = 0
            temp_multi_modal_data = None
            temp_raw_prompt = None
            if action == 'answer':
                temp_next_obs, temp_done, temp_valid_action, temp_is_tool, temp_image_data, temp_valid_tool, temp_raw_prompt = '', True, 0, 1, None, 0, None
            elif action == 'error':
                temp_next_obs = self.tool_manager.get_prompt(
                    input_data=tool_result,
                    tokenizer=tokenizer,
                    mode='tool_call',
                    add_generation_prompt=True
                )
                temp_done, temp_valid_action, temp_is_tool, temp_image_data, temp_raw_prompt = False, 0, 0, None, temp_next_obs
            elif action == 'actions':
                # print(f" action: {action} tool_result: {tool_result}\n\n")

                temp_next_obs = self.tool_manager.get_prompt(
                    input_data=tool_result[0],
                    tokenizer=tokenizer,
                    mode='tool_call',
                    add_generation_prompt=True
                )
                temp_raw_prompt = deepcopy(temp_next_obs)
                if mm_output and "<|image_pad|>" not in temp_next_obs:
                    print(f"mm_output and <|image_pad|> not in temp_next_obs")
                    breakpoint()
                temp_multi_modal_data, temp_image_data, temp_next_obs = self._mm_process(mm_output, image_result, temp_next_obs, processor)
                temp_done, temp_valid_action, temp_is_tool, temp_valid_tool = False, 0, 1, 1 if mm_output else 0
            else:
                raise ValueError('Unexpected action: {}'.format(action))

            next_obs.append(temp_next_obs)
            dones.append(temp_done)
            valid_action.append(temp_valid_action)
            is_tool.append(temp_is_tool)
            new_image.append(temp_image_data)
            multi_modal_data.append(temp_multi_modal_data)
            valid_tool.append(temp_valid_tool)
            raw_prompt.append(temp_raw_prompt)

        print(f"image valid tool execute is {sum(valid_tool)} and overall batch size is  {len(dones)}",file=sys.stderr, flush=True)
        assert sum(valid_tool) == sum(1 for item in new_image if isinstance(item, Image.Image))
        return next_obs, dones, valid_action, is_tool, new_image, raw_prompt, multi_modal_data, valid_tool

    def compute_score(self, reward_rollout_wg, reward_tokenizer, tokenizer, data: DataProto, if_val=False, use_process_reward=False):
        if reward_rollout_wg is not None:
            scores = self._compute_score_with_reward_rollout_wg(reward_rollout_wg, reward_tokenizer, data)
        else:
            score = self._compute_score_with_rules(data, tokenizer, if_val=if_val)
            if use_process_reward and not if_val:
                scores = []
                missing_tool_use_scores = 0
                for i in range(len(data)):
                    data_item = data[i]
                    if "tool_use_scores" in data_item.batch.keys():
                        tool_use_score = data_item.batch['tool_use_scores']
                        validate_score = tool_use_score[~torch.isnan(tool_use_score)].tolist()
                        scores.append(validate_score + score[i])
                    else:
                        missing_tool_use_scores += 1
                        scores.append(score[i])
                if missing_tool_use_scores > 0:
                    warnings.warn(
                        f"tool_use_scores missing for {missing_tool_use_scores}/{len(data)} samples; "
                        "falling back to final-score-only reward for those samples."
                    )
            else:
                scores = score

        return scores

    def _compute_score_with_rules(self, data, tokenizer, if_val=False):
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            # process the data_item to the token and decode them
            processed_data = self._process_data(data_item=data_item, tokenizer=tokenizer)
            ground_truth, response_str = processed_data['ground_truth'], processed_data['response_str']

            # reserved for compatibility
            prompt_str, data_source, extra_info = processed_data['prompt_str'], processed_data['data_source'], processed_data['extra_info']

        scores = [[0.0]] * len(data)

        return scores

    def get_prompt_for_reward(self, reward_tokenizer, data: DataProto):
        '\u83b7\u53d6\u5956\u52b1\u6a21\u578b\u4f7f\u7528\u7684prompt'
        reward_prompt_strs = []

        for i in range(len(data)):
            data_item = data[i]
            step_mask = data.batch['step_mask'][i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            non_tensor_data = {
                'data_source': data_item.non_tensor_batch['data_source'],
                'ground_truth': data_item.non_tensor_batch['reward_model']['ground_truth'],
                'extra_info': data_item.non_tensor_batch.get('extra_info', None)
            }

            mask_indices = torch.where(step_mask == 1)[0]
            assert len(mask_indices) > 0, "no step mask"

            reward_prompt_str_list = []
            start_idx = 0

            for end_idx in mask_indices:

                current_response = valid_response_ids[start_idx:end_idx]
                current_response_str = reward_tokenizer.decode(current_response, skip_special_tokens=True)

                reward_prompt_str = self._get_single_prompt_str(
                    data_source=non_tensor_data['data_source'],
                    solution_str=current_response_str,
                    ground_truth=non_tensor_data['ground_truth'],
                    extra_info=non_tensor_data['extra_info'],
                    reward_tokenizer=reward_tokenizer
                )
                reward_prompt_str_list.append(reward_prompt_str)

                start_idx = end_idx

            assert start_idx == len(valid_response_ids) - 1, "start_idx is not the last index"

            reward_prompt_strs.append(reward_prompt_str_list)

        return reward_prompt_strs

    def _get_single_prompt_str(self, data_source, solution_str, ground_truth, extra_info, reward_tokenizer):
        # If you need use the reasoning model to generate the reward, you need to modify the following code
        # result = reward_tokenizer.apply_chat_template([
        #     {'role': 'system', 'content': 'You are a assistant. '},

        # ], add_generation_prompt=True, tokenize=False)
        # return result
        raise NotImplementedError

    def _compute_score_with_reward_rollout_wg(self, reward_rollout_wg, reward_tokenizer, data: DataProto):

        reward_prompt_strs = self.get_prompt_for_reward(reward_tokenizer, data)

        flat_prompts = []
        original_shapes = []
        for prompts in reward_prompt_strs:
            original_shapes.append(len(prompts))
            flat_prompts.extend(prompts)

        input_ids = []
        attention_mask = []
        for prompt in flat_prompts:

            ids, mask = tokenize_and_postprocess_data(
                prompt=prompt,
                tokenizer=reward_tokenizer,
                max_length=self.max_prompt_length,
                pad_token_id=reward_tokenizer.pad_token_id,
                left_pad=True,
                truncation='error'
            )
            input_ids.append(ids)
            attention_mask.append(mask)

        tensors = {
            "input_ids": torch.cat(input_ids, dim=0),
            "attention_mask": torch.cat(attention_mask, dim=0)
        }

        tensors["position_ids"] = compute_position_id_with_mask(tensors["attention_mask"])

        data_proto = DataProto.from_dict(tensors=tensors)

        responses_data = reward_rollout_wg.generate_sequences(data_proto)

        flat_scores = []
        for i, temp_response_data in enumerate(responses_data):

            data_idx = 0
            prompt_count = 0
            while data_idx < len(original_shapes) and prompt_count + original_shapes[data_idx] <= i:
                prompt_count += original_shapes[data_idx]
                data_idx += 1

            data_item = data[data_idx]

            temp_prompt_ids = temp_response_data.batch['prompts']
            temp_prompt_length = temp_prompt_ids.shape[-1]
            temp_response_ids = temp_response_data.batch['responses']
            temp_valid_response_length = temp_response_data.batch['attention_mask'][temp_prompt_length:].sum()
            temp_valid_response_ids = temp_response_ids[:temp_valid_response_length]
            response_str = reward_tokenizer.decode(temp_valid_response_ids, skip_special_tokens=True)

            score = self._compute_single_score_with_reward_rollout_wg(
                data_item.non_tensor_batch['data_source'],
                response_str,
                data_item.non_tensor_batch['reward_model']['ground_truth'],
                data_item.non_tensor_batch.get('extra_info', None)
            )
            flat_scores.append(score)

        assert data_idx == len(data) - 1, "data_idx is {}".format(data_idx)

        scores = []
        start_idx = 0
        for shape in original_shapes:
            end_idx = start_idx + shape
            scores.append(flat_scores[start_idx:end_idx])
            start_idx = end_idx

        return scores

    def _compute_single_score_with_reward_rollout_wg(self, data_source, solution_str, ground_truth, extra_info):
        # print("solution_str: ", solution_str)
        # return 1.0
        raise NotImplementedError

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com

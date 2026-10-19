# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import torch
import re
import numpy as np
import json
import ast

from verl.utils.reward_score.evaluation.custom_eval import evaluate_plan
from typing import Dict, Optional
from verl import DataProto

def calculate_final_answer_reward(
    decoded_text: str,
    hard_constraint_data: Optional[Dict] = None,
    answer_format_score: float = 1.0
) -> float:
    '\n    \u4ece\u89e3\u7801\u540e\u7684\u6587\u672c\u4e2d\u63d0\u53d6 <answer>...</answer> \u90e8\u5206\uff0c\u5e76\u8ba1\u7b97\u5956\u52b1\u3002\n    \n    \u53c2\u6570\uff1a\n        decoded_text: \u89e3\u7801\u540e\u7684\u6587\u672c\n        hard_constraint_data: \u9884\u8bbe\u5b9a\u7684\u67e5\u8be2\u6570\u636e\uff08\u53ef\u9009\uff09\n    \n    \u8fd4\u56de\uff1a\n        \u8ba1\u7b97\u5f97\u5230\u7684\u5956\u52b1\u5206\u6570\uff08float\uff09\n    '
    # total_reward = answer_format_reward + final_answer_reward
    total_reward = 0.0
    answer_format_reward = 0.0
    final_answer_reward = 0.0
    pattern = r'<answer>(.*?)</answer>'
    final_answers = re.findall(pattern, decoded_text, re.DOTALL)

    if len(final_answers) < 2:
        return total_reward, answer_format_reward, final_answer_reward

    second_answer = final_answers[-1].strip()
    second_answer = re.sub(r'```(json)?|```', '', second_answer)

    try:
        test_data = json.loads(second_answer)
    except json.JSONDecodeError as e:
        print(f"DEBUG: JSON \u89e3\u6790\u5931\u8d25 - {e}")
        return total_reward, answer_format_reward, final_answer_reward
    except Exception as e:
        print(f"DEBUG: \u672a\u77e5\u89e3\u6790\u9519\u8bef - {e}")
        return total_reward, answer_format_reward, final_answer_reward

    try:
        final_answer_reward = evaluate_plan(hard_constraint_data, test_data)
        answer_format_reward = 1.0
        total_reward = total_reward + answer_format_reward + final_answer_reward
    except Exception as e:
        print(f"DEBUG: evaluate_plan \u6267\u884c\u5931\u8d25 - {e}")
        return total_reward, answer_format_reward, final_answer_reward

    return total_reward, answer_format_reward, final_answer_reward

def compute_score_fn(data: DataProto,
                     tool_reward: float,
                     memory_reward: float,
                     answer_format_reward: float,
                     tokenizer=None):
    "\n    \u8ba1\u7b97\u6279\u6b21\u6570\u636e\u7684\u5956\u52b1\u5206\u6570\uff0c\u5305\u542b\u5de5\u5177\u4f7f\u7528\u3001\u8bb0\u5fc6\u4f7f\u7528\u548c\u7b54\u6848\u8d28\u91cf\u7684\u7efc\u5408\u8bc4\u4f30\u3002\n\n    \u53c2\u6570:\n        data (DataProto): \u5305\u542b\u6279\u6b21\u6570\u636e\u7684\u5bf9\u8c61\uff0c\u9700\u5305\u542b\u4ee5\u4e0b\u5185\u5bb9:\n            - batch: \u5305\u542b 'prompts'\u3001'responses'\u3001'attention_mask' \u7684\u5b57\u5178\n            - meta_info: \u5305\u542b 'tool_action_stats' \u548c 'memory_action_stats' \u7684\u7edf\u8ba1\u4fe1\u606f\n            - non_tensor_batch: \u5305\u542b 'reward_model' \u7684\u67e5\u8be2\u6570\u636e\n        tool_reward (float): \u4f7f\u7528\u5de5\u5177\u7684\u5956\u52b1\u5206\u6570\n        memory_reward (float): \u4f7f\u7528\u8bb0\u5fc6\u7684\u5956\u52b1\u5206\u6570\n        answer_format_reward (float): \u7b54\u6848\u683c\u5f0f\u7684\u5956\u52b1\u5206\u6570\n        final_answer_reward (float): \u6700\u7ec8\u7b54\u6848\u7684\u5956\u52b1\u5206\u6570\n        tokenizer: \u7528\u4e8e\u89e3\u7801token\u5e8f\u5217\u7684\u5206\u8bcd\u5668\u5bf9\u8c61\n\n    \u8fd4\u56de:\n        torch.Tensor: \u5f62\u72b6\u4e3a (batch_size, response_length) \u7684\u5956\u52b1\u5f20\u91cf\uff0c\n                     \u4ec5\u5728\u6bcf\u4e2a\u5e8f\u5217\u7684\u6700\u540e\u4e00\u4e2a\u6709\u6548token\u4f4d\u7f6e\u8bbe\u7f6e\u5956\u52b1\u503c\n\n    \u6ce8\u610f:\n        1. \u5956\u52b1\u8ba1\u7b97\u903b\u8f91:\n           - \u5982\u679c\u4f7f\u7528\u4e86\u5de5\u5177: reward = tool_reward + final_answer_reward\n           - \u5982\u679c\u4f7f\u7528\u4e86\u8bb0\u5fc6: reward = memory_reward + final_answer_reward\n           - \u5176\u4ed6\u60c5\u51b5: reward = final_answer_reward\n        \n        2. \u51fd\u6570\u4f1a\u66f4\u65b0 meta_info \u4e2d\u7684\u4ee5\u4e0b\u7edf\u8ba1\u4fe1\u606f:\n           - reward/tool_rewards\n           - reward/memory_rewards\n           - reward/answer_format_rewards\n           - reward/final_answer_rewards\n    "
    batch_size = len(data)
    all_scores = []
    # breakpoint()
    meta_info = data.meta_info

    response_length = data.batch['responses'].size(1)
    reward_tensor = torch.zeros((batch_size, response_length), dtype=torch.float32)

    print(f"\u6279\u6b21\u5927\u5c0f batch_size: {batch_size}")
    print(f"\u56de\u590d\u957f\u5ea6 response_length: {response_length}")

    tool_stats = data.meta_info.get("tool_action_stats", [0] * batch_size)
    memory_stats = data.meta_info.get("memory_action_stats", [0] * batch_size)

    # breakpoint()

    query_datas = data.non_tensor_batch['reward_model']

    tool_rewards = []
    memory_rewards = []
    answer_format_rewards = []
    final_answer_rewards = []

    print(f"tool_stats: {tool_stats}")
    print(f"memory_stats: {memory_stats}")

    # if 1 in tool_stats:
    for i in range(batch_size):
        data_item = data[i]

        prompts = data_item.batch['prompts'].unsqueeze(0)
        responses = data_item.batch['responses'].unsqueeze(0)
        attention_mask = data_item.batch['attention_mask'].unsqueeze(0)

        # print(f"prompts shape: {prompts.shape}")
        # print(f"responses shape: {responses.shape}")
        # print(f"attention_mask shape: {attention_mask.shape}")

        prompt_length = prompts.size(1)
        valid_prompt_length = int(attention_mask[0, :prompt_length].sum().item())

        # print(f"prompt_length: {prompt_length}")
        # print(f"valid_prompt_length: {valid_prompt_length}")

        valid_prompt_ids = prompts[:, -valid_prompt_length:]

        valid_response_length = int(attention_mask[0, prompt_length:].sum().item())

        valid_response_ids = responses[:, :valid_response_length]

        sequences = torch.cat((valid_prompt_ids, valid_response_ids), dim=1)  # shape (1, L_effective)

        sequences_str = tokenizer.decode(sequences[0].tolist())

        # query_data = ast.literal_eval(query_datas[i]['ground_truth'])
        query_data = query_datas[i]['ground_truth']
        final_answer_reward, answer_format_score, final_answer_reward = calculate_final_answer_reward(sequences_str, query_data)
        answer_format_rewards.append(answer_format_score)
        final_answer_rewards.append(final_answer_reward)

        all_reward = 0.0

        if tool_stats[i] > 0:
            tool_rewards.append(tool_reward)
            all_reward = tool_reward + final_answer_reward
            # tool_rewards.append(0.0)
            # reward_value = final_answer_reward
        else:
            tool_rewards.append(0.0)
            all_reward = final_answer_reward

        if memory_stats[i] > 0:
            memory_rewards.append(memory_reward)
            all_reward = memory_reward + final_answer_reward
        else:
            memory_rewards.append(0.0)
            all_reward = final_answer_reward

        all_scores.append(all_reward)

        if valid_response_length > 0:
            reward_tensor[i, valid_response_length - 1] = all_reward

    meta_info['reward/tool_rewards'] = tool_rewards
    meta_info['reward/memory_rewards'] = memory_rewards
    meta_info['reward/answer_format_rewards'] = answer_format_rewards
    meta_info['reward/final_answer_rewards'] = final_answer_rewards
    meta_info['reward/all_scores'] = all_scores
    data.meta_info.update(meta_info)

    return reward_tensor

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com

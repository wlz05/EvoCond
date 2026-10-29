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
    'Extract <answer> content from decoded text and score it against optional hard constraints.'
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
        print(f"DEBUG: JSON parsing failed - {e}")
        return total_reward, answer_format_reward, final_answer_reward
    except Exception as e:
        print(f"DEBUG: Unknown parsing error - {e}")
        return total_reward, answer_format_reward, final_answer_reward

    try:
        final_answer_reward = evaluate_plan(hard_constraint_data, test_data)
        answer_format_reward = 1.0
        total_reward = total_reward + answer_format_reward + final_answer_reward
    except Exception as e:
        print(f"DEBUG: evaluate_plan failed - {e}")
        return total_reward, answer_format_reward, final_answer_reward

    return total_reward, answer_format_reward, final_answer_reward

def compute_score_fn(data: DataProto,
                     tool_reward: float,
                     memory_reward: float,
                     answer_format_reward: float,
                     tokenizer=None):
    'Score tool use, memory use, answer format and final answers; place reward on the last valid response token and update reward metrics in meta_info.'
    batch_size = len(data)
    all_scores = []
    # breakpoint()
    meta_info = data.meta_info

    response_length = data.batch['responses'].size(1)
    reward_tensor = torch.zeros((batch_size, response_length), dtype=torch.float32)

    print(f"Batch size: {batch_size}")
    print(f"Response length: {response_length}")

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

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com

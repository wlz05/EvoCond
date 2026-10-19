# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import re
from typing import Dict, List
import json

def parse_freethink(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    '\n    Parse response in format: <think>...</think><answer>...</answer>\n\n    Returns a dict with keys:\n    - llm_raw_response: the original response\n    - llm_response: the response with <think> and <answer> tags\n    - think_content: the content inside <think> tag\n    - action_content: the content inside <answer> tag\n    - actions: a list of actions extracted from action_content\n    - format_correct: whether the response strictly follows the expected format\n    '
    response = response.replace("<image>", "")
    # Pattern to check for content strictly in the format <think>...</think><answer>...</answer>
    strict_pattern = r'^\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)

    # Pattern to extract content from think and answer tags
    extraction_pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
    match = re.search(extraction_pattern, response, re.DOTALL)
    format_correct = strict_match is not None

    if not strict_match:
        think_content, action_content, actions = "", "", []
    else:
        think_content, action_content = match.group(1), match.group(2)
        if special_token_list is not None:
            for special_token in special_token_list:  # remove all special tokens in responses to forbid confusion in training
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]  # Only the first MAX_ACTIONS actions are kept in the rollout.
            action_content = (" " + action_sep + " ").join(actions)

    llm_response = "<think>" + think_content.strip() + "</think>" + "<answer>" + action_content.strip() + "</answer>"
    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct
    }

def parse_no_think(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    '\n    Parse response in format: <answer>...</answer>\n\n    Returns a dict with keys:\n    - llm_raw_response: the original response\n    - llm_response: the response with <answer> tag\n    - think_content: empty string (no think content in this format)\n    - action_content: the content inside <answer> tag\n    - actions: a list of actions extracted from action_content\n    - format_correct: whether the response strictly follows the expected format\n    '
    response = response.replace("<image>", "")
    # Pattern to check for content strictly in the format <answer>...</answer>
    strict_pattern = r'^\s*<answer>(.*?)</answer>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    format_correct = strict_match is not None

    # Pattern to extract content from answer tag
    extraction_pattern = r'<answer>(.*?)</answer>'
    match = re.search(extraction_pattern, response, re.DOTALL)
    # format_correct = match is not None

    if not strict_match:
        think_content, action_content, actions = "", "", []
    else:
        action_content = match.group(1)
        think_content = ""  # No think content in this format
        if special_token_list is not None:
            for special_token in special_token_list:
                action_content = action_content.replace(special_token, "").strip()
        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = (" " + action_sep + " ").join(actions)

    llm_response = "<answer>" + action_content.strip() + "</answer>"
    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct
    }

def parse_grounding(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    '\n    Parse response in format: <think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>\n\n    Returns a dict with keys:\n    - llm_raw_response: the original response\n    - llm_response: the response with all tags\n    - observation_content: the content inside <observation> tag\n    - think_content: the entire content inside <think> tag\n    - reasoning_content: the content inside <reasoning> tag\n    - action_content: the content inside <answer> tag\n    - actions: a list of actions extracted from action_content\n    - format_correct: whether the response strictly follows the expected format\n    '
    response = response.replace("<image>", "")
    # Pattern to check for content strictly in the expected format
    strict_pattern = r'^\s*<think>\s*<observation>(.*?)</observation>\s*<reasoning>(.*?)</reasoning>\s*</think>\s*<answer>(.*?)</answer>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    format_correct = strict_match is not None

    # Pattern to extract content from tags
    extraction_pattern = r'<think>\s*<observation>(.*?)</observation>\s*<reasoning>(.*?)</reasoning>\s*</think>\s*<answer>(.*?)</answer>'
    match = re.search(extraction_pattern, response, re.DOTALL)

    if not match:
        observation_content, reasoning_content, action_content, actions = "", "", "", []
        think_content = ""
    else:
        observation_content = match.group(1)
        reasoning_content = match.group(2)
        action_content = match.group(3)
        think_content = "<observation>" + observation_content + "</observation><reasoning>" + reasoning_content + "</reasoning>"

        if special_token_list is not None:
            for special_token in special_token_list:
                observation_content = observation_content.replace(special_token, "").strip()
                reasoning_content = reasoning_content.replace(special_token, "").strip()
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()

        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = (" " + action_sep + " ").join(actions)

    # Reconstruct the cleaned llm_response
    llm_response = "<think>" + think_content.strip() + "</think>" + "<answer>" + action_content.strip() + "</answer>"

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "observation_content": observation_content,
        "think_content": think_content,
        "reasoning_content": reasoning_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct
    }

def parse_worldmodeling(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    '\n    Parse response in format: <think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>\n\n    Returns a dict with keys:\n    - llm_raw_response: the original response\n    - llm_response: the response with all tags\n    - think_content: the entire content inside <think> tag\n    - reasoning_content: the content inside <reasoning> tag\n    - prediction_content: the content inside <prediction> tag\n    - action_content: the content inside <answer> tag\n    - actions: a list of actions extracted from action_content\n    - format_correct: whether the response strictly follows the expected format\n    '
    response = response.replace("<image>", "")
    # Pattern to check for content strictly in the expected format
    strict_pattern = r'^\s*<think>\s*<reasoning>(.*?)</reasoning>\s*<prediction>(.*?)</prediction>\s*</think>\s*<answer>(.*?)</answer>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    format_correct = strict_match is not None

    # Pattern to extract content from tags
    extraction_pattern = r'<think>\s*<reasoning>(.*?)</reasoning>\s*<prediction>(.*?)</prediction>\s*</think>\s*<answer>(.*?)</answer>'
    match = re.search(extraction_pattern, response, re.DOTALL)

    if not match:
        reasoning_content, prediction_content, action_content, actions = "", "", "", []
        think_content = ""
    else:
        reasoning_content = match.group(1)
        prediction_content = match.group(2)
        action_content = match.group(3)
        think_content = "<reasoning>" + reasoning_content + "</reasoning><prediction>" + prediction_content + "</prediction>"

        if special_token_list is not None:
            for special_token in special_token_list:
                reasoning_content = reasoning_content.replace(special_token, "").strip()
                prediction_content = prediction_content.replace(special_token, "").strip()
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()

        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = (" " + action_sep + " ").join(actions)

    # Reconstruct the cleaned llm_response
    llm_response = "<think>" + think_content.strip() + "</think>" + "<answer>" + action_content.strip() + "</answer>"

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "reasoning_content": reasoning_content,
        "prediction_content": prediction_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct
    }

def parse_grounding_worldmodeling(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    '\n    Parse response in format: <think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>\n\n    Returns a dict with keys:\n    - llm_raw_response: the original response\n    - llm_response: the response with all tags\n    - observation_content: the content inside <observation> tag\n    - reasoning_content: the content inside <reasoning> tag\n    - prediction_content: the content inside <prediction> tag\n    - think_content: the entire content inside <think> tag\n    - action_content: the content inside <answer> tag\n    - actions: a list of actions extracted from action_content\n    - format_correct: whether the response strictly follows the expected format\n    '
    response = response.replace("<image>", "")
    # Pattern to check for content strictly in the expected format
    strict_pattern = r'^\s*<think>\s*<observation>(.*?)</observation>\s*<reasoning>(.*?)</reasoning>\s*<prediction>(.*?)</prediction>\s*</think>\s*<answer>(.*?)</answer>\s*$'
    strict_match = re.match(strict_pattern, response.strip(), re.DOTALL)
    format_correct = strict_match is not None

    # Pattern to extract content from tags
    extraction_pattern = r'<think>\s*<observation>(.*?)</observation>\s*<reasoning>(.*?)</reasoning>\s*<prediction>(.*?)</prediction>\s*</think>\s*<answer>(.*?)</answer>'
    match = re.search(extraction_pattern, response, re.DOTALL)

    if not match:
        observation_content, reasoning_content, prediction_content, action_content, actions = "", "", "", "", []
        think_content = ""
    else:
        observation_content = match.group(1)
        reasoning_content = match.group(2)
        prediction_content = match.group(3)
        action_content = match.group(4)
        think_content = "<observation>" + observation_content + "</observation><reasoning>" + reasoning_content + "</reasoning><prediction>" + prediction_content + "</prediction>"

        if special_token_list is not None:
            for special_token in special_token_list:
                observation_content = observation_content.replace(special_token, "").strip()
                reasoning_content = reasoning_content.replace(special_token, "").strip()
                prediction_content = prediction_content.replace(special_token, "").strip()
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()

        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = (" " + action_sep + " ").join(actions)

    # Reconstruct the cleaned llm_response
    llm_response = "<think>" + think_content.strip() + "</think>" + "<answer>" + action_content.strip() + "</answer>"

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "observation_content": observation_content,
        "reasoning_content": reasoning_content,
        "prediction_content": prediction_content,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct
    }

'\n\u5728\u591a\u79cd\u73af\u5883\u6216\u7b56\u7565\u4e0b\uff0cLLM Agent \u7684\u8f93\u51fa\u683c\u5f0f\u4e0d\u540c\uff0c\u6bd4\u5982\uff1a\n\u201cfree_think\u201d\uff1a\u81ea\u7531\u601d\u8003\uff0c\u53ef\u80fd\u5305\u542b <think>...</think><ans>...</ans>\n\u201cgrounding\u201d\uff1a\u5f3a\u8c03\u201c\u611f\u77e5\u5230\u7684\u73af\u5883\u201d\u4fe1\u606f\n\u201csymbolic\u201d\uff1a\u4f7f\u7528\u7b26\u53f7\u903b\u8f91\u7ed3\u6784\n\u201cstructured\u201d\uff1a\u7ed3\u6784\u5316\u63d0\u793a\n'
PARSE_FUNC_MAP = {
    "free_think": parse_freethink,
    "no_think": parse_no_think,
    "grounding": parse_grounding,
    "worldmodeling": parse_worldmodeling,
    "grounding_worldmodeling": parse_grounding_worldmodeling,
    "grounding_structured": parse_grounding,
    "worldmodeling_structured": parse_worldmodeling,
    "grounding_worldmodeling_structured": parse_grounding_worldmodeling,
    "grounding_symbolic": parse_grounding,
    "worldmodeling_symbolic": parse_worldmodeling,
    "grounding_worldmodeling_symbolic": parse_grounding_worldmodeling,
}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com

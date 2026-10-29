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
'\nMetrics related to the PPO trainer.\n'

from collections import defaultdict
from functools import partial
from typing import Any, Callable

import numpy as np
import torch

from verl import DataProto
from verl.utils.imports import deprecated

@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    '\n    Reduces a dictionary of metric lists by computing the mean of each list.\n\n    Args:\n        metrics: A dictionary mapping metric names to lists of metric values.\n\n    Returns:\n        A dictionary with the same keys but with each list replaced by its mean value.\n\n    Example:\n        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}\n        >>> reduce_metrics(metrics)\n        {"loss": 2.0, "accuracy": 0.8}\n    '
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)

def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    '\n    Computes information about prompts and responses from a batch.\n\n    This is an internal helper function that extracts masks and lengths for prompts and responses.\n\n    Args:\n        batch: A DataProto object containing batch data with responses and attention masks.\n\n    Returns:\n        A dictionary containing:\n            - response_mask: Attention mask for the response tokens\n            - prompt_length: Tensor of prompt lengths for each item in the batch\n            - response_length: Tensor of response lengths for each item in the batch\n    '
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )

def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    '\n    Computes various metrics from a batch of data for PPO training.\n\n    This function calculates metrics related to scores, rewards, advantages, returns, values,\n    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)\n    for each metric category.\n\n    Args:\n        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.\n        use_critic: Whether to include critic-specific metrics. Defaults to True.\n\n    Returns:\n        A dictionary of metrics including:\n            - critic/score/mean, max, min: Statistics about sequence scores\n            - critic/rewards/mean, max, min: Statistics about sequence rewards\n            - critic/advantages/mean, max, min: Statistics about advantages\n            - critic/returns/mean, max, min: Statistics about returns\n            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)\n            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)\n            - response_length/mean, max, min, clip_ratio: Statistics about response lengths\n            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths\n            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations\n    '
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["response_mask"].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # multi-turn conversation
    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    return metrics

def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    '\n    Computes timing metrics for different processing stages in PPO training.\n\n    This function calculates both raw timing metrics (in seconds) and per-token timing metrics\n    (in milliseconds) for various processing stages like generation, reference computation,\n    value computation, advantage computation, and model updates.\n\n    Args:\n        batch: A DataProto object containing batch data with responses and attention masks.\n        timing_raw: A dictionary mapping stage names to their execution times in seconds.\n\n    Returns:\n        A dictionary containing:\n            - timing_s/{name}: Raw timing in seconds for each stage\n            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage\n\n    Note:\n        Different stages use different token counts for normalization:\n        - "gen" uses only response tokens\n        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens\n          (prompt + response)\n    '
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }

def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    '\n    Computes throughput metrics for PPO training.\n\n    This function calculates performance metrics related to token processing speed,\n    including the total number of tokens processed, time per step, and throughput\n    (tokens per second per GPU).\n\n    Args:\n        batch: A DataProto object containing batch data with meta information about token counts.\n        timing_raw: A dictionary mapping stage names to their execution times in seconds.\n                   Must contain a "step" key with the total step time.\n        n_gpus: Number of GPUs used for training.\n\n    Returns:\n        A dictionary containing:\n            - perf/total_num_tokens: Total number of tokens processed in the batch\n            - perf/time_per_step: Time taken for the step in seconds\n            - perf/throughput: Tokens processed per second per GPU\n\n    Note:\n        The throughput is calculated as total_tokens / (time * n_gpus) to normalize\n        across different GPU counts.\n    '
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)

    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }

def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    '\n    Performs bootstrap resampling to estimate statistics of metrics.\n\n    This function uses bootstrap resampling to estimate the mean and standard deviation\n    of metrics computed by the provided reduction functions on random subsets of the data.\n\n    Args:\n        data: List of data points to bootstrap from.\n        subset_size: Size of each bootstrap sample.\n        reduce_fns: List of functions that compute a metric from a subset of data.\n        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.\n        seed: Random seed for reproducibility. Defaults to 42.\n\n    Returns:\n        A list of tuples, where each tuple contains (mean, std) for a metric\n        corresponding to each reduction function in reduce_fns.\n\n    Example:\n        >>> data = [1, 2, 3, 4, 5]\n        >>> reduce_fns = [np.mean, np.max]\n        >>> bootstrap_metric(data, 3, reduce_fns)\n        [(3.0, 0.5), (4.5, 0.3)]  # Example values\n    '
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]

def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    '\n    Calculate a value based on majority voting.\n\n    This function identifies the most common value for a specified vote key\n    in the data, then returns the corresponding value for that majority vote.\n\n    Args:\n        data: List of dictionaries, where each dictionary contains both vote_key and val_key.\n        vote_key: The key in each dictionary used for voting/counting.\n        val_key: The key in each dictionary whose value will be returned for the majority vote.\n\n    Returns:\n        The value associated with the most common vote.\n\n    Example:\n        >>> data = [\n        ...     {"pred": "A", "val": 0.9},\n        ...     {"pred": "B", "val": 0.8},\n        ...     {"pred": "A", "val": 0.7}\n        ... ]\n        >>> calc_maj_val(data, vote_key="pred", val_key="val")\n        0.9  # Returns the first "val" for the majority vote "A"\n    '
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val

def process_validation_metrics(
    data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    '\n    Process validation metrics into a structured format with statistical analysis.\n\n    This function organizes validation metrics by data source and prompt, then computes\n    various statistical measures including means, standard deviations, best/worst values,\n    and majority voting results. It also performs bootstrap sampling to estimate statistics\n    for different sample sizes.\n\n    Args:\n        data_sources: List of data source identifiers for each sample.\n        sample_inputs: List of input prompts corresponding to each sample.\n        infos_dict: Dictionary mapping variable names to lists of values for each sample.\n        seed: Random seed for bootstrap sampling. Defaults to 42.\n\n    Returns:\n        A nested dictionary with the structure:\n        {\n            data_source: {\n                variable_name: {\n                    metric_name: value\n                }\n            }\n        }\n\n        Where metric_name includes:\n        - "mean@N": Mean value across N samples\n        - "std@N": Standard deviation across N samples\n        - "best@N/mean": Mean of the best values in bootstrap samples of size N\n        - "best@N/std": Standard deviation of the best values in bootstrap samples\n        - "worst@N/mean": Mean of the worst values in bootstrap samples\n        - "worst@N/std": Standard deviation of the worst values in bootstrap samples\n        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)\n        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)\n\n    Example:\n        >>> data_sources = ["source1", "source1", "source2"]\n        >>> sample_inputs = ["prompt1", "prompt1", "prompt2"]\n        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}\n        >>> result = process_validation_metrics(data_sources, sample_inputs, infos_dict)\n        >>> # result will contain statistics for each data source and variable\n    '
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                            data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
                        )
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"], strict=True)
                            ]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val

def compute_reward_breakdown_metrics(batch: DataProto) -> dict[str, Any]:
    "\n    Computes fine-grained reward component metrics from a batch.\n\n    This function extracts detailed reward breakdown information from the batch's\n    non_tensor_batch, providing insights into different reward components:\n    - format_reward: Reward for correct output formatting\n    - tool_penalty: Penalty/bonus for tool usage\n    - answer_correctness: Correctness of the final answer\n    - num_tool_calls: Actual number of tool calls made\n    - expected_tool_calls: Expected number of tool calls\n\n    Args:\n        batch: A DataProto object containing batch data with non_tensor_batch\n               that includes reward breakdown information\n\n    Returns:\n        A dictionary of metrics including:\n            - reward/format_reward/mean, max, min: Statistics about format rewards\n            - reward/tool_penalty/mean, max, min: Statistics about tool usage penalties\n            - reward/answer_correctness/mean: Average answer correctness\n            - reward/correct_answer_ratio: Ratio of correct answers (correctness > 0)\n            - tools/num_calls/mean, max, min: Statistics about actual tool calls\n            - tools/expected_calls/mean: Average expected tool calls\n            - tools/call_deviation/mean: Average deviation from expected calls\n            - tools/over_use_ratio: Ratio of samples with more calls than expected\n            - tools/under_use_ratio: Ratio of samples with fewer calls than expected\n            - tools/exact_match_ratio: Ratio of samples with exact expected calls\n\n    Note:\n        The batch.non_tensor_batch should contain 'reward_breakdown' which is a list\n        of dictionaries with keys: format_reward, tool_penalty, answer_correctness,\n        num_tool_calls, expected_tool_calls. This data comes from the modified\n        compute_score() function in rewards.py.\n    "
    # Extract reward breakdown from batch
    if "reward_breakdown" not in batch.non_tensor_batch:
        # Return empty dict if no breakdown data available
        return {}

    reward_breakdowns = batch.non_tensor_batch["reward_breakdown"]

    # Extract individual components
    format_rewards = [r.get("format_reward", 0.0) for r in reward_breakdowns]
    tool_penalties = [r.get("tool_penalty", 0.0) for r in reward_breakdowns]
    answer_correctness = [r.get("answer_correctness", 0.0) for r in reward_breakdowns]
    num_tool_calls = [r.get("num_tool_calls", 0) for r in reward_breakdowns]
    expected_tool_calls = [r.get("expected_tool_calls", 0) for r in reward_breakdowns]

    # Convert to tensors for easier computation
    format_rewards_t = torch.tensor(format_rewards, dtype=torch.float32)
    tool_penalties_t = torch.tensor(tool_penalties, dtype=torch.float32)
    answer_correctness_t = torch.tensor(answer_correctness, dtype=torch.float32)
    num_tool_calls_t = torch.tensor(num_tool_calls, dtype=torch.float32)
    expected_tool_calls_t = torch.tensor(expected_tool_calls, dtype=torch.float32)

    # Compute metrics
    metrics = {
        # Format reward statistics
        "reward/format_reward/mean": torch.mean(format_rewards_t).item(),
        "reward/format_reward/max": torch.max(format_rewards_t).item(),
        "reward/format_reward/min": torch.min(format_rewards_t).item(),
        # Tool penalty statistics
        "reward/tool_penalty/mean": torch.mean(tool_penalties_t).item(),
        "reward/tool_penalty/max": torch.max(tool_penalties_t).item(),
        "reward/tool_penalty/min": torch.min(tool_penalties_t).item(),
        # Answer correctness
        "reward/answer_correctness/mean": torch.mean(answer_correctness_t).item(),
        "reward/correct_answer_ratio": torch.mean((answer_correctness_t > 0).float()).item(),
        # Tool usage statistics
        "tools/num_calls/mean": torch.mean(num_tool_calls_t).item(),
        "tools/num_calls/max": torch.max(num_tool_calls_t).item(),
        "tools/num_calls/min": torch.min(num_tool_calls_t).item(),
        "tools/expected_calls/mean": torch.mean(expected_tool_calls_t).item(),
    }

    # Compute tool usage deviation metrics
    if len(num_tool_calls) > 0 and len(expected_tool_calls) > 0:
        call_deviation = torch.abs(num_tool_calls_t - expected_tool_calls_t)
        over_use = (num_tool_calls_t > expected_tool_calls_t).float()
        under_use = (num_tool_calls_t < expected_tool_calls_t).float()
        exact_match = (num_tool_calls_t == expected_tool_calls_t).float()

        metrics.update({
            "tools/call_deviation/mean": torch.mean(call_deviation).item(),
            "tools/over_use_ratio": torch.mean(over_use).item(),
            "tools/under_use_ratio": torch.mean(under_use).item(),
            "tools/exact_match_ratio": torch.mean(exact_match).item(),
        })

    return metrics

def compute_mcp_metrics(centralized_actor_handle, timeout: float = 5.0) -> dict[str, Any]:
    '\n    Computes MCP (Model-Created Primitives) tool statistics from the centralized actor.\n\n    This function retrieves statistics about dynamically created MCP tools, providing\n    insights into tool evolution during training:\n    - total_mcps: Total number of MCP tools available\n    - official_mcps: Number of officially registered tools\n    - temp_mcps: Number of temporary/experimental tools\n    - mcp_version: Version number indicating tool list updates\n    - graph_nodes/edges: Statistics about the tool dependency graph\n\n    Args:\n        centralized_actor_handle: Ray actor handle for the CentralizedToolActor\n        timeout: Maximum time to wait for the remote call in seconds. Defaults to 5.0.\n\n    Returns:\n        A dictionary of metrics including:\n            - mcp/total_count: Total number of MCP tools\n            - mcp/official_count: Number of official MCP tools\n            - mcp/temp_count: Number of temporary MCP tools\n            - mcp/version: MCP version number (increments on updates)\n            - mcp/temp_ratio: Ratio of temporary to total MCPs\n            - mcp/graph_nodes: Number of nodes in the tool dependency graph\n            - mcp/graph_edges: Number of edges in the tool dependency graph\n            - mcp/avg_edges_per_node: Average edges per node in the graph\n\n        Returns empty dict if the remote call fails or times out.\n\n    Note:\n        This function requires the CentralizedToolActor to have a get_mcp_statistics()\n        method. The actor handle is typically obtained via:\n        ray.get_actor("centralized_tool_actor")\n    '
    try:
        import ray

        # Get MCP statistics from the centralized actor
        mcp_stats = ray.get(
            centralized_actor_handle.get_mcp_statistics.remote(),
            timeout=timeout
        )

        # Build metrics dictionary
        metrics = {
            "mcp/total_count": mcp_stats["total_mcps"],
            "mcp/official_count": mcp_stats["official_mcps"],
            "mcp/temp_count": mcp_stats["temp_mcps"],
            "mcp/version": mcp_stats["mcp_version"],
            "mcp/graph_nodes": mcp_stats["graph_nodes"],
            "mcp/graph_edges": mcp_stats["graph_edges"],
        }

        # Compute derived metrics
        if mcp_stats["total_mcps"] > 0:
            metrics["mcp/temp_ratio"] = mcp_stats["temp_mcps"] / mcp_stats["total_mcps"]

        if mcp_stats["graph_nodes"] > 0:
            metrics["mcp/avg_edges_per_node"] = mcp_stats["graph_edges"] / mcp_stats["graph_nodes"]

        return metrics

    except Exception as e:
        # Log warning but don't crash training
        print(f"Warning: Failed to compute MCP metrics: {e}")
        return {}

def compute_gigpo_metrics() -> dict[str, Any]:
    "\n    Computes GiGPO algorithm statistics for monitoring.\n\n    This function retrieves the latest GiGPO statistics that were computed during\n    the advantage estimation step. These statistics provide insights into:\n    - Trajectory grouping (number of groups, group sizes)\n    - Anchor observation quality (NaN ratios)\n    - Advantage distributions (episode-level and step-level)\n    - Final score statistics\n\n    Returns:\n        A dictionary of metrics including:\n            - gigpo/num_groups: Number of step-level groups formed\n            - gigpo/avg_group_size: Average size of each group\n            - gigpo/max_group_size: Maximum group size\n            - gigpo/min_group_size: Minimum group size\n            - gigpo/single_groups_ratio: Ratio of groups with only one sample\n            - gigpo/anchor_obs_nan_ratio: Ratio of NaN values in anchor observations\n            - gigpo/episode_adv_mean: Mean of episode-level advantages\n            - gigpo/episode_adv_std: Standard deviation of episode-level advantages\n            - gigpo/step_adv_mean: Mean of step-level advantages\n            - gigpo/step_adv_std: Standard deviation of step-level advantages\n            - gigpo/final_scores_mean: Mean of final combined scores\n            - gigpo/final_scores_std: Standard deviation of final scores\n            - gigpo/final_scores_min: Minimum final score\n            - gigpo/final_scores_max: Maximum final score\n            - gigpo/step_advantage_weight: Weight used for step-level advantages\n\n        Returns empty dict if GiGPO hasn't been called yet or if an error occurs.\n\n    Note:\n        This function reads from a global variable that is updated by the\n        compute_gigpo_outcome_advantage function in algorithms.py. Call this\n        function after the advantage computation step to get the latest statistics.\n\n    Example:\n        >>> # In training loop, after advantage computation\n        >>> metrics.update(compute_gigpo_metrics())\n    "
    try:
        from verl.trainer.ppo import algorithms as core_algos

        # Get the latest GiGPO statistics from the global variable
        gigpo_stats = getattr(core_algos, '_last_gigpo_stats', {})

        if not gigpo_stats:
            # No GiGPO stats available yet
            return {}

        # Build metrics dictionary with proper prefixing
        metrics = {}

        # Grouping statistics
        if "num_groups" in gigpo_stats:
            metrics["gigpo/num_groups"] = gigpo_stats["num_groups"]
            metrics["gigpo/avg_group_size"] = gigpo_stats["avg_group_size"]
            metrics["gigpo/max_group_size"] = gigpo_stats["max_group_size"]
            metrics["gigpo/min_group_size"] = gigpo_stats["min_group_size"]
            metrics["gigpo/single_groups_ratio"] = gigpo_stats["single_groups_ratio"]

        # Anchor observation quality
        if "anchor_obs_nan_ratio" in gigpo_stats:
            metrics["gigpo/anchor_obs_nan_ratio"] = gigpo_stats["anchor_obs_nan_ratio"]
            metrics["gigpo/anchor_obs_nan_count"] = gigpo_stats["anchor_obs_nan_count"]

        # Advantage statistics
        if "episode_adv_mean" in gigpo_stats:
            metrics["gigpo/episode_adv_mean"] = gigpo_stats["episode_adv_mean"]
            metrics["gigpo/episode_adv_std"] = gigpo_stats["episode_adv_std"]

        if "step_adv_mean" in gigpo_stats:
            metrics["gigpo/step_adv_mean"] = gigpo_stats["step_adv_mean"]
            metrics["gigpo/step_adv_std"] = gigpo_stats["step_adv_std"]

        # Final score statistics
        if "final_scores_mean" in gigpo_stats:
            metrics["gigpo/final_scores_mean"] = gigpo_stats["final_scores_mean"]
            metrics["gigpo/final_scores_std"] = gigpo_stats["final_scores_std"]
            metrics["gigpo/final_scores_min"] = gigpo_stats["final_scores_min"]
            metrics["gigpo/final_scores_max"] = gigpo_stats["final_scores_max"]

        # Algorithm parameters
        if "step_advantage_weight" in gigpo_stats:
            metrics["gigpo/step_advantage_weight"] = gigpo_stats["step_advantage_weight"]

        return metrics

    except Exception as e:
        # Log warning but don't crash training
        print(f"Warning: Failed to compute GiGPO metrics: {e}")
        return {}

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com

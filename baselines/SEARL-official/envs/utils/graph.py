# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import os
import logging
import asyncio
import itertools

import numpy as np
import ray
import torch
from verl import DataProto
from tensordict import TensorDict
from time import time
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))
import zlib

def pad_matrix_to_shape(data, target_shape, pad_value=np.nan):
    'Pad ragged nested lists to (n, m, k); return a NumPy array using pad_value, default NaN.'
    n, m, k = target_shape

    padded = np.full((n, m, k), pad_value, dtype=object)

    for i in range(min(n, len(data))):
        if i < len(data):
            for j in range(min(m, len(data[i]))):
                if j < len(data[i]):
                    for l in range(min(k, len(data[i][j]))):
                        if l < len(data[i][j]):
                            padded[i, j, l] = data[i][j][l]

    return padded

def auto_pad_matrix(data, pad_value=np.nan):
    'Infer dimensions and pad nested lists to a rectangular array.'
    if not data:
        return np.array([])

    max_dim1 = len(data)
    max_dim2 = max(len(item) for item in data) if data else 0
    max_dim3 = 10

    truncated_data = []
    for item in data:
        if item is None:
            truncated_data.append([])
            continue
        truncated_item = []
        for subitem in item:
            if isinstance(subitem, list):

                truncated_item.append(subitem[:10] if len(subitem) > 10 else subitem)
            else:

                truncated_item.append(subitem)
        truncated_data.append(truncated_item)

    return pad_matrix_to_shape(truncated_data, (max_dim1, max_dim2, max_dim3), pad_value)

class ToolUtilsGraph:
    def __init__(self, tokenizer, meta_info, config, env_object, is_validate=False):
        self.tokenizer = tokenizer
        self.final_str = config.stop[-1] if config.stop else ''
        self.config_prompt_length = config.prompt_length
        self.config_response_length = config.response_length
        self.stop_id = self.tokenizer.encode(config.stop[0], add_special_tokens=False)[0]
        self.max_turns = config.max_turns
        self.max_prompt_length = config.prompt_length
        self.max_tool_response_length = config.tool_response_length
        self.strict_runtime = bool(getattr(env_object.tool_manager.verl_config, 'runtime_config_path', None))
        self.context_budget = config.get('max_model_len', config.prompt_length+config.response_length)-config.response_length
        self.sampling_config = {name: config.get(name) for name in ('temperature', 'top_p', 'top_k')}
        if self.strict_runtime:
            from control.runtime import load_config, verify_rollout_contract
            runtime_config = load_config(env_object.tool_manager.verl_config.runtime_config_path)
            verify_rollout_contract(runtime_config, {'max_turns': self.max_turns,
                'max_context': self.context_budget+self.config_response_length,
                'max_prompt': self.max_prompt_length, 'max_response': self.config_response_length})
            meta_info['sample_tool_turns'] = not is_validate and runtime_config.get('sample_tool_turns', True)
        self.rollout_num = config.n
        self.is_validate = is_validate
        pad_token_id = meta_info.get('pad_token_id')
        if pad_token_id is not None:
            self.pad_token_id = pad_token_id
        else:
            eos_token_id = meta_info.get('eos_token_id')
            if isinstance(eos_token_id, (list, tuple)):
                self.pad_token_id = eos_token_id[-1]
            else:
                self.pad_token_id = eos_token_id

        eos_token_id = meta_info.get('eos_token_id')
        if isinstance(eos_token_id, (list, tuple)):
            self.eos_token_id = eos_token_id[0]
        else:
            self.eos_token_id = eos_token_id

        self.meta_info = meta_info
        self.loop_cnt = 0

        self.env_object = env_object

    def postprocess_output(self, output: DataProto, step: int, questions: list=None, mcp_worthiness: float=None, tool_call_count: int=None, is_validation: bool=False):
        '''output: cpu'''
        import time
        start_time = time.time()

        is_validation = self.is_validate if hasattr(self, 'is_validate') else is_validation

        print(f"[TOOL_UTILS_GRAPH] postprocess_output started for step {step}, loop_cnt {self.loop_cnt}, is_validation={is_validation}")
        print(f"[TOOL_UTILS_GRAPH] Input size: batch_size={output.batch.batch_size[0] if hasattr(output.batch, 'batch_size') else 'unknown'}")
        print(f"[TOOL_UTILS_GRAPH] Start timestamp: {start_time}")
        print(f"[TOOL_UTILS_GRAPH] Memory usage: {torch.cuda.memory_allocated()/1024**3:.2f}GB" if torch.cuda.is_available() else '[TOOL_UTILS_GRAPH] CPU mode')

        # init loop responses token
        if self.loop_cnt == 0:
            self.batch_size = output.batch.batch_size[0]
            self.loop_responses_token = [[] for _ in range(self.batch_size)]
            self.init_prompt_token = output.batch.get('prompts')
            self.tool_use = [[] for _ in range(self.batch_size)]
            self.mcp_ids = [[] for _ in range(self.batch_size)]
            self.tool_call_count = tool_call_count
            prompt_length = self.init_prompt_token.shape[-1]
            self.init_attention_mask = output.batch.get('attention_mask')[:,:prompt_length]

            batch_idxs = list(range(self.batch_size))
            for idx in range(self.batch_size):
                prompt_token = self.init_prompt_token[idx]
                prompt_token_list = prompt_token[prompt_token != self.pad_token_id].tolist()
                self.loop_responses_token[idx].append(prompt_token_list)
        else:
            batch_idxs = output.meta_info['index']

        responses = output.batch.get('responses')
        process_response = []
        for idx, batch_idx in enumerate(batch_idxs):
            response_token = responses[idx]
            response_token_list = response_token[response_token != self.pad_token_id].tolist()
            if not response_token_list:

                response_token_list = [self.stop_id]
            elif self.env_object.use_process_reward:

                if response_token_list[-1] != self.stop_id:
                    if len(response_token_list) != self.config_response_length:
                        response_token_list.append(self.stop_id)
                    else:
                        response_token_list[-1] = self.stop_id
            self.loop_responses_token[batch_idx].append(response_token_list)
            process_response.append(response_token_list)

        # decode responses for env step (detect tool call)
        decode_start = time.time()
        responses_str = self.tokenizer.batch_decode(
            process_response,
            skip_special_tokens=False,
        )
        decode_end = time.time()
        print(f"[TOOL_UTILS_GRAPH] Token decoding time: {decode_end - decode_start:.3f} seconds")

        step_start_time = time.time()
        capturing = getattr(self, 'capture_turn', None) == step
        if self.loop_cnt > 0 and getattr(self.env_object.tool_manager.verl_config, 'runtime_config_path', None):
            import ray
            prompts = output.batch['prompts']
            masks = output.batch['attention_mask'][:, :prompts.shape[-1]]
            for local, batch_idx in enumerate(batch_idxs):
                identity = self.env_object.plan_list[batch_idx]['_runtime_context']
                input_tokens = int(masks[local].sum().item())
                output_tokens = int((responses[local] != self.pad_token_id).sum().item())
                ray.get(self.env_object.tool_manager.centralized_actor_handle.runtime_charge.remote(
                    identity, 'actor', {'input_tokens': input_tokens, 'output_tokens': output_tokens}))
        try:
            infos_str, dones, valid_actions, is_tool, tool_names = self.env_object.step(
                responses=responses_str,
                tokenizer=None if capturing else self.tokenizer,
                loop_cnt=self.loop_cnt,
                questions=questions,
                mcp_worthiness=mcp_worthiness,
                is_validation=is_validation,
                batch_indices=batch_idxs
            )
        except TypeError:
            if getattr(self.env_object.tool_manager.verl_config, 'runtime_config_path', None):
                raise
            infos_str, dones, valid_actions, is_tool, tool_names = self.env_object.step(
                responses=responses_str,
                tokenizer=None if capturing else self.tokenizer,
                loop_cnt=self.loop_cnt,
                questions=questions,
            )
        step_end_time = time.time()
        if self.loop_cnt == 0 and getattr(self.env_object.tool_manager.verl_config, 'runtime_config_path', None):
            import ray
            prompts = output.batch['prompts']
            masks = output.batch['attention_mask'][:, :prompts.shape[-1]]
            for local, batch_idx in enumerate(batch_idxs):
                identity = self.env_object.plan_list[batch_idx]['_runtime_context']
                input_tokens = int(masks[local].sum().item())
                output_tokens = int((responses[local] != self.pad_token_id).sum().item())
                ray.get(self.env_object.tool_manager.centralized_actor_handle.runtime_charge.remote(
                    identity, 'actor', {'input_tokens': input_tokens, 'output_tokens': output_tokens}))
        print(f"[TOOL_UTILS_GRAPH] Environment step time: {step_end_time - step_start_time:.3f} seconds")
        #if not use_process_reward will be 0
        if self.env_object.use_process_reward:
            step_scores = self.env_object.get_step_reward(responses=responses_str, mcp_worthiness=mcp_worthiness)
        else:
            step_scores = [0] * len(responses_str)

        if capturing:
            from envs.utils.continuation import capture
            return capture(self, step, infos_str, dones, valid_actions, is_tool, tool_names, step_scores, batch_idxs)

        from envs.utils.continuation import append_observations
        return append_observations(self, infos_str, dones, tool_names, step_scores, batch_idxs)

    def find_step_id_by_mcp_name(self, mcp_name_list, mcp_dict=None):
        'Map tool names to stable CRC32 integer IDs; an optional MCP mapping supports registration warnings.'
        if mcp_dict is None:
            mcp_dict = {}

        if not isinstance(mcp_name_list, (list, tuple)):
            print(f"Warning: find_step_id_by_mcp_name expected list/tuple, got {type(mcp_name_list)}")
            return [np.nan]

        tool_id_list = []
        for item in mcp_name_list:

            if not isinstance(item, str):
                print(f"Warning: Non-string tool name: {item} (type: {type(item)})")
                tool_id_list.append(np.nan)
                continue

            if item in ["search-search", "execute_python_code-execute_python_code"]:
                tool_id = np.nan
            else:
                tool_id = zlib.crc32(item.encode('utf-8'))

            if item not in mcp_dict:

                if item not in ["plan", "plan failed", "<error>", "<empty>", "None", "search-search", "execute_python_code-execute_python_code"]:
                    print(f"Info: Tool '{item}' not found in registered MCPs (using hash ID {tool_id})")

            tool_id_list.append(tool_id)

        return tool_id_list

    async def _process_single_response_async(self, idx, responses, optimal_idx, mcp_box):
        'Process one trajectory through mcp_box; collect plans for later batch registration.'
        temp_dict = {}
        temp_prompts_list = []
        loss_mask = []

        is_optimal_trajectory = (idx == optimal_idx)
        input_ids_list = list(itertools.chain.from_iterable(responses))

        planning_info = None

        corrected_turns = []
        if getattr(self, 'strict_runtime', False):
            identity = self.env_object.plan_list[idx]['_runtime_context']
            corrected_turns = await mcp_box.runtime_binding_mask.remote(identity)

        for turn_idx, response_turn in enumerate(responses):
            length = len(response_turn)

            if is_optimal_trajectory:
                if not self.is_validate:
                    if turn_idx == 0:
                        plan_str = self.tokenizer.decode(response_turn, skip_special_tokens=True)
                        planning_info = {
                            'trajectory_idx': idx,
                            'plan_str': plan_str,
                            'max_turns': self.max_turns
                        }

            loss_mask.extend([(turn_idx + 1) % 2] * length)

        if corrected_turns:
            from control.binding import response_loss_mask
            loss_mask = response_loss_mask(responses, corrected_turns)

        return {
            'input_ids_list': input_ids_list,
            'loss_mask': loss_mask,
            'total_length': len(input_ids_list),
            'planning_info': planning_info,
            'trajectory_idx': idx
        }

    def compose_final_output(self, step, group) -> DataProto:
        """Compose final generation output."""
        start_time = time()
        input_ids_list = []
        loss_mask_list = []
        length_list = []
        print("Start to compose final output")

        mcp_box_start = time()
        mcp_box = getattr(self.env_object, "centralized_actor_handle", None)
        if mcp_box is None and getattr(self.env_object, "tool_manager", None) is not None:
            mcp_box = getattr(self.env_object.tool_manager, "centralized_actor_handle", None)
        if mcp_box is None:
            raise AttributeError('centralized_actor_handle is not initialized; inject the Ray actor into env_object/tool_manager.')
        print(f"mcp_box lookup time: {time() - mcp_box_start:.4f}s")

        reward_calc_start = time()
        total_rewards = []
        for tool_use in self.tool_use:
            if tool_use:
                total_rewards.append(float(np.nansum(tool_use)))
            else:
                total_rewards.append(0.0)
        optimal_indices = {}
        for start in range(0, len(self.loop_responses_token), self.rollout_num):
            end = start + self.rollout_num
            group_rewards = total_rewards[start:end]
            if not group_rewards:
                continue
            local_best = int(np.argmax(group_rewards))
            for i in range(len(group_rewards)):
                optimal_indices[start + i] = start + local_best
        print(f"Reward and best-index computation time: {time() - reward_calc_start:.4f}s")

        tasks = [
            self._process_single_response_async(idx, responses[1:], optimal_indices.get(idx), mcp_box)
            for idx, responses in enumerate(self.loop_responses_token)
        ]

        async def _run_tasks(ts):
            return await asyncio.gather(*ts)

        async_exec_start = time()
        try:
            gathered_results = asyncio.run(_run_tasks(tasks))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                gathered_results = loop.run_until_complete(_run_tasks(tasks))
            finally:
                loop.close()
        print(f"Asynchronous execution time: {time() - async_exec_start:.4f}s")

        async def _batch_process_async():
            from collections import defaultdict
            batch_process_start = time()
            if getattr(self.env_object.tool_manager.verl_config, 'runtime_config_path', None):
                # The identified runtime already admits immutable isolated tools.
                # Never invoke the historical host-source registration/merge path.
                return {r['trajectory_idx']: r for r in gathered_results if r}, {}

            all_planning_tasks = []
            trajectory_results = {}

            optimal_trajectory_set = set(optimal_indices.values())
            print(f"Best trajectories: {optimal_trajectory_set}")

            for result in gathered_results:
                if result:
                    trajectory_idx = result['trajectory_idx']
                    trajectory_results[trajectory_idx] = result

                    if result.get('planning_info'):
                        all_planning_tasks.append(result['planning_info'])

            planning_results = {}
            if all_planning_tasks:
                print(f"Processing {len(all_planning_tasks)} plans...")
                planning_batch_start = time()

                try:

                    plan_texts = [p['plan_str'] for p in all_planning_tasks]
                    traj_indices = [p['trajectory_idx'] for p in all_planning_tasks]
                    max_turns_list = [p['max_turns'] for p in all_planning_tasks]

                    try:
                        future = mcp_box.batch_extract_dag_and_parallel.remote(plan_texts, traj_indices, max_turns_list)
                        batch_results = await asyncio.wait_for(future, timeout=90.0)

                        for traj_idx, result in zip(traj_indices, batch_results):
                            planning_results[traj_idx] = result

                        planning_batch_duration = time() - planning_batch_start
                        print(f"Batch planning time: {planning_batch_duration:.3f}s (mean {planning_batch_duration/len(all_planning_tasks):.3f}s/item)")
                    except asyncio.TimeoutError:
                        print(f"Warning: batch_extract_dag_and_parallel timeout after 90s, falling back to sequential planning")

                        for planning_info in all_planning_tasks:
                            try:
                                fallback_future = mcp_box.sequential_planning.remote(planning_info['max_turns'])
                                fallback_result = await asyncio.wait_for(fallback_future, timeout=5.0)
                                planning_results[planning_info['trajectory_idx']] = fallback_result
                            except Exception as e:
                                logger.error(f"Sequential planning also failed for trajectory {planning_info['trajectory_idx']}: {e}")
                                planning_results[planning_info['trajectory_idx']] = (None, None, None)
                except AttributeError:

                    print("Warning: batch_extract_dag_and_parallel not found, falling back to sequential processing")
                    planning_futures = []
                    for planning_info in all_planning_tasks:
                        try:
                            future = mcp_box.extract_dag_and_parallel.remote(planning_info['plan_str'])
                            planning_futures.append((planning_info['trajectory_idx'], future, planning_info))
                        except Exception as e:
                            logger.error(f"Error preparing DAG extraction for trajectory {planning_info['trajectory_idx']}: {e}")
                            future = mcp_box.sequential_planning.remote(planning_info['max_turns'])
                            planning_futures.append((planning_info['trajectory_idx'], future, planning_info))

                    planning_completed = await asyncio.gather(
                        *[asyncio.wait_for(future, timeout=10.0) for _, future, _ in planning_futures],
                        return_exceptions=True
                    )

                    for (traj_idx, _, planning_info), result in zip(planning_futures, planning_completed):
                        if isinstance(result, (Exception, asyncio.TimeoutError)):
                            logger.error(f"Planning failed for trajectory {traj_idx}: {result}")
                            try:
                                fallback_future = mcp_box.sequential_planning.remote(planning_info['max_turns'])
                                fallback_result = await asyncio.wait_for(fallback_future, timeout=5.0)
                                planning_results[traj_idx] = fallback_result
                            except Exception as fallback_e:
                                logger.error(f"Fallback planning also failed for trajectory {traj_idx}: {fallback_e}")
                                planning_results[traj_idx] = (None, None, None)
                        else:
                            planning_results[traj_idx] = result
                except Exception as e:
                    logger.error(f"Batch planning failed: {e}")

                    for planning_info in all_planning_tasks:
                        planning_results[planning_info['trajectory_idx']] = (None, None, None)

            print(f"\n=== Stage 2: collect MCP registrations from plans ===")
            registration_by_trajectory = defaultdict(list)

            for traj_idx, result in trajectory_results.items():

                if traj_idx not in optimal_trajectory_set:
                    continue

                if traj_idx not in planning_results:
                    print(f"Warning: Trajectory {traj_idx} has no plan; skipping MCP registration")
                    continue

                dag_list, task_id, subtask_dict = planning_results[traj_idx]

                if task_id is None:
                    print(f"Warning: Trajectory {traj_idx} planning failed (task_id=None); skipping MCP registration")
                    continue

                print(f"Processing trajectory {traj_idx}: task_id={task_id}")

                responses = self.loop_responses_token[traj_idx]

                for turn_idx in range(1, len(responses), 2):
                    if turn_idx >= len(responses):
                        break

                    tool_call_str = self.tokenizer.decode(responses[turn_idx], skip_special_tokens=True)

                    registration_by_trajectory[traj_idx].append({
                        'tool_call_str': tool_call_str,
                        'task_id': task_id,
                        'subtask_dict': subtask_dict,
                        'turn_idx': turn_idx
                    })

                print(f"Trajectory {traj_idx}: collected {len(registration_by_trajectory[traj_idx])} tool calls")

            if registration_by_trajectory:
                print(f"\nProcessing {sum(len(v) for v in registration_by_trajectory.values())} MCP registrations...")
                registration_batch_start = time()

                try:
                    registration_futures = []
                    for traj_idx, tasks in registration_by_trajectory.items():

                        tool_call_strs = [t['tool_call_str'] for t in tasks]
                        task_id = tasks[0]['task_id']
                        subtask_dict = tasks[0]['subtask_dict']

                        try:

                            future = mcp_box.batch_register_mcps.remote(tool_call_strs, task_id, subtask_dict)
                            registration_futures.append((traj_idx, len(tasks), future))
                            print(f"Submitting registration batch for trajectory {traj_idx}, {len(tasks)} tools")
                        except AttributeError:

                            print(f"WARNING: batch_register_mcps unavailable; registering tools individually.")
                            for task in tasks:
                                future = mcp_box.register_mcp.remote(
                                    task['tool_call_str'], task['task_id'], task['subtask_dict']
                                )
                                registration_futures.append((traj_idx, 1, future))

                    if registration_futures:

                        registration_results = await asyncio.gather(
                            *[asyncio.wait_for(future, timeout=60.0) for _, _, future in registration_futures],
                            return_exceptions=True
                        )

                        success_count = 0
                        error_count = 0
                        for (traj_idx, count, _), result in zip(registration_futures, registration_results):
                            if isinstance(result, (Exception, asyncio.TimeoutError)):
                                error_count += count
                                logger.error(f"MCP registration failed for trajectory {traj_idx}: {result}")
                            else:
                                success_count += count

                        registration_batch_duration = time() - registration_batch_start
                        avg_time = registration_batch_duration / max(1, success_count + error_count)
                        print(f"Batch MCP registration time: {registration_batch_duration:.3f}s | passed: {success_count}, failed: {error_count} | mean: {avg_time:.3f}s/item")

                except Exception as e:
                    logger.error(f"Batch MCP registration failed: {e}")
            else:
                print(f"WARNING: No MCP registrations were collected.")

            return trajectory_results, planning_results

        try:
            trajectory_results, planning_results = asyncio.run(_batch_process_async())
        except RuntimeError:

            async def run_in_existing_loop():
                return await _batch_process_async()

            loop = asyncio.new_event_loop()
            try:
                trajectory_results, planning_results = loop.run_until_complete(run_in_existing_loop())
            finally:
                loop.close()

        result_process_start = time()
        for traj_idx, result in trajectory_results.items():
            input_ids_list.append(result['input_ids_list'])
            loss_mask_list.append(result['loss_mask'])
            length_list.append(result['total_length'])

        result_process_duration = time() - result_process_start
        print(f"Result assembly time: {result_process_duration:.3f}s")
        print("End to compose final output")
        # max_len = max(max(length_list), self.config_response_length)
        reduce_start = time()
        max_response_length = torch.tensor([max(length_list)], device=torch.cuda.current_device())
        # because only tp=0 will exec postprocess_output and compose_final_output
        # so we exec all_reduce in specified group(dp group)
        torch.distributed.all_reduce(max_response_length, op=torch.distributed.ReduceOp.MAX, group=group)
        max_len = int(max_response_length)
        print(f"Response-length computation and all_reduce time: {time() - reduce_start:.4f}s")
        print("End to reduce max_response_length")
        # right pad
        padding_start = time()
        input_ids = []
        loss_mask = []
        for idx, input_ids in enumerate(input_ids_list):
            input_ids = input_ids + [self.pad_token_id] * (max_len - len(input_ids))
            loss_mask = loss_mask_list[idx] + [0] * (max_len - len(loss_mask_list[idx]))
            input_ids_list[idx] = input_ids
            loss_mask_list[idx] = loss_mask[0:max_len]
        print(f"input_ids and loss_mask padding time: {time() - padding_start:.4f}s")

        tensor_creation_start = time()
        response_token = torch.tensor(input_ids_list, dtype=torch.int64)[:,:max_len]
        response_loss_mask = torch.tensor(loss_mask_list, dtype=torch.float32)
        response_attention_mask = (response_token != self.pad_token_id).long()
        print(f"Response tensor creation time: {time() - tensor_creation_start:.4f}s")

        # get the max length of the process rewards
        tool_process_start = time()
        max_tool_use_len = self.max_turns
        for tool_use_item in self.tool_use:
            max_tool_use_len = max(max_tool_use_len, len(tool_use_item))

        tool_use_tensor = []
        mcp_id_tensor = []
        candidate_mcp_dict = ray.get(mcp_box.get_mcps.remote())

        print(f"\n=== Tool ID diagnostics ===")
        print(f"Registered MCP tools: {len(candidate_mcp_dict)}")
        if candidate_mcp_dict:
            print(f"Registered MCP examples (first 5): {list(candidate_mcp_dict.keys())[:5]}")

        # Pad tool_use to have consistent dimensions
        for idx in range(len(self.tool_use)):

            if not self.tool_use[idx]:
                padded_tool_use = [torch.nan] * max_tool_use_len
            else:
                padded_tool_use = self.tool_use[idx] + [torch.nan] * (max_tool_use_len - len(self.tool_use[idx]))

            if not self.mcp_ids[idx]:
                padded_tool_id = [[np.nan]] * max_tool_use_len
            else:
                tool_id_list = []
                for turn_idx, turn_tools in enumerate(self.mcp_ids[idx]):

                    if isinstance(turn_tools, list):

                        tool_ids = self.find_step_id_by_mcp_name(turn_tools, candidate_mcp_dict)
                        tool_id_list.append(tool_ids)

                    elif isinstance(turn_tools, str):

                        tool_ids = self.find_step_id_by_mcp_name([turn_tools], candidate_mcp_dict)
                        tool_id_list.append(tool_ids)

                    else:

                        print(f"Warning: Trajectory {idx}, Turn {turn_idx}: Unexpected tool type {type(turn_tools)}, value: {turn_tools}")
                        tool_id_list.append([np.nan])

                padded_tool_id = tool_id_list + [[np.nan]] * (max_tool_use_len - len(tool_id_list))

            mcp_id_tensor.append(padded_tool_id)
            tool_use_tensor.append(padded_tool_use)

        total_tool_ids = sum(len(traj) for traj in mcp_id_tensor)
        nan_count = sum(
            sum(1 for tool_id_list in traj for tool_id in tool_id_list if isinstance(tool_id, float) and np.isnan(tool_id))
            for traj in mcp_id_tensor
        )
        print(f"Tool ID statistics: total={total_tool_ids}, NaN count={nan_count}, NaN ratio={nan_count/max(1, total_tool_ids)*100:.1f}%")
        print(f"tool_use and mcp_ids processing time: {time() - tool_process_start:.4f}s")

        final_batch_creation_start = time()
        tool_use_score = torch.tensor(tool_use_tensor)
        step_rewards_tensor = torch.nan_to_num(tool_use_score, nan=0.0)

        anchor_obs = auto_pad_matrix(mcp_id_tensor)

        traj_index = np.arange(len(mcp_id_tensor))
        input_ids = torch.cat([self.init_prompt_token, response_token], dim=-1)
        attention_mask = torch.cat([self.init_attention_mask, response_attention_mask], dim=-1)
        # position_ids = torch.cumsum(attention_mask, dim=1, dtype=torch.long) - 1
        position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None) * attention_mask
        loss_mask = torch.cat([torch.zeros_like(self.init_attention_mask, dtype=torch.float32), response_loss_mask], dim=-1)
        final_batch = TensorDict(
            {
                'prompts': self.init_prompt_token,
                'responses': response_token,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'loss_mask': loss_mask,
                'tool_use_scores': tool_use_score,
                'step_rewards': step_rewards_tensor,
            },
            batch_size=self.batch_size,
        )
        if self.tool_call_count is None:
            self.tool_call_count = np.array([0] * self.batch_size, dtype=object)
        final_output = DataProto(
            batch=final_batch,
            non_tensor_batch={
                "mcp_name": anchor_obs,
                "anchor_obs": anchor_obs,
                "traj_index": traj_index,
                "tool_call_count": self.tool_call_count
            },
        )
        if getattr(self.env_object.tool_manager.verl_config, 'runtime_config_path', None):
            if list(trajectory_results) != list(range(self.batch_size)) or len(self.env_object.plan_list) != self.batch_size:
                raise ValueError('Runtime episode identities do not align with generated tensor rows')
            final_output.non_tensor_batch['runtime_episode_id'] = np.array(
                [row['_runtime_context'] for row in self.env_object.plan_list], dtype=object)
        print(f"Final batch/output construction time: {time() - final_batch_creation_start:.4f}s")
        print(f"compose_final_output total time: {time() - start_time:.4f}s")
        return final_output

if __name__ == "__main__":
    'Test ToolUtilsGraph output processing and MCP step lookup with minimal local dependencies.'
    import types
    from types import SimpleNamespace

    class DummyEnv:
        def __init__(self):
            self.use_process_reward = False
            self.mcp_enable = False

            import ray
            from envs.tool_manager.centralized.central import CentralizedToolActor

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, include_dashboard=False)

            class MockConfig:
                def __init__(self):
                    self.tool_timeout = 30
                    self.mcp_mode = "stdio"
                    self.config_path = "prompt/workflow.yaml"
                    self.enable_limiter = False
                    self.max_turns = 6

                def get(self, k, default=None):
                    return getattr(self, k, default)

            self.centralized_actor_handle = CentralizedToolActor.options(name="test_centralized_tool_actor").remote(
                MockConfig()
            )

        def step(self, responses, tokenizer, loop_cnt, questions=None):
            n = len(responses)
            return ["info"] * n, [False] * n, None, None, ["plan"] * n

        def get_step_reward(self, responses):
            return [0.5] * len(responses)

    def build_dummy_output(batch_size=2, prompt_len=4, resp_len=3):
        prompts = torch.ones(batch_size, prompt_len, dtype=torch.int64)
        responses = torch.ones(batch_size, resp_len, dtype=torch.int64)
        attention_mask = torch.ones(batch_size, prompt_len + resp_len, dtype=torch.int64)
        position_ids = torch.arange(prompt_len + resp_len).unsqueeze(0).expand(batch_size, -1)
        batch = {
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        return DataProto.from_dict(batch)

    def main():

        try:
            from transformers import AutoTokenizer

            tok_path = os.environ.get("TEST_TOKENIZER_PATH", "Qwen/Qwen3-4B")
            tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
            if tokenizer.pad_token_id is None:

                tokenizer.pad_token_id = tokenizer.eos_token_id
            meta_info = {"pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id}
        except Exception as e:
            print(f"Load real tokenizer failed, fallback Dummy. Error: {e}")

            class DummyTokenizer:
                def __init__(self):
                    self.pad_token_id = 0
                    self.eos_token_id = 1

                def encode(self, s, add_special_tokens=False):
                    return [2, 3]  # mock stop_id

                def batch_decode(self, seqs, skip_special_tokens=False):
                    return ["resp" for _ in seqs]

                def decode(self, seq, skip_special_tokens=True):
                    return "resp"

                def __call__(self, texts, truncation=True, max_length=16):
                    if isinstance(texts, str):
                        texts = [texts]
                    return SimpleNamespace(input_ids=[[9, 9] for _ in texts])

            tokenizer = DummyTokenizer()
            meta_info = {"pad_token_id": 0, "eos_token_id": 1}
        config = SimpleNamespace(
            stop=["<stop>"],
            prompt_length=8,
            response_length=3,
            tool_response_length=4,
            max_turns=2,
            n=2,
        )
        env = DummyEnv()
        tu = ToolUtilsGraph(tokenizer, meta_info, config, env_object=env, is_validate=False)

        output = build_dummy_output()
        next_data = tu.postprocess_output(output, step=0, questions=["q1", "q2"])
        print("postprocess_output next_data is None:", next_data is None)

        tu.tool_use = [[0.1], [0.2]]
        tu.mcp_ids = [["plan"], ["plan"]]
        tu.rollout_num = 2

        for idx in range(len(tu.loop_responses_token)):
            tu.loop_responses_token[idx].append([5, 6])

        import torch.distributed as dist
        orig_all_reduce = getattr(dist, "all_reduce", None)
        dist.all_reduce = lambda tensor, op=None, group=None: tensor
        orig_current_device = torch.cuda.current_device
        torch.cuda.current_device = lambda: 0

        try:
            final = tu.compose_final_output(step=1, group=None)
            print("compose_final_output keys:", final.batch.keys(), final.non_tensor_batch.keys())
            print("step_rewards:", final.non_tensor_batch["step_rewards"])
            print("anchor_obs:", final.non_tensor_batch["anchor_obs"])
        finally:

            dist.all_reduce = orig_all_reduce
            torch.cuda.current_device = orig_current_device

        # find_step_id_by_mcp_name
        sid = tu.find_step_id_by_mcp_name("plan")
        print("find_step_id_by_mcp_name('plan') ->", sid)

    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com

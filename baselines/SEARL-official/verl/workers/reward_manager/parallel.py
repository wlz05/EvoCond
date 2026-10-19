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

import torch
import asyncio
import numpy as np
from verl import DataProto
from envs.base import Env
from verl.workers.reward_manager import register

@register("parallel")
class AsyncRewardManager:
    'The reward manager with async processing.\n    '
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.env_object = None
        self.reward_rollout_wg = None
        self.reward_tokenizer = None
        self.reward_fn_key = reward_fn_key
        self.if_val = False
        self.use_process_reward = False
        self.stop_token = None

    def set_env_object(self, env_object):
        self.env_object: Env = env_object

    def set_reward_rollout_wg(self, reward_rollout_wg):
        self.reward_rollout_wg = reward_rollout_wg

    def set_reward_tokenizer(self, reward_tokenizer):
        self.reward_tokenizer = reward_tokenizer

    def verify(self, data):
        device = data[0].batch['prompts']

        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        if self.env_object.use_verify_tool:
            data = self._get_verified_results(data)

        step_mask = torch.zeros_like(data.batch['responses'], dtype=torch.long)
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            response_attention_mask = data_item.batch['attention_mask'][prompt_length:]

            last_one_idx = torch.where(response_attention_mask == 1)[0][-1]

            step_mask[i, last_one_idx] = 1

        data.batch['step_mask'] = step_mask

        tool_call_count_list = data.non_tensor_batch.get('tool_call_count', None)
        assert tool_call_count_list is not None, "tool_call_count_list is None"
        scores = self.env_object.compute_score(
            self.reward_rollout_wg,
            self.reward_tokenizer,
            self.tokenizer,
            data,
            if_val=self.if_val,
            tool_call_count=tool_call_count_list
        )

        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=device)

        return scores

    def find_tool_end_positions(self, data: DataProto, tokenizer):
        'Find positions of all stop_tokens in responses, marking the end of tool calls.\n        \n        Args:\n            data (DataProto): The data containing responses\n            tokenizer: The tokenizer to decode token IDs\n            \n        Returns:\n            list: A list of dictionaries, each containing positions of stop_token for one sample\n        '
        tool_end_positions = []

        # Get the token ID for stop_token
        im_end_token_id = tokenizer.encode(self.stop_token, add_special_tokens=False)
        if isinstance(im_end_token_id, list) and len(im_end_token_id) > 0:
            im_end_token_id = im_end_token_id[0]

        for i in range(len(data)):
            data_item = data[i]

            # Get the entire response tokens
            response_ids = data_item.batch['responses']

            # Find all occurrences of im_end_token_id in response_ids
            positions = []

            # Convert to numpy for easier processing if it's a tensor
            if isinstance(response_ids, torch.Tensor):
                response_ids_np = response_ids.cpu().numpy()
            else:
                response_ids_np = response_ids

            # Find all positions where the token ID matches im_end_token_id
            for idx, token_id in enumerate(response_ids_np):
                if token_id == im_end_token_id:
                    positions.append(idx)

            tool_end_positions.append({
                'response_length': len(response_ids_np),
                'im_end_positions': positions,  # Positions in the response_ids array
                'im_end_count': len(positions)
            })

        return tool_end_positions

    def get_step_mask(self, data: DataProto):

        step_mask = torch.zeros_like(data.batch['responses'], dtype=torch.long)

        tool_end_positions = self.find_tool_end_positions(data,self.tokenizer)

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            response_attention_mask = data_item.batch['attention_mask'][prompt_length:]
            last_one_idx = torch.where(response_attention_mask == 1)[0][-1].item()

            step_index=[]

            #

            if tool_end_positions[i]['im_end_count']%2 == 1:

                for j in range((tool_end_positions[i]['im_end_count'] // 2) + 1):
                    step_index.append(tool_end_positions[i]['im_end_positions'][2 * j])
            else:

                for j in range((tool_end_positions[i]['im_end_count'] // 2)):
                    step_index.append(tool_end_positions[i]['im_end_positions'][2 * j])

                if last_one_idx not in step_index:
                    step_index.append(last_one_idx)

            step_index = sorted(set(step_index))

            if self.use_process_reward and not self.if_val:
                for idx in step_index:
                    step_mask[i, idx] = 1
            else:
                step_mask[i, last_one_idx] = 1

        return step_mask, tool_end_positions

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        if self.env_object.use_verify_tool:
            data = self._get_verified_results(data)

        #if use process reward
        self.use_process_reward = self.env_object.use_process_reward

        step_mask, tool_end_positions = self.get_step_mask(data)

        data.batch['step_mask'] = step_mask

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        mcp_worthiness_list = data.non_tensor_batch.get('mcp_worthiness', None)
        tool_call_count_list = data.non_tensor_batch.get('tool_call_count', None)
        # assert tool_call_count_list is not None, "tool_call_count_list is None"
        scores = self.env_object.compute_score(self.reward_rollout_wg, self.reward_tokenizer, self.tokenizer, data, if_val=self.if_val, use_process_reward=self.use_process_reward, mcp_worthiness=mcp_worthiness_list, tool_call_count=tool_call_count_list)

        step_mask = self._adjust_step_mask_to_scores(step_mask, scores, data, tool_end_positions)
        data.batch['step_mask'] = step_mask

        reward_tensor = self._set_reward_tensor(scores, data)

        acc_list = []
        for s in scores:
            if isinstance(s, (list, tuple, np.ndarray)):

                acc_list.append(s[-1] if len(s) > 0 else 0.0)
            else:

                acc_list.append(s)

        acc_arr = np.array(acc_list, dtype=np.float32)

        try:
            data.batch["acc"] = torch.tensor(acc_arr, dtype=torch.float32, device=reward_tensor.device)
        except Exception:
            data.batch["acc"] = torch.tensor(acc_arr, dtype=torch.float32)

        if return_dict:
            extra = {"acc": acc_arr.tolist()}
            breakdown = data.non_tensor_batch.get('reward_breakdown')
            if breakdown is not None:
                if len(breakdown) != len(data):
                    raise ValueError('Final-answer reward breakdown is misaligned')
                extra['answer_correctness'] = [row.get('answer_correctness') if isinstance(row, dict) else None
                                               for row in breakdown]
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": extra,
            }
        else:
            return reward_tensor

    def _adjust_step_mask_to_scores(self, step_mask, scores, data: DataProto, tool_end_positions):
        '\n        \u667a\u80fd\u8c03\u6574 step_mask \u4ee5\u5339\u914d scores\uff0c\u786e\u4fdd reward \u5bf9\u9f50\u7684\u6b63\u786e\u6027\u3002\n\n        \u6838\u5fc3\u6539\u8fdb\uff1a\n        1. \u79fb\u9664\u4f4d\u7f6e\u65f6\uff1a\u4f18\u5148\u4fdd\u7559\u524d\u9762\u91cd\u8981\u7684\u6b65\u9aa4\u4f4d\u7f6e\n        2. \u6dfb\u52a0\u4f4d\u7f6e\u65f6\uff1a\u667a\u80fd\u9009\u62e9\u771f\u5b9e\u7684 im_end \u4f4d\u7f6e\uff0c\u4fdd\u6301\u987a\u5e8f\n        3. \u8bb0\u5f55\u5bf9\u9f50\u8d28\u91cf\uff0c\u8b66\u544a\u53ef\u80fd\u7684 reward \u5931\u8c03\n\n        Args:\n            step_mask: \u539f\u59cb\u7684 step_mask\n            scores: \u8ba1\u7b97\u5f97\u5230\u7684\u5206\u6570\u5217\u8868\uff08\u6bcf\u4e2a\u6837\u672c\u4e00\u4e2a\u5217\u8868\uff09\n            data: DataProto \u5bf9\u8c61\n            tool_end_positions: \u5df2\u8ba1\u7b97\u7684 tool_end_positions \u5217\u8868\n\n        Returns:\n            \u8c03\u6574\u540e\u7684 step_mask\n        '
        adjusted_mask = step_mask.clone()
        adjustment_made = False
        alignment_quality = {'perfect': 0, 'good_remove': 0, 'good_add': 0, 'poor_add': 0}

        for i in range(len(data)):
            cur_scores = scores[i]
            mask_indices = torch.where(adjusted_mask[i] == 1)[0]
            mask_count = len(mask_indices)
            scores_count = len(cur_scores)

            if mask_count == scores_count:

                alignment_quality['perfect'] += 1
                continue

            adjustment_made = True
            print(f"\n{'='*80}")
            print(f"[\u667a\u80fd\u8c03\u6574] \u6837\u672c {i}")
            print(f"  \u539f\u59cb step_mask \u4f4d\u7f6e\u6570: {mask_count}")
            print(f"  scores \u6570\u91cf: {scores_count}")
            print(f"  \u5dee\u5f02: {scores_count - mask_count}")

            if mask_count < scores_count:

                data_item = data[i]
                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                response_attention_mask = data_item.batch['attention_mask'][prompt_length:]
                valid_positions = torch.where(response_attention_mask == 1)[0]

                needed = scores_count - mask_count
                print(f"  \u9700\u8981\u6dfb\u52a0 {needed} \u4e2a\u4f4d\u7f6e")
                print(f"  \u5f53\u524d\u6807\u8bb0\u7684\u4f4d\u7f6e: {mask_indices.tolist()}")

                tool_end_info = tool_end_positions[i]
                all_im_end_positions = sorted(tool_end_info['im_end_positions'])

                unused_im_end = [pos for pos in all_im_end_positions
                                if adjusted_mask[i, pos] == 0]

                added_positions = []
                if len(unused_im_end) >= needed:

                    current_marked = mask_indices.tolist()

                    for pos in unused_im_end[:needed]:
                        adjusted_mask[i, pos] = 1
                        added_positions.append(pos)
                        print(f"  \u2713 \u6dfb\u52a0 im_end \u4f4d\u7f6e: {pos}")

                    alignment_quality['good_add'] += 1
                    print(f"  \u2713 \u5bf9\u9f50\u8d28\u91cf: GOOD\uff08\u4f7f\u7528\u771f\u5b9e im_end \u4f4d\u7f6e\uff09")

                else:

                    for pos in unused_im_end:
                        adjusted_mask[i, pos] = 1
                        added_positions.append(pos)
                        print(f"  \u2713 \u6dfb\u52a0 im_end \u4f4d\u7f6e: {pos}")
                        needed -= 1

                    if needed > 0:
                        last_idx = valid_positions[-1].item()
                        if adjusted_mask[i, last_idx] == 0:
                            adjusted_mask[i, last_idx] = 1
                            added_positions.append(last_idx)
                            needed -= 1
                            print(f"  \u25b3 \u6dfb\u52a0\u6700\u540e\u4f4d\u7f6e: {last_idx}")

                    if needed > 0:
                        for pos in reversed(valid_positions.tolist()):
                            if adjusted_mask[i, pos] == 0:
                                adjusted_mask[i, pos] = 1
                                added_positions.append(pos)
                                needed -= 1
                                print(f"  \u2717 [\u964d\u7ea7] \u6dfb\u52a0\u666e\u901a\u4f4d\u7f6e: {pos}")
                                if needed == 0:
                                    break

                    alignment_quality['poor_add'] += 1
                    print(f"  \u26a0\ufe0f \u5bf9\u9f50\u8d28\u91cf: POOR\uff08\u7f3a\u5c11 im_end \u4f4d\u7f6e\uff0creward \u53ef\u80fd\u5931\u8c03\uff09")
                    import warnings
                    warnings.warn(
                        f"\u6837\u672c {i}: \u65e0\u6cd5\u627e\u5230\u8db3\u591f\u7684 im_end \u4f4d\u7f6e\u6765\u5bf9\u9f50 {scores_count} \u4e2a\u5206\u6570\u3002"
                        f"\u8fd9\u4f1a\u5bfc\u81f4\u90e8\u5206 step reward \u5bf9\u9f50\u4e0d\u51c6\u786e\uff01"
                    )

                print(f"  \u6dfb\u52a0\u7684\u4f4d\u7f6e: {sorted(added_positions)}")
                print(f"  \u8c03\u6574\u540e\u4f4d\u7f6e: {torch.where(adjusted_mask[i] == 1)[0].tolist()}")

            else:

                excess = mask_count - scores_count
                print(f"  \u9700\u8981\u79fb\u9664 {excess} \u4e2a\u4f4d\u7f6e")
                print(f"  \u5f53\u524d\u6807\u8bb0\u7684\u4f4d\u7f6e: {mask_indices.tolist()}")

                kept_indices = mask_indices[:scores_count]
                removed_indices = mask_indices[scores_count:]

                adjusted_mask[i] = 0
                for idx in kept_indices:
                    adjusted_mask[i, idx] = 1

                alignment_quality['good_remove'] += 1
                print(f"  \u2713 \u4fdd\u7559\u524d {scores_count} \u4e2a\u4f4d\u7f6e: {kept_indices.tolist()}")
                print(f"  \u2713 \u79fb\u9664\u540e {excess} \u4e2a\u4f4d\u7f6e: {removed_indices.tolist()}")
                print(f"  \u2713 \u5bf9\u9f50\u8d28\u91cf: GOOD\uff08\u4fdd\u7559\u91cd\u8981\u6b65\u9aa4\uff09")

            print(f"{'='*80}\n")

        if adjustment_made:
            total = len(data)
            poor_rate = alignment_quality['poor_add'] / total if total > 0 else 0

            print(f"\n{'='*80}")
            print(f"[\u5bf9\u9f50\u8d28\u91cf\u7edf\u8ba1]")
            print(f"  \u5b8c\u7f8e\u5339\u914d: {alignment_quality['perfect']}/{total} ({alignment_quality['perfect']/total*100:.1f}%)")
            print(f"  \u826f\u597d\u6dfb\u52a0: {alignment_quality['good_add']}/{total}")
            print(f"  \u826f\u597d\u79fb\u9664: {alignment_quality['good_remove']}/{total}")
            print(f"  \u5dee\u52b2\u6dfb\u52a0: {alignment_quality['poor_add']}/{total}")

            if poor_rate > 0.05:  # 5%
                print(f"\n\u26a0\ufe0f\u26a0\ufe0f\u26a0\ufe0f \u4e25\u91cd\u8b66\u544a\uff1a{poor_rate*100:.1f}% \u7684\u6837\u672c\u5bf9\u9f50\u8d28\u91cf\u5dee\uff01")
                print(f"\u8fd9\u4f1a\u5bfc\u81f4 Process Reward \u5931\u8c03\uff0c\u5f71\u54cd\u8bad\u7ec3\u6548\u679c\uff01")
                print(f"\u5efa\u8bae\uff1a")
                print(f"  1. \u68c0\u67e5 tool_use \u548c response \u751f\u6210\u662f\u5426\u540c\u6b65")
                print(f"  2. \u51cf\u5c11 rollback \u89e6\u53d1\u9891\u7387\uff08\u5e94\u7528\u914d\u7f6e\u4f18\u5316\uff09")
                print(f"  3. \u8003\u8651\u7981\u7528\u8fc7\u7a0b\u5956\u52b1\uff0c\u53ea\u7528\u6700\u7ec8\u5956\u52b1")
            elif poor_rate > 0:
                print(f"\n\u26a0\ufe0f \u8b66\u544a\uff1a{poor_rate*100:.1f}% \u7684\u6837\u672c\u5bf9\u9f50\u8d28\u91cf\u5dee\uff0c\u8bf7\u5173\u6ce8")

            print(f"{'='*80}\n")

        return adjusted_mask

    def _set_reward_tensor(self, scores, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        step_mask = data.batch['step_mask']
        for i in range(len(data)):
            cur_step_mask, cur_scores = step_mask[i], scores[i]
            mask_indices = torch.where(cur_step_mask == 1)[0]
            # torch.save(data, "outputs/debug/data.pth")
            assert cur_step_mask.sum() == len(cur_scores), f"cur_step_mask.sum(): {cur_step_mask.sum()}, len(cur_scores): {len(cur_scores)}"

            for j, idx in enumerate(mask_indices):
                reward_tensor[i, idx] = cur_scores[j]
            '\n            if self.use_process_reward:\n                for j, idx in enumerate(mask_indices):\n                    reward_tensor[i, idx] = cur_scores[j]\n            else:\n                 for j, idx in enumerate(mask_indices):\n                    if j == len(mask_indices)-1:\n                        reward_tensor[i, idx] = cur_scores[-1]\n                        #reward_tensor[i,idx]=cur_scores.sum()\n            '

        return reward_tensor

    def _get_verified_results(self, data: DataProto):
        async def _get_single_result(data_source, solution_str, ground_truth, extra_info):

            return await asyncio.to_thread(
                self.env_object.verify_tool,
                data_source,
                solution_str,
                ground_truth,
                extra_info
            )

        async def _process_all():
            tasks = []
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
                data_source = data_item.non_tensor_batch['data_source']
                extra_info = data_item.non_tensor_batch.get('extra_info', None)

                task = _get_single_result(data_source, response_str, ground_truth, extra_info)
                tasks.append(task)

            return await asyncio.gather(*tasks)

        results = asyncio.run(_process_all())

        for i in range(len(data)):
            data[i].non_tensor_batch['reward_model']['ground_truth']['verified_results'] = results[i]

        return data

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com

# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Accumulate existing native batches until a fixed actor-memory barrier.'
import numpy as np
FULL = {'input_ids', 'attention_mask', 'position_ids', 'loss_mask'}
RESPONSE = {'responses', 'response_mask', 'old_log_probs', 'ref_log_prob', 'rollout_log_probs', 'values',
            'advantages', 'returns', 'token_level_scores', 'token_level_rewards'}

def barrier(phase, epoch_batches, count):
    if type(phase) is not int or phase < 0 or type(count) is not int or count < 1:
        raise ValueError('Invalid actor window position')
    if epoch_batches is not None and (type(epoch_batches) is not int or epoch_batches < 1):
        raise ValueError('Invalid epoch batch count')
    position = phase % epoch_batches if epoch_batches else phase
    return position == 0 or position % count == 0

def width(phase, epoch_batches, count):
    barrier(phase, epoch_batches, count)
    position = (max(1, phase)-1) % epoch_batches if epoch_batches else max(1, phase)-1
    return min(count, epoch_batches-position//count*count) if epoch_batches else count

def layout(questions, batch, count, *, tail='error', epochs=1):
    if any(type(value) is not int or value < 1 for value in (questions, batch, count, epochs)) or tail not in ('error', 'keep'):
        raise ValueError('Invalid training window layout')
    full, remainder = divmod(questions, batch)
    if tail == 'error' and (remainder or full < count or full % count):
        raise ValueError('Training split must contain complete frozen actor-memory windows with no dropped questions')
    batches = full+bool(remainder)
    sizes = [min(batch, questions-index*batch) for index in range(batches)]
    windows = [sum(sizes[index:index+count]) for index in range(0, batches, count)]
    return dict(batches_per_epoch=batches, frozen_batches=count,
        actor_updates_per_epoch=len(windows), total_actor_updates=len(windows)*epochs,
        dropped_training_questions=0, tail_questions=remainder,
        questions_per_epoch=questions, window_questions=windows, tail_policy=tail)
def tensor_pad(value, shape, fill, slices):
    result = value.new_full(shape, fill)
    result[slices] = value
    return result
def join(batches, concatenate, pad_token):
    if not batches or type(pad_token) is not int:
        raise ValueError('Native windows require a real tokenizer pad id')
    keys = set(batches[0].batch.keys())
    auxiliary = set(batches[0].non_tensor_batch)
    if any(set(row.batch.keys()) != keys or set(row.non_tensor_batch) != auxiliary for row in batches):
        raise ValueError('Native window tensor or reward fields changed')
    prompt = max(row.batch['prompts'].shape[-1] for row in batches)
    response = max(row.batch['responses'].shape[-1] for row in batches)
    for key in sorted(keys, key=lambda value: (value not in FULL, value)):
        shapes = [row.batch[key].shape for row in batches]
        if key in FULL:
            if any(len(shape) != 2 for shape in shapes):
                raise ValueError('Contextual windows currently require text-only sequence tensors')
            for row in batches:
                value = row.batch[key]
                old_prompt = row.batch['prompts'].shape[-1]
                old_response = row.batch['responses'].shape[-1]
                if value.shape[-1] != old_prompt+old_response:
                    raise ValueError('Native full-sequence and response tensors are misaligned')
                result = value.new_full((value.shape[0], prompt+response), pad_token if key == 'input_ids' else 0)
                result[:, prompt-old_prompt:prompt] = value[:, :old_prompt]
                result[:, prompt:prompt+old_response] = value[:, old_prompt:]
                row.batch[key] = result
        elif key in RESPONSE or key in ('tool_use_scores', 'step_rewards'):
            if any(len(shape) != 2 for shape in shapes):
                raise ValueError('Native response or step tensor has an unsupported shape')
            width = response if key in RESPONSE else max(shape[-1] for shape in shapes)
            fill = pad_token if key == 'responses' else float('nan') if key == 'tool_use_scores' else 0
            for row in batches:
                value = row.batch[key]
                row.batch[key] = tensor_pad(value, (value.shape[0], width), fill, (slice(None), slice(0, value.shape[-1])))
        elif key != 'prompts' and len({tuple(shape[1:]) for shape in shapes}) != 1:
            raise ValueError('Unknown ragged native tensor cannot be padded safely')
    for row in batches:
        value = row.batch['prompts']
        row.batch['prompts'] = tensor_pad(value, (value.shape[0], prompt), pad_token, (slice(None), slice(prompt-value.shape[-1], prompt)))
    for key in auxiliary:
        arrays = [row.non_tensor_batch[key] for row in batches]
        if len({value.shape[1:] for value in arrays}) == 1:
            continue
        if key not in ('anchor_obs', 'mcp_name') or len({value.ndim for value in arrays}) != 1:
            raise ValueError('Unknown ragged native metadata cannot be padded safely')
        maximum = tuple(max(value.shape[i] for value in arrays) for i in range(1, arrays[0].ndim))
        for row, value in zip(batches, arrays):
            result = np.full((value.shape[0], *maximum), np.nan, dtype=value.dtype)
            result[(slice(None), *[slice(0, width) for width in value.shape[1:]])] = value
            row.non_tensor_batch[key] = result
    result = concatenate(batches)
    result.meta_info = dict(result.meta_info)
    result.meta_info['global_token_num'] = result.batch['attention_mask'].sum(-1).tolist()
    result.meta_info['frozen_native_batches'] = len(batches)
    return result
def accumulate(pending, batch, ready, count, concatenate):
    if type(count) is not int or count < 1 or type(ready) is not bool:
        raise ValueError('Invalid frozen-window schedule')
    if len(pending) >= count:
        raise ValueError('Native batch window exceeded its fixed budget')
    pending.append(batch)
    if not ready:
        if len(pending) == count:
            raise ValueError('Coordinator withheld a complete actor update window')
        return None
    if len(pending) != count:
        raise ValueError('Coordinator requested an incomplete actor update window')
    result = concatenate(pending) if count > 1 else pending[0]
    pending.clear()
    return result

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com

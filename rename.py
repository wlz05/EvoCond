# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Mechanical module migration with explicit receipts and preserved import bindings.'
from __future__ import annotations
import argparse
import ast
from functools import lru_cache
from io import StringIO
from hashlib import sha256
import json
import keyword
import os
from pathlib import Path
import re
import tokenize
from zipfile import ZipFile, ZIP_DEFLATED
ALIASES = dict(item.split(':') for item in (
    '_state_dict_utils:dictionary activation_offload:offload agent_loop:agent aggregate_logger:aggregate android_control:control android_env:environment android_utils:android api_generator:generate',
    'accommodation_server:accommodation append_plan_to_prompt:append async_generator_test:asynchrony async_mcp_manager:asynchronous async_rollout_utils:sampling async_server:server async_sglang_server:serve attraction_server:attraction',
    'base_config:base base_env:environment base_env_config:configuration base_generator:base base_manager:manager base_model_merger:base base_server:base base_tool:base batch_evaluation:batch cache_base:base cachebox_cache:cache',
    'centralized_qwen3_manager:central chat_scheduler:scheduler check_custom_rwd_fn:custom check_license:license check_results:results checkpoint_manager:checkpoint commonsense_constraint:commonsense concurrency_limiter:limiter',
    'config_converter:convert config_manager:configuration constants_ppo:constants context_utils:context converter_hf_to_mcore:convert core_algos:algorithms create_dataset:dataset create_model_tokenizer:tokenizer custom_eval:evaluate',
    'dapo_ray_trainer:adaptive data_process:process data_processing:processing dense_common:dense disk_persist:disk dist_checkpointing:distributed distance_server:distance dp_actor:actor dp_critic:critic dynamicgen_dataset:dynamic EED:distance',
    'empty_annotations:annotations engine_impl:engine env_config:configuration env_manager:manager extended_zss:extension flight_server:flight flops_counter:counter flow_editor:editor fsdp_checkpoint_manager:checkpoint fsdp_model_merger:merge',
    'fsdp_sft_trainer:supervised fsdp_sglang:language fsdp_ulysses:sequence fsdp_utils:sharding fsdp_vllm:inference fsdp_workers:workers full_hh_rlhf:feedback geo3k:geometry geo3k_tool:geometry get_env_codes:codes get_prompt:prompt',
    'gpt_judge:judge graph_rag_retrieval:retrieval gsm8k:arithmetic gsm8k_interaction:arithmetic gsm8k_multiturn_w_tool:arithmetic gsm8k_tool:arithmetic hard_constraint:hard hard_constraints:constraints hdfs_io:filesystem hf_rollout:rollout',
    'import_utils:imports interaction_registry:registry isolated_dispatch:dispatch kimi_vl:vision latex_pre_process:notation linear_cross_entropy:entropy llama_loader:loader llama_loader_depracated:legacy llama_saver:saver llama3_manager:language',
    'local_search:local logging_utils:logging main_dapo:adapt main_eval:eval main_evaluate:evaluate main_generation:generate main_ppo:train main_ppo_split:partition main_prime:prime main_sppo:symmetric main_trainer:trainer',
    'math_batch:batch math_dapo:reasoning math_dataset:mathematics math_hard_verify:verify math_normalize:normalize math_verify:mathematics mcp_base:base mcp_base_tool:base mcp_creation:creation mcp_creation_back:legacy mcp_manager:manager mcp_search_tool:search McpClientManager:client',
    'megatron_actor:actor megatron_checkpoint_manager:parallel megatron_critic:critic megatron_model_merger:parallel megatron_sglang:parallel megatron_utils:parallel megatron_vllm:parallel megatron_workers:parallel memory_buffer:memory metric_utils:metrics mm_base_manager:multimodal mm_tool_utils:multimodal',
    'model_forward:forward model_forward_fused:fused model_initializer:initialize model_merger:merger modeling_llama_megatron:language modeling_qwen2_megatron:reasoning monkey_patch:patch mstx_profile:profile multiturn_sft_dataset:conversation',
    'naive_chat_scheduler:simple naive_rollout:naive net_utils:network npu_patch:hardware nq_search:search nvtx_profile:tracing overall_evaluation:overall parallel_attention:attention parallel_decoder:decoder parallel_linear:linear parallel_mlp:perceptron parallel_rmsnorm:normalization parallel_state:state',
    'parse_utils:parse patch_v012:compatibility persist_base:base pipeline_parallel:pipeline plot_logs:plot pool_api:pool prime_core_algos:prime prime_dp_rm:reward prime_fsdp_workers:workers prime_ray_trainer:trainer process_vl:process project_management:projects py_functional:functional',
    'qwen_math:mathematics qwen2:model qwen2_5_manager:reasoning qwen2_5_vl:multimodal qwen2_5_vl_manager:visual qwen2_loader:reasoning qwen2_loader_depracated:deprecated qwen2_saver:reasoning qwen2_vl:visual qwen3_manager:reasoner qwen3_manager_baseline:baseline',
    'r1_dataset:reasoning r1_reward:reasoning ray_backend:backend ray_trainer:trainer ray_utils:distributed redis_cache_manager:cache redis_port:port restaurant_server:restaurant retrieval_server:retrieval reward_definition:rewards reward_evaluator:evaluator reward_model:reward reward_rollout_example:example reward_score:score reward_test:rewards reward_utils:rewards',
    'rl_dataset:reinforcement rm_dataset:reward rollout_trace:trace rope_utils:rotary run_fsdp_vllm:run sandbox_fusion_tools:sandbox search_r1_like_qa_em:answer search_r1_like_utils:search search_tool:search searl_baseline:baseline searl_reward:searl searl_toolcall:toolcall',
    'seqlen_balancing:balancing sequence_parallel:sequence sft_dataset:supervised sglang_rollout:language single_turn_agent_loop:single sole_planning:planning split_logs:split split_monkey_patch:split sppo_ray_trainer:trainer sppo_worker:worker state_dict:state storage_manager:storage storage_test:storage stress_test_mcp:stress stress_test_tools:pressure suppress_mcp_warnings:warnings',
    'teacher_collection:collection teacher_continuation:continuation tensor_parallel:tensor testing_util:testing tool_agent_loop:tool tool_definition:tools tool_parser:parser tool_registry:registry tool_utils:tools tool_utils_graph:graph tool_utils_rollback:rollback torch_dtypes:types torch_functional:functional training_deployment:deployment travelplanner_preprocess:travel utils_sglang:language verified_creation:verified',
    'vision_config:configuration vision_model:model vision_transformer_block:transformer vision_utils:vision vllm_async_server:asynchronous vllm_rollout_spmd:rollout vllm_utils:inference weight_converter:weights weight_loader_registry:registry worker_group:group',
    'test_async_tool_manager:asynchronous test_auto_padding:padding test_batch_eval:batch test_batch_fix:repair test_chat_scheduler_async:asynchrony test_chat_scheduler_non_tensor_passthrough:passthrough test_check_worker_alive:alive test_colocated_workers:colocated test_colocated_workers_fused:fusion test_data_transfer:transfer test_decorator:decorator test_driverfunc_to_worker:driver',
    'test_env_process_reward_fallback:fallback test_flops_counter:counter test_fs:filesystem test_fsdp_ckpt:checkpoint test_fused_workers:fused test_hf_rollout:rollout test_high_level_scheduling_api:scheduling test_import:importing test_import_utils:imports test_isolated_dispatch:dispatch test_linear_cross_entropy:entropy test_llama3_manager:language test_local_search_and_python:python test_local_search_bounds:bounds',
    'test_mcp_cache_real:cache test_mcp_isolated_entrypoints:entrypoints test_memory_buffers:memory test_metric_utils:metrics test_model:model test_module:module test_multiturn_sft_dataset:conversation test_nested_format:format test_ops:operations test_parallel:parallel test_plan_call_prompt:planning test_protocol:protocol test_qwen3_manager:reasoning test_ray_local_envs:environments',
    'test_rl_dataset:reinforcement test_rm_dataset:reward test_rvdz:rendezvous test_sandbox:sandbox test_sandbox_fusion:isolation test_seqlen_balancing:balancing test_sft_dataset:supervised test_sglang_async_rollout_w_tools:tools test_sglang_async_spmd:asynchronous test_sglang_spmd:language test_sp_loss_match:loss test_tensor_dict:tensor test_tensor_dict_utilities:dictionary test_timeout_decorator:timeout',
    'test_tool_use:usage test_torch_functional:functional test_transformer:transformer test_transformers_ulysses:sequence test_update_mcp_box:update test_verified_creation:verified test_vllm_hf_loader:loader test_vllm_multi_turn:multiturn test_vllm_spmd:inference test_vllm_tool_calling:calling test_worker_group_basics:basics test_worker_group_torch:collective',
) for item in item.split())
WORDS = 'adapter application artifact assembly assessment audit balancing batch binding buffer cache candidate catalogue cell chain channel checkpoint class client cluster collection comparison composition compression concurrency configuration connection constraint context contract controller conversion coordination creation dataset decoder descriptor destination detection diagnostic dictionary dispatcher distribution document encoder endpoint engine entry environment estimation evaluation event execution experiment feature feedback filter flow format function fusion geometry gradient graph group guard history identity inference input inspection integration interface kernel label layer learning lifecycle loader logging mapping mask matrix mechanism memory message metric migration mixture model module network node normalization observer occupancy operation optimization option output padding parameter parser partition passage payload pipeline policy pool prediction primitive procedure process profile projection prompt protocol query ranking reader receipt record reference registry relation repair report resource response retrieval reward role rollback rollout routing runtime sample sandbox scheduler schema scope score search selection semantic sequence serialization server session signal snapshot source stage state statistic storage strategy stream structure subtask summary support symbol target teacher tensor terminal test threshold token tool trace trainer trajectory transformation transition transport type update validation value vector version view vocabulary window worker wrapper'.split()
SPECIAL = {'__init__.py', '__main__.py'}
IGNORED = {'.git', '__pycache__', '.test-tmp', 'runs', 'assets'}
TEXT = {'.py', '.sh', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.md'}
def files(root):
    result = []
    for directory, children, names in os.walk(root, followlinks=False):
        children[:] = [name for name in children if name not in IGNORED and not name.startswith(('.evocond-backup', 'tmp'))]
        result.extend(Path(directory)/name for name in names if (Path(directory)/name).is_file())
    return sorted(result)
def valid(name):
    return bool(re.fullmatch('[a-z]+', name))
def plan(root):
    paths = [path for path in files(root) if path.suffix == '.py']
    occupied = {parent: {child.name.casefold() for child in parent.iterdir()} for parent in {path.parent for path in paths}}
    result = {}
    for path in paths:
        if path.name in SPECIAL or valid(path.stem):
            continue
        preferred = ALIASES.get(path.stem.strip())
        if preferred is None:
            raise ValueError('Missing explicit English name for '+path.relative_to(root).as_posix())
        choices = [preferred, preferred+'s', *WORDS]
        name = next((word for word in choices if valid(word) and not keyword.iskeyword(word) and (word+'.py').casefold() not in occupied[path.parent]), None)
        if name is None:
            raise ValueError('No unambiguous English module name remains')
        target = path.with_name(name+'.py')
        if not target.resolve().is_relative_to(root.resolve()) or path.is_symlink() or target.exists():
            raise ValueError('Unsafe or occupied module migration target')
        occupied[path.parent].add(target.name.casefold())
        result[path] = target
    return result
@lru_cache(maxsize=16)
def anchors(root):
    return [root, *[path for path in (root/'baselines').iterdir() if path.is_dir() and path.name != 'CostBench-history']] if (root/'baselines').is_dir() else [root]
def modules(root, mapping):
    result = {}
    for anchor in anchors(root):
        for old, new in mapping.items():
            if old.is_relative_to(anchor):
                before, after = ('.'.join(path.relative_to(anchor).with_suffix('').parts) for path in (old, new))
                if all(part.isidentifier() for part in before.split('.')):
                    if before in result and result[before] != after:
                        raise ValueError('Conflicting shared upstream module aliases: '+before)
                    result[before] = after
    return result
@lru_cache(maxsize=4)
def pattern(keys):
    trie = {}
    for key in keys:
        node = trie
        for char in key:
            node = node.setdefault(char, {})
        node[''] = None
    def build(node):
        children = [re.escape(char)+build(child) for char, child in sorted(node.items()) if char]
        branch = '(?:'+'|'.join(children)+')' if len(children) > 1 else children[0] if children else ''
        return '(?:'+branch+')?' if '' in node and branch else branch
    return re.compile('(?<![A-Za-z0-9_])'+build(trie)+'(?![A-Za-z0-9_])')
def replace(text, replacements):
    if not replacements:
        return text
    return pattern(tuple(replacements)).sub(lambda match: replacements[match.group()], text)
def substitutions(root, mapping, modulemap):
    replacements, stemtargets = dict(modulemap), {}
    for old, new in mapping.items():
        for anchor in anchors(root):
            if old.is_relative_to(anchor):
                left, right = (item.relative_to(anchor).as_posix() for item in (old, new))
                replacements[left] = right
                replacements[left.replace('/', '\\')] = right.replace('/', '\\')
        stemtargets.setdefault(old.name, set()).add(new.name)
    replacements.update({key: next(iter(values)) for key, values in stemtargets.items() if len(values) == 1})
    return replacements
def rewrite(path, text, root, mapping, modulemap, replacements=None):
    if path in (root/'rename.py', root/'tests/names.py', root/'configs/names.json', root/'configs/imports.json'):
        return text
    if path.suffix == '.py':
        candidates = {}
        for old, new in mapping.items():
            candidates.setdefault(old.stem, set()).add(new.stem)
        bare = {key: next(iter(values)) for key, values in candidates.items() if len(values) == 1}
        tree = ast.parse(text, filename=str(path))
        lines = text.splitlines(keepends=True)
        offsets, total = [], 0
        for line in lines:
            offsets.append(total)
            total += len(line)
        def point(line, column):
            return offsets[line-1]+len(lines[line-1].encode('utf-8')[:column].decode('utf-8'))
        edits, bindings = [], {}
        anchor = max((item for item in anchors(root) if path.is_relative_to(item)), key=lambda item: len(item.parts))
        package = list(path.relative_to(anchor).parent.parts)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = modulemap.get(alias.name, bare.get(alias.name, alias.name))
                    bound = alias.asname or alias.name.split('.')[0]
                    bindings[bound] = alias.name if alias.asname else alias.name.split('.')[0]
                    if name != alias.name:
                        suffix = ' as '+alias.asname if alias.asname else ' as '+alias.name if '.' not in alias.name else ''
                        edits.append((point(alias.lineno, alias.col_offset), point(alias.end_lineno, alias.end_col_offset), name+suffix))
            elif isinstance(node, ast.ImportFrom):
                prefix = '.'.join(package[:len(package)-node.level+1]) if node.level else ''
                absolute = '.'.join(value for value in (prefix, node.module) if value)
                name = modulemap.get(absolute, bare.get(absolute, absolute) if not node.level else absolute)
                if name != absolute:
                    changed = name[len(prefix)+1:] if prefix else name
                    start, end = point(node.lineno, node.col_offset), point(node.end_lineno, node.end_col_offset)
                    segment = text[start:end]
                    match = re.match(r'from\s+(\.*)([\w.]+)\s+import', segment)
                    if match is None:
                        raise ValueError('Unsupported multiline module import')
                    edits.append((start+match.start(2), start+match.end(2), changed))
                for alias in node.names:
                    qualified = absolute+'.'+alias.name if absolute else alias.name
                    bindings[alias.asname or alias.name] = qualified
                    if qualified in modulemap:
                        changed = modulemap[qualified].rsplit('.', 1)[-1]
                        edits.append((point(alias.lineno, alias.col_offset), point(alias.end_lineno, alias.end_col_offset), changed+' as '+(alias.asname or alias.name)))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            def dotted(value):
                if isinstance(value, ast.Name):
                    return value.id
                if isinstance(value, ast.Attribute):
                    head = dotted(value.value)
                    return head+'.'+value.attr if head else None
                return None
            value = dotted(node)
            if value is None:
                continue
            head, _, tail = value.partition('.')
            qualified = bindings.get(head, head)+('.'+tail if tail else '')
            if qualified in modulemap and tail:
                target = modulemap[qualified]
                replacement = head+'.'+target[len(bindings.get(head, head))+1:]
                edits.append((point(node.lineno, node.col_offset), point(node.end_lineno, node.end_col_offset), replacement))
        for start, end, value in sorted(set(edits), reverse=True):
            text = text[:start]+value+text[end:]
    replacements = substitutions(root, mapping, modulemap) if replacements is None else replacements
    if path.suffix != '.py':
        return replace(text, replacements)
    lines = text.splitlines(keepends=True)
    offsets, total = [], 0
    for line in lines:
        offsets.append(total)
        total += len(line)
    edits = []
    for token in tokenize.generate_tokens(StringIO(text).readline):
        if token.type not in (tokenize.STRING, tokenize.COMMENT):
            continue
        changed = replace(token.string, replacements)
        if changed != token.string:
            edits.append((offsets[token.start[0]-1]+token.start[1], offsets[token.end[0]-1]+token.end[1], changed))
    for start, end, value in reversed(edits):
        text = text[:start]+value+text[end:]
    return text
def migrate(root, *, run=False, receipt=None, backup=None, relink=None):
    root = root.resolve(strict=True)
    mapping = plan(root)
    references = dict(mapping)
    if relink is not None:
        prior = json.loads(relink.read_text(encoding='utf-8'))
        if prior.get('format') != 'evocond_module_migration_v1':
            raise ValueError('Unrecognized prior module migration receipt')
        for old, new in prior['paths'].items():
            before, after = root/old, root/new
            if not before.resolve().is_relative_to(root) or not after.resolve().is_relative_to(root) or not after.is_file():
                raise ValueError('Prior module mapping escapes or differs from current sources')
            references[before] = after
    modulemap = modules(root, references)
    replacements = substitutions(root, references, modulemap)
    edits, errors = {}, []
    for path in files(root):
        if path.suffix not in TEXT and not (path.suffix == '.json' and 'configs' in path.parts):
            continue
        try:
            with tokenize.open(path) if path.suffix == '.py' else path.open(encoding='utf-8') as stream:
                text = stream.read()
                encoding = stream.encoding
            changed = rewrite(path, text, root, references, modulemap, replacements)
            if path.suffix == '.py':
                ast.parse(changed, filename=str(path))
            if changed != text:
                edits[path] = (text, changed, encoding)
        except (SyntaxError, UnicodeError) as error:
            errors.append(dict(path=path.relative_to(root).as_posix(), reason=str(error)))
    if errors:
        raise ValueError('Pre-existing or transformed syntax errors: '+json.dumps(errors))
    result = dict(format='evocond_module_migration_v1', applied=run, renamed=len(mapping), rewritten=len(edits),
        python_special_files=sorted(SPECIAL), gpu_started=False,
        paths={old.relative_to(root).as_posix(): new.relative_to(root).as_posix() for old, new in mapping.items()},
        original_sha256={path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest() for path in set(mapping)|set(edits)})
    if not run:
        return result
    if receipt is None or receipt.exists() or not receipt.resolve().is_relative_to(root):
        raise ValueError('Declare a new local migration receipt before changing sources')
    if backup is None or backup.exists() or not backup.resolve().is_relative_to(root):
        raise ValueError('Declare a new local source backup before changing sources')
    backup.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(backup, 'x', ZIP_DEFLATED) as archive:
        for path in sorted(set(mapping)|set(edits)):
            archive.writestr(path.relative_to(root).as_posix(), path.read_bytes())
        archive.writestr('manifest.json', json.dumps(result, indent=2, sort_keys=True))
    with ZipFile(backup) as archive:
        if archive.testzip() is not None:
            raise ValueError('Source backup verification failed')
        for path in set(mapping)|set(edits):
            if archive.read(path.relative_to(root).as_posix()) != path.read_bytes():
                raise ValueError('Source backup content mismatch')
    result['backup'] = backup.resolve().relative_to(root).as_posix()
    result['backup_sha256'] = sha256(backup.read_bytes()).hexdigest()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    for path, (_, changed, encoding) in edits.items():
        path.write_text(changed, encoding=encoding, newline='')
    for old, new in mapping.items():
        old.rename(new)
    result['changed_sha256'] = {new.relative_to(root).as_posix(): sha256(new.read_bytes()).hexdigest() for new in mapping.values()}
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    return result
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--receipt', type=Path, default=Path('configs/names.json'))
    parser.add_argument('--backup', type=Path, default=Path('configs/original.zip'))
    parser.add_argument('--relink', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    result = migrate(args.root, run=args.run, receipt=args.receipt, backup=args.backup, relink=args.relink)
    print(json.dumps({key: value for key, value in result.items() if key not in ('paths', 'original_sha256', 'changed_sha256')}, sort_keys=True))
if __name__ == '__main__':
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com

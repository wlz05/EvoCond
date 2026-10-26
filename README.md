# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

**EvoCond** learns how accumulated tool experience should guide a co-evolving language agent. Compatible tools share a capability identity, while their relation events retain the conditions and feedback of each transition. At each retrieval, EvoCond reconstructs supported ordering proposals and learns their value to the current actor from randomized tool-list presentation and terminal task utility. Evidence accumulates across tasks; adoption value is re-estimated as the actor and inventory evolve.

We study **structural memory fatigue**: later training stages accumulate more relation evidence while the utility gained from structural guidance declines relative to semantic retrieval on the same execution basis. An improving actor can make memory-assisted tasks more successful while needing less help from memory. EvoCond separates historical relation support from current adoption value, retaining reusable evidence while adapting its influence through current execution feedback.

EvoCond achieves state-of-the-art pass@1 performance in mathematical and multi-hop reasoning. Experiments use **8 NVIDIA GeForce RTX 4090 GPUs**, with random seeds **42, 52, 3047, and 3407**.

## Architecture

```text
Current task and visible execution state
           |
           v
Shared capabilities and a common semantic candidate pool
           |
           v
Compatible relation events -> current signed ordering evidence
           |
           v
Ordering evidence -> semantic, structural, and fused tool lists
           |
           v
Current-actor adoption policy -> actual tool-list presentation
           |
           v
Qwen3-4B planning and execution -> terminal utility and full cost
           |
           +----> relation evidence and fixed-actor adoption feedback
           +----> stage-boundary actor update and adoption re-evaluation
```

Capability compatibility, relation support, and adoption value answer different questions. Compatibility determines which implementations can share an identity. Conditional support generates an ordering proposal. Randomized presentation estimates whether that proposal improves the current actor's terminal utility relative to the semantic list. The teacher selects the presented list; the actor retains planning, tool-call, and argument decisions. Hard execution dependencies remain enforced for every list.

Within a frozen execution window, a preselected retrieval point receives one randomized list in an existing rollout. The completed episode supplies its utility; logged action probabilities support doubly robust value estimation. An accepted teacher update is used in the next window on the same actor and public inventory. At the co-evolution boundary, memory is published and the actor is updated, starting a new basis for adoption evaluation. Test evaluation freezes public memory and learning state.

### Key Contributions

- **Evidence-preserving relation reuse.** Shared capability identities retain condition-feedback events, from which each query reconstructs directed ordering proposals.
- **Current-actor adoption value.** Actual tool-list presentation is the intervention, and terminal utility relative to semantic retrieval is the learning target.
- **Adoption under co-evolution.** Frozen windows support value estimation and deployment on the same execution basis; actor and inventory updates trigger re-evaluation.

## Project Structure

```text
EvoCond/
|-- control/       # Conditional memory, adoption, and execution controls
|-- flow/          # Dynamic tool execution and trajectory collection
|-- model/         # Policy and training support
|-- state/         # Tool memory and persistent execution state
|-- baselines/     # SEARL and EVOLT source adaptations
|-- configs/       # Experiment, isolation, and evaluation settings
|-- tests/         # CPU regression tests
|-- assets.py      # Download and verify model and dataset sources
|-- convert.py     # Convert declared source schemas into training splits
|-- experiment.py  # Freeze, train, resume, and evaluate an experiment
|-- run.py         # Launch the four eight-GPU seed configurations
|-- study.py       # Matched main and ablation conditions
|-- summary.py     # Aggregate complete four-seed test reports
|-- fatigue.py     # Inspect memory value and actor-by-memory controls
|-- test.py        # Run regression tests
```

## Requirements

- Linux with NVIDIA drivers and a CUDA-compatible PyTorch installation.
- **8 RTX 4090 GPUs** for the distributed actor experiment.
- Python 3.10 or newer, Ray, vLLM, Transformers, PyArrow, and the dependencies of the bundled SEARL runtime.
- An isolated tool-execution service, a retrieval service, and configured judge endpoints.

The actor is [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B). Tool retrieval uses [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2). The training and evaluation judge configurations support [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) and [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B), respectively. Judge services run separately from the eight-GPU actor allocation.

## Installation

```bash
git clone https://github.com/wlz05/EvoCond.git
cd EvoCond
python install.py --profile native
python install.py --profile native --requirements assets/deployment/requirements.txt --run
python -B test.py
```

Place the exact dependency lock for the selected CUDA environment at `assets/deployment/requirements.txt`. The first installation command checks dependency provenance; the second installs that lock. Keep the repository root on `PYTHONPATH` when invoking the bundled baseline directly. The experiment launcher sets this automatically for its workers.

## Datasets

Training uses the mixed mathematical and reasoning tasks distributed as [Tool-star / Multi-Tool-RL-10K](https://huggingface.co/datasets/dongguanting/Multi-Tool-RL-10K). The data interface also accepts [ARPO-RL-Reasoning-10K](https://huggingface.co/datasets/dongguanting/ARPO-RL-Reasoning-10K).

| Benchmark | Task | Source |
|---|---|---|
| GSM8K | Grade-school mathematical reasoning | [GitHub](https://github.com/openai/grade-school-math) |
| MATH500 | Competition mathematical reasoning | [Dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) |
| AIME24 | Invitational mathematics | [SEARL paper](https://aclanthology.org/2026.acl-long.1125/) |
| HotpotQA | Multi-hop question answering | [GitHub](https://github.com/hotpotqa/hotpot) |
| 2WikiMultihopQA | Multi-hop question answering | [GitHub](https://github.com/Alab-NII/2wikimultihop) |
| MuSiQue | Compositional multi-hop reasoning | [GitHub](https://github.com/StonyBrookNLP/musique) |
| Bamboogle | Compositional question answering | [GitHub](https://github.com/ofirpress/self-ask) |

The primary metric is **pass@1 accuracy**, following [SEARL](https://aclanthology.org/2026.acl-long.1125/). Evaluation records also retain tool calls, execution cost, tokens, and elapsed time. Tool cost and model inference cost are reported separately.

## Data Preparation

```bash
python assets.py --root assets --assets datasets evaluation retriever judge --mirror --run
python convert.py --inspect-source assets/data/toolstar_rl10k/grpo_mix_train.parquet --input-format parquet
python convert.py --manifest assets/deployment/datasets.json
python convert.py --manifest assets/deployment/datasets.json --output assets/splits --run
```

Use a conversion manifest that specifies the source files, checksums, field mapping, and train/validation/test assignments. `python convert.py --help` and `python partition.py --help` expose the conversion and grouped-split interfaces. Keep related task variants in the same partition. Downloaded upstream sources and converted experiment splits are separate artifacts.

Arrange local artifacts as follows:

```text
assets/
|-- models/Qwen3-4B/
|-- models/e5-base-v2/
|-- deployment/requirements.txt
|-- deployment/datasets.json
|-- deployment/isolation.json
|-- deployment/inspection.json
|-- splits/train.parquet
|-- splits/validation.parquet
|-- splits/test.parquet
```

## Reproducibility

`configs/contextual.json` specifies the eight-GPU experiment. `run.py` generates configurations for seeds **42, 52, 3047, 3407**, holding dataset partitions fixed. Every condition starts from the same initial Qwen3-4B checkpoint with an empty generated-tool inventory and the same built-in Python and search tools. Source, weights, data, deployment receipt, initialization stage, and execution settings are frozen before training.

The `initialization` section declares a pretrained or cold-start checkpoint. A cold-start comparison uses the same supplied SFT checkpoint in every condition. Model architecture and weight fingerprints are checked; a folder named `Qwen3-4B` cannot substitute for the 4B architecture.

## Main Experiments

| Condition | Retrieval and memory | Actor update cadence |
|---|---|---|
| `searl` | Pinned SEARL tool graph and tool-anchored credit | Every collected batch |
| `pace` | Same native graph and credit | Every two batches |
| `evocond` | Full conditional memory and adoption teacher | Every two batches |

`pace` isolates the actor update cadence; its graph keeps the SEARL update path. All conditions share actor initialization, data order per seed, tools, isolation, judge, and generation budgets. The previous contextual three-arm entry used `searl` as a semantic control. Current main experiments keep that control in the ablation suite and run the native tool graph in `searl`.

```bash
python run.py configs --suite main --run
python run.py prepare --suite main --run
python run.py train --suite main --run
python run.py status --suite main
```

These commands train each condition and seed sequentially on the same eight GPUs. `--conditions evocond --seeds 42` selects one job. Without `--run`, configuration and training commands print their plans. A shared GPU lock prevents a second condition from starting on the same allocation.

The collection batch is 256 base questions, with eight rollouts per question. Full two-batch windows contain 512 questions. The final batch and final short window are retained. For 10,000 training questions this gives 40 collection batches, a final batch of 16, and a final two-batch update window of 272 questions. These are schedule calculations, not benchmark measurements. The frozen protocol records actual window sizes and update calls. SEARL's paper Table 2 reports global batch 384 on H200; the released entry uses 256. The eight-4090 runs use the explicit 256 collection configuration above.

## Ablation Experiments

| Condition | Change from full EvoCond |
|---|---|
| `semantic` | Present semantic retrieval only |
| `global` | Erase context conditioning from relation evidence; retain the learned teacher |
| `identity` | Remove learned identity alignment; retain conditional reading and the teacher |
| `fixed` | Replace learned adoption with fixed fusion, alpha = 0.5 |
| `raw` | Replace the learned context projection with raw context distance |

```bash
python run.py configs --suite ablation --run
python run.py prepare --suite ablation --run
python run.py train --suite ablation --run
python ablation.py --suite statistics
```

Each training ablation restarts from the common actor checkpoint and its own empty inventory. The `statistics` suite contains full EvoCond, the combined raw conditional bandit control (`simple`), and the confidence-bound-only control (`bernstein`). Fixed-actor, fixed-memory mechanism evaluation is separate from these independently trained systems. The complete comparison protocol is in [experiments.md](experiments.md).

## Memory Fatigue

Each contextual training phase records the inventory used for retrieval, the pending publication inventory, and a randomized estimate of nonsemantic tool-list value relative to the semantic list. The estimate includes all preselected base questions and separates terminal success lift, clipped cost lift, and net utility. It uses the logged probabilities with a fixed reward midpoint, adding no reward head or adoption threshold.

```bash
python fatigue.py --root runs/reproduction/main/evocond/seed42/teacher/runtime \
  --output outputs/memoryvalue.json
python fatigue.py --panel assets/deployment/fatigue/seed42/panel.json \
  --output outputs/fatigue42.json
python fatigue.py --panels \
  assets/deployment/fatigue/seed42/panel.json \
  assets/deployment/fatigue/seed52/panel.json \
  assets/deployment/fatigue/seed3047/panel.json \
  assets/deployment/fatigue/seed3407/panel.json \
  --output outputs/fatiguesummary.json
```

The trace command summarizes completed training windows. The panel commands analyze exported frozen evaluations with old/new actors crossed with old/new relation banks, using common tasks, capabilities, budgets, and a fixed structural presentation rule. They separate actor, relation-bank, and interaction effects; optional adopted outcomes measure the current teacher's recovery separately. Four-seed aggregation uses independent training runs as its uncertainty unit. Prepare the measured cell files and [panel manifest](configs/fatigue.example.json) as specified in [experiments.md](experiments.md#memory-fatigue). The existing DR teacher and deployment rule govern training; diagnostic outputs do not change the policy.

## Evaluation

```bash
python run.py evaluate --suite main --conditions evocond --seeds 42 \
  --resume runs/reproduction/main/evocond/seed42/teacher/checkpoints/global_step_40 \
  --split validation --run
```

Choose the checkpoint on validation, then use `--split test --allow-test` with the same checkpoint. `global_step_40` illustrates the 10,000-question schedule; use the actual validation-selected checkpoint shown by `status`. Batch evaluation accepts a frozen map through `--checkpoints`; see [the checkpoint-map example](configs/checkpoints.example.json). No latest-checkpoint guessing or test-score selection is performed.

The example map contains the three main conditions for seed 42. Use `--seeds 42` with that map, or extend it to the exact selected seed/condition grid. Judge and result manifest paths are relative to the manifest file, not the shell's working directory. Set the real file checksums before running either manifest.

Evaluation uses one greedy rollout per scheduled question, a frozen tool inventory, and task-private newly created tools. Failure, timeout, and missing prediction count as zero in the v2 judge manifest; duplicates and unaligned identities are rejected. The final Qwen3-32B scoring uses the author answer parser:

```bash
python judge.py --manifest assets/deployment/judge.json \
  --official-root baselines/SEARL-official --model assets/models/Qwen3-32B \
  --output outputs/judging --run
python summary.py --manifest assets/deployment/results.json --output outputs/summary.json
```

Use [the v2 judge manifest](configs/judge.example.json) for the actual study labels and [the four-seed result manifest](configs/results.example.json) for aggregation. Run scoring after the actor job or on a separate allocation. `summary.py` reports per-benchmark mean and sample standard deviation over the four training seeds, plus macro and question-weighted summaries. A complete test denominator is required in every condition.

For interrupted training, select one condition and seed and pass its paired checkpoint through `run.py train --resume`. Checkpoints bind actor weights, optimizer and data position, tool graph, registry, and adoption state. Existing frozen runs retain their original configuration; create the new `runs/reproduction` study for this protocol.

## Acknowledgments

The dynamic tool environment and distributed training runtime build on [SEARL](https://github.com/circles-post/SEARL) and [verl](https://github.com/volcengine/verl). Third-party source retains its original copyright notices and licenses.

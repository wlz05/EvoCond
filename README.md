# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

**EvoCond** learns the current influence of accumulated tool relations. Shared capabilities retain condition-feedback events; each query reconstructs a directed combination proposal, and randomized presentation estimates its terminal benefit relative to the current actor's semantic execution. Adoption is re-evaluated when the actor and public inventory change. Historical support determines the proposal; current execution benefit determines its influence.

We study **structural memory fatigue**: relation evidence keeps growing while its incremental execution benefit declines. Actor learning changes that benefit from both sides: fewer remaining errors can be rescued, while more successful decisions can be disrupted. Historical support establishes a combination proposal; current execution determines its influence. EvoCond retains conditional evidence and learns adoption relative to the same actor's semantic execution.

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

- **Conditional tool-relation reuse.** Accumulated events support task-specific proposals whose influence is learned from current execution benefit.
- **An executable adoption decision.** Actual tool-list presentation is the intervention, and terminal utility relative to the current actor's semantic execution is the learning target.
- **A matched study of memory fatigue.** Condition-by-adoption controls test the proposal-to-benefit link; crossed actor-by-bank evaluation tests how that benefit changes during learning.

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
|-- source.py      # Restore pinned runtime resources
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

- Linux with NVIDIA drivers and a CUDA-compatible [PyTorch installation](https://pytorch.org/get-started/locally/).
- **8 RTX 4090 GPUs** for the distributed actor experiment.
- Python 3.10 or newer, [Ray](https://docs.ray.io/en/latest/), [vLLM](https://docs.vllm.ai/en/latest/), Transformers, [PyArrow](https://arrow.apache.org/docs/python/), and the dependencies of the bundled [SEARL runtime](https://github.com/circles-post/SEARL).
- An isolated tool-execution service, a Wikipedia retrieval service, and a separate final-scoring allocation.

The actor is [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B). Tool retrieval uses [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2). The released native entry disables reward-model rollouts and uses its answer, format, and tool-call reward functions during training and online validation. Final benchmark scoring loads [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) through [judge.py](judge.py), after training or on another allocation. The upstream QwQ-32B configuration is inactive in this recipe.

## Installation

```bash
git clone https://github.com/wlz05/EvoCond.git
cd EvoCond
python source.py --run
python install.py --profile native
python install.py --profile native --requirements assets/deployment/requirements.txt
python install.py --profile native --requirements assets/deployment/requirements.txt --run
python -B test.py
mkdir -p outputs
```

Run [source.py](source.py) with `--run` after cloning or pulling source updates to restore the pinned configurations, prompts, package map, and native launch scripts. Existing local settings are preserved. A recognized older source receipt is refreshed after the bundled source passes verification. The command downloads no models or datasets.

Run commands from the repository root in a Linux shell. Place the exact dependency lock for the selected CUDA environment at `assets/deployment/requirements.txt` before the lock-specific commands. The initial native audit reads upstream dependency declarations; the next command checks the supplied lock, and `--run` installs it. Keep the repository root on `PYTHONPATH` when invoking the bundled baseline directly. The experiment launcher sets this automatically for its workers.

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

The experiment proceeds from fixed source partitions to prepared Parquet inputs, seeded online training, validation-selected paired checkpoints, and final benchmark scoring.

```bash
python assets.py --root assets --assets datasets evaluation retriever judge --mirror --run
python convert.py --inspect-source assets/data/toolstar_rl10k/grpo_mix_train.parquet --input-format parquet
python convert.py --manifest assets/deployment/datasets.json
python convert.py --manifest assets/deployment/datasets.json --output assets/splits --run
```

### Partitions

The data recipe follows [SEARL, Sections 4.1 and 4.3](https://aclanthology.org/2026.acl-long.1125.pdf#page=6): Tool-star supplies the 10K RL training pool, and benchmark test membership follows [ARPO, Section 4.1](https://arxiv.org/html/2507.19849v1#S4.SS1). The [released SEARL entry](https://github.com/circles-post/SEARL/blob/main/main_searl.sh) assigns its processed files as follows:

| Role | Released source | Local experiment artifact |
|---|---|---|
| Training | `train_v4_plan_merged.parquet`, passed through `TRAIN_FILES` | `assets/splits/train.parquet` |
| Online validation | `qa_test_our_plan.parquet`, `test_math_v2_our_plan.parquet`, and `test_aime25_our_plan.parquet`, passed through `VAL_FILES` | `assets/splits/validation.parquet` |
| Final benchmark evaluation | The fixed ARPO benchmark test partitions used by SEARL | Frozen test exports named by the experiment and per-benchmark judge manifests |

Use the supplied partition memberships and record source versions, row counts, and checksums in `assets/deployment/datasets.json`. The released entry specifies validation files rather than a new split percentage. File roles follow `TRAIN_FILES` and `VAL_FILES`, including validation filenames containing `test`. An export into one file per role preserves the contributing rows and source identities. All conditions and seeds use the same prepared partitions.

### Preparation and Cache

[assets.py](assets.py) downloads versioned sources into `assets/data` and the requested retriever and final judge into `assets/models`. Supply the initial Qwen3-4B actor checkpoint separately; this download command does not create its cold-start SFT weights. The optional `wiki` download retrieves source archives; prepare the corresponding corpus and matching FAISS index for the retrieval service.

[convert.py](convert.py) prepares the declared inputs under `assets/splits`; `dataset_audit.json` records the source manifest, row counts, and output checksums. Keep that metadata with the Parquet files when moving the prepared directory. Conversion writes to a new output directory. If interrupted before the audit is published, preserve that partial directory and rerun into a fresh directory, then update the experiment's dataset paths to the completed export.

Use `mode: author_preprocessed` for one supplied author-preprocessed Parquet per role; the converter preserves its bytes. Set the experiment's `dataset_provenance` to the same mode. Use `mode: adapted_upstream` when mapping raw public data into the native schema, and record the field mapping in the manifest. Downloaded Tool-star sources and the author's processed training exports are separate preparation stages.

For a JSONL source with `question` and `answer` fields, an entry in `assets/deployment/datasets.json` has this form:

```json
{
  "format": "searl_dataset_manifest_v1",
  "mode": "adapted_upstream",
  "sources": [{
    "name": "custom_train",
    "path": "../raw/train.jsonl",
    "sha256": "<source-sha256-from-inspection>",
    "format": "jsonl",
    "split": "train",
    "source_split": "train",
    "source_uri": "<dataset-source-url>",
    "source_revision": "<immutable-release-or-commit>",
    "mapping": {"question": "/question", "answer": "/answer"},
    "data_source": "arpo",
    "ability": "reasoning"
  }]
}
```

Replace the source identity and pointers with inspected values, then add the validation and test entries. Paths use forward slashes relative to the manifest directory. The preparation command records the actual converted row counts; the experiment configuration names all three prepared roles.

The native Parquet loader uses the [SEARL/verl dataset path](baselines/SEARL-official/verl/utils/dataset/reinforcement.py): local files are read directly, remote files use the loader's local cache, and prompts are tokenized with the actor tokenizer. The loader cache defaults to `cache/verl/rlhf`; each experiment keeps its GraphRAG cache under its runtime directory. Training trajectories are generated online with the current actor and tool inventory. Reusing prepared input files does not reuse an earlier actor's rollouts. A paired checkpoint restores the data iterator along with the actor and memory state.

For additional task-grouped exports, [partition.py](partition.py) preserves declared benchmark membership by default and accepts explicitly requested grouped research splits. Group identifiers use NFKC normalization and trimmed text, so numeric `7` and text `"7"` identify the same group. All records are checked against the JSONL reader's limits before output begins. Each split streams into a temporary file with an incremental checksum and is published after the complete write; `split_manifest.json` follows the completed split files. After an interruption without that manifest, retain the partial directory and prepare a fresh export.

Arrange local artifacts as follows:

```text
assets/
|-- models/Qwen3-4B/
|-- models/e5-base-v2/
|-- models/Qwen3-32B/
|-- wiki/wiki-18.jsonl
|-- wiki/e5_Flat.index
|-- deployment/requirements.txt
|-- deployment/datasets.json
|-- deployment/isolation.json
|-- deployment/inspection.json
|-- splits/train.parquet
|-- splits/validation.parquet
|-- splits/test.parquet
```

### Execution Services

Set the following fields in `configs/contextual.json` before generating the study matrix:

| Field | Required artifact or service |
|---|---|
| `model` and `initialization` | Common initial Qwen3-4B weights and their pretrained or cold-start provenance |
| `encoder` | Local e5-base-v2 weights used by retrieval and conditional memory |
| `datasets` and `dataset_provenance` | The three completed Parquet roles and their matching preparation mode |
| `search_url` | The running Wikipedia service's `/retrieve` endpoint |
| `isolation_config` | Actual executor endpoint, deployment identity, and enforced execution limits |
| `isolation_receipt` | Matching, current deployment inspection receipt |

The [bundled retrieval server](baselines/SEARL-official/rag_server/retrieval.py) serves port 5003. Start it on its own allocation with the corpus and index built for the declared e5 encoder:

```bash
PYTHONPATH=. python baselines/SEARL-official/rag_server/retrieval.py \
  --index_path assets/wiki/e5_Flat.index \
  --corpus_path assets/wiki/wiki-18.jsonl \
  --retriever_model assets/models/e5-base-v2
```

This entry uses GPU FAISS and loads the encoder; provision its resources separately from the eight actor GPUs. Set `search_url` to an address reachable by all actor workers. A corpus and an index from different document orderings cannot be interchanged.

Provision the isolated executor before training. Use `configs/searl_isolation.example.json` as the configuration schema and record the real deployment in `assets/deployment/isolation.json`. The deployment operator supplies `assets/deployment/inspection.json` following [the receipt schema](control/isolation.py): its configuration hash, endpoint, deployment identity, inspector, validity interval, and checks describe the inspected running service. `prepare` validates this receipt; it does not start or inspect the service. The native launcher connects to the configured retrieval and execution services, while final scoring is a separate command.

An expired execution receipt, a broken retrieval service, or a malformed executor response stops collection. Invalid model-supplied tool arguments remain recoverable tool observations. A completed search with no matching documents is a valid empty result. Restore the inspected service before resuming from the last complete paired checkpoint; service outages do not supply tool-success evidence.

## Reproducibility

`configs/contextual.json` specifies the eight-GPU experiment. `run.py` generates configurations for seeds **42, 52, 3047, 3407**, holding dataset partitions fixed. Every condition starts from the same initial Qwen3-4B checkpoint with an empty generated-tool inventory and the same built-in Python and search tools. Source, weights, data, deployment receipt, initialization stage, and execution settings are frozen before training.

The `initialization` section declares the starting checkpoint. To follow the [released SEARL entry](https://github.com/circles-post/SEARL/blob/main/main_searl.sh), set `initialization.stage` to `coldstart`, record the supplied SFT checkpoint in `initialization.source`, and point `model` to that Qwen3-4B checkpoint in every condition. The starter configuration also supports a declared `pretrained` comparison. Model architecture and weight fingerprints identify the common starting actor.

## Main Experiments

| Condition | Retrieval and memory | Actor update cadence |
|---|---|---|
| `searl` | Pinned SEARL tool graph and tool-anchored credit | Every collected batch |
| `pace` | Same native graph and credit | Every two batches |
| `evocond` | Full conditional memory and adoption teacher | Every two batches |

`pace` isolates the actor update cadence; its graph keeps the SEARL update path. All conditions share actor initialization, data order per seed, tools, isolation, reward functions, and generation budgets. The semantic-only control belongs to the ablation suite; `searl` uses the native tool graph.

```bash
python run.py configs --suite main --run
python run.py prepare --suite main --run
python run.py train --suite main --run
python run.py status --suite main
```

These commands train each condition and seed sequentially on the same eight GPUs. `--conditions evocond --seeds 42` selects one job. Without `--run`, configuration and training commands print their plans. A shared GPU lock prevents a second condition from starting on the same allocation.

Repeat the same `configs` or `prepare` command after an interrupted preparation. Matching complete artifacts are reused and missing configurations are filled in; differing artifacts stop preparation. A missing runtime configuration inside an arm that already contains execution state requires recovery of that original configuration before launch.

### Training Sampling

The [native sampler](baselines/SEARL-official/verl/trainer/train.py) shuffles the fixed training rows without replacement using the job seed. One epoch visits the training partition once; each selected base question generates eight online rollouts. Conditions with the same seed receive the same question order. SEARL's tool-anchored credit and EvoCond's adoption learner operate on those collected trajectories.

The collection batch is 256 base questions, with eight rollouts per question. Full two-batch windows contain 512 questions. The final batch and final short window are retained. For 10,000 training questions this gives 40 collection batches, a final batch of 16, and a final two-batch update window of 272 questions. These are schedule calculations, not benchmark measurements. The frozen protocol records actual window sizes and update calls. SEARL's paper Table 2 reports global batch 384 on H200; the released entry uses 256. The eight-4090 runs use the explicit 256 collection configuration above.

## Ablation Experiments

| Condition | Change from full EvoCond |
|---|---|
| `semantic` | Present semantic retrieval only |
| `pooled` | Erase relation conditions and use fixed fusion, alpha = 0.5 |
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

Each training ablation restarts from the common actor checkpoint and its own empty inventory. The `statistics` suite contains full EvoCond, the combined raw conditional bandit control (`simple`), and the confidence-bound-only control (`bernstein`). Fixed-actor, fixed-memory mechanism evaluation is separate from these independently trained systems. The complete comparison protocol is in [the experimental protocol](#experimental-protocol).

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

The trace command summarizes completed training windows. The panel commands analyze exported frozen evaluations with old/new actors crossed with old/new relation banks, using common tasks, capabilities, budgets, and a fixed structural presentation rule. They separate actor, relation-bank, and interaction effects; optional adopted outcomes measure the current teacher's recovery separately. Four-seed aggregation uses independent training runs as its uncertainty unit. Prepare the measured cell files and `configs/fatigue.example.json` as specified in [the experimental protocol](#experimental-protocol). The existing DR teacher and deployment rule govern training; diagnostic outputs do not change the policy.

Each panel cell reports `paired_success`: observed rescues, harms, unchanged successes, and unchanged failures. Conditional rescue and harm rates retain their opportunity counts. The panel also reports `clipped_cost_lift`, making the net-utility change traceable to success changes and the exact cost scale used by the reward.

## Evaluation

### Validation and Checkpoint Selection

Training begins with validation, following the [released SEARL entry](https://github.com/circles-post/SEARL/blob/main/main_searl.sh). Its defaults validate every five collection steps and set `save_freq=0`. The eight-GPU recipe in this repository sets both `validation_freq` and `save_freq` to two collection batches, matching its actor-memory barriers. Validation runs through the native reward path; candidate checkpoints retain the paired actor and memory snapshot used for that validation.

```bash
python run.py evaluate --suite main --conditions evocond --seeds 42 \
  --resume runs/reproduction/main/evocond/seed42/teacher/checkpoints/global_step_40 \
  --split validation --run
```

For each condition and seed, choose the saved candidate with the highest declared validation score on the fixed validation partition. Native `val-core` metrics use `acc` when available and `reward` otherwise; keep the scoring route and aggregate selection metric the same across conditions. Record the selected paired checkpoint in `configs/checkpoints.example.json` or a study-specific map with the same schema. The runner uses that explicit selection. `global_step_40` is a checkpoint-path example for a 10,000-question schedule; replace it with the selected saved step reported by `status`.

The checkpoint map restored by [source.py](source.py) at `configs/checkpoints.example.json` contains the three main conditions for seed 42. Use `--seeds 42` with that map, or extend it to the exact selected seed/condition grid. Judge and result manifest paths are relative to the manifest file, not the shell's working directory. Set the real file checksums before running either manifest.

### Final Benchmark Test

Run the selected checkpoint map on the fixed test partitions:

```bash
python run.py evaluate --suite main --seeds 42 \
  --checkpoints configs/checkpoints.example.json --split test --allow-test --run
```

Evaluation uses one greedy rollout per scheduled question, a frozen tool inventory, and task-private newly created tools. Failure, timeout, and missing prediction count as zero in the v2 judge manifest; duplicates and unaligned identities are rejected. Final scoring follows [SEARL, Section 4.4](https://aclanthology.org/2026.acl-long.1125.pdf#page=6): Qwen3-32B judges binary correctness, the answer parser extracts the final answer, and the reported metric is pass@1 accuracy.

```bash
python judge.py --manifest assets/deployment/judge.json \
  --official-root baselines/SEARL-official --model assets/models/Qwen3-32B \
  --tensor-parallel 8 --output outputs/judging --run
python summary.py --manifest assets/deployment/results.json --output outputs/summary.json
```

Use `configs/judge.example.json` for the actual study labels and `configs/results.example.json` for aggregation. These examples are restored by [source.py](source.py). The scoring command distributes Qwen3-32B across eight GPUs; run it after releasing the actor allocation or on a separate allocation. [summary.py](summary.py) reports per-benchmark mean and sample standard deviation over the four training seeds, plus macro and question-weighted summaries. A complete test denominator is required in every condition.

For interrupted training, select one condition and seed and pass its paired checkpoint explicitly:

```bash
python run.py train --suite main --conditions evocond --seeds 42 \
  --resume runs/reproduction/main/evocond/seed42/teacher/checkpoints/global_step_20 --run
```

Replace the example step with a paired checkpoint in the `status` output's `checkpoints` list. `incomplete_checkpoints` lists rejected candidates and their reasons. Checkpoints bind actor weights, optimizer and data position, tool graph, registry, and adoption state. Training cannot resume an older step while later checkpoint paths remain in the same branch. Preserve and move those later artifacts outside that branch before replaying an interrupted window; evaluation can use an earlier validation-selected checkpoint without changing later checkpoints. Existing frozen runs retain their original configuration; a changed source or protocol uses a new study root. Judge, summary, and fatigue commands preserve existing output paths; choose a new output name for another report. The judge output's parent directory must exist.

## Experimental Protocol

### Method Questions

EvoCond's method chain is capability sharing, conditional proposal reconstruction, current-actor adoption, and renewed evaluation after co-evolution. The mechanism study follows the same order:

| Question | Matched comparison | Observation |
|---|---|---|
| Which historical distinctions change the current ordering? | Preserve, erase, or permute condition-feedback links with the same actor, tool versions, event counts, candidate pool, and budget | Actual lists, ordering choices, and paired terminal utility |
| Does a supported proposal help the current actor? | Randomize actual lists at a preselected point and continue under the same frozen execution policy | Incremental terminal utility relative to the semantic list, full cost, and subsequent calls |
| Does adoption value transfer after co-evolution? | Cross old and new actors with old and new relation banks, keeping capability identities and versions fixed; compare old and re-evaluated adoption on each current basis | Receiver and bank effects, retained rescues, observed harms, and cost |

Vary relation scale and condition heterogeneity separately. More edges or events alone do not identify the cause of a performance change. Historical relation support and randomized adoption utility have separate feedback records. These fixed-execution mechanism comparisons complement the independently trained main and ablation conditions below.

### Shared Setup

All main and training-ablation conditions use eight RTX 4090 GPUs, Qwen3-4B in non-thinking mode, and seeds 42, 52, 3047, 3407. Each seed uses the same data order across conditions. Data partitions do not change with training seeds. Checkpoint content, rather than its directory name, identifies the common starting actor. Declare `pretrained` or `coldstart` in `initialization`; a cold-start checkpoint is shared by all conditions.

Training uses a learning rate of 1e-6, one epoch, eight rollouts per base question, a 256-question collection batch, temperature 0.7, top-p 0.98, and repetition penalty 1.05. Prompt, response, and total context limits are 4096, 2048, and 32768 tokens; the execution limit is six turns. Tool-output tokens are excluded from policy loss. The native tool environment, permissions, retrieval endpoint, reward route, and timeout rules are common execution inputs.

SEARL's [paper, Appendix C](https://aclanthology.org/2026.acl-long.1125.pdf#page=13) uses Qwen3-4B and reports a 384-question global batch on H200. Its [released entry](https://github.com/circles-post/SEARL/blob/main/main_searl.sh) uses a 256-question batch. The local eight-4090 comparison uses the released 256 collection setting, not an unlabeled mixture of the two recipes.

### Main Comparison

| Label | Memory and retrieval | Actor update calls per full two-batch interval |
|---|---|---|
| `searl` | Native tool graph and tool-anchored credit | 2 |
| `pace` | Native graph and credit, actor cadence control | 1 |
| `evocond` | Conditional relations and learned adoption | 1 |

SEARL and EvoCond each maintain a private inventory generated by their own training trajectories. `pace` changes actor update cadence while preserving the native graph update path. The main comparison measures the complete learned systems. The cadence control separates a change in actor optimization rhythm from a change in retrieval memory. Collection calls, actual update calls, generated trajectories, tokens, wall time, and memory usage are distinct budget fields.

With all three main conditions present, [summary.py](summary.py) reports `system_comparison`: `total` is EvoCond minus SEARL, `cadence` is pace minus SEARL, and `matched_cadence` is EvoCond minus pace. The last two add to the total for each benchmark and seed. These comparisons describe independently trained systems; the frozen mechanism panels below evaluate individual decisions.

`main/searl` uses the native graph; `ablation/semantic` supplies the semantic-reading control. Job names are frozen in `matrix.json` and each job's `protocol.json`.

### Training Ablations

| Label | Intervention | Preserved components |
|---|---|---|
| `evocond` | Full reference | All components |
| `semantic` | Semantic-only presentation | Tool admission and common execution |
| `pooled` | Context-erased statistics and fixed alpha 0.5 | Identity alignment and execution budgets |
| `global` | Context-erased relation statistics | Identity alignment, adoption teacher, budgets |
| `identity` | No learned identity projection | Conditional evidence, adoption teacher, budgets |
| `fixed` | Fixed alpha 0.5; no teacher fitting | Conditional evidence and identity alignment |
| `raw` | Raw context distance; no learned context projection | Identity alignment and adoption teacher |

These experiments independently train actor and memory from the common starting checkpoint. Their final inventories are outcomes, not matched covariates. For a fixed-actor mechanism comparison, separately freeze the same actor, memory bank, candidate pool, task order, and token budget, then replace only the target retrieval or adoption decision. Do not interpret independently retrained systems as same-prefix interventions.

The core comparison crosses the evidence representation with the adoption rule:

| Relation evidence | Fixed alpha 0.5 | Learned current-value adoption |
|---|---|---|
| Context-erased | `pooled` | `global` |
| Conditional | `fixed` | `evocond` |

Each row tests adoption with a fixed evidence rule; each column tests conditional evidence with a fixed adoption rule. With all four labels present, `summary.py` adds `conditional_adoption`: the four simple effects, each factor's effect averaged over the other factor, and the interaction `evocond - fixed - global + pooled`. Coefficients and per-seed estimates are included. Differences use accuracy fractions; multiply by 100 for percentage points. These are contrasts between independently trained systems. Compare against `semantic` to measure retained assistance and against `simple` to measure the benefit of learned representations. Frozen paired evaluations distinguish retained rescues from avoided harms, with full cost reported alongside utility.

Generate this four-condition subset in a new run root; frozen matrices and completed experiments are preserved:

```bash
python run.py configs --root runs/relations --suite ablation --conditions pooled global fixed evocond --run
```

The statistical suite compares full EvoCond, `simple` (raw geometry and untrained identity with the same conditional bandit), and `bernstein` (only the confidence rule changes). `simple` is a combined reference, not a single-module ablation. Sensitivity sweeps use validation only; test results do not select thresholds, confidence rules, or checkpoints.

### Epoch Tails and Checkpoints

Before a suite starts, the runner checks every selected job against its frozen configuration, including normalized paths, sampling settings, training budget, runtime configurations, and the explicit recovery checkpoint. A missing runtime configuration or an existing checkpoint without a recovery selection stops the suite before the first worker starts. Checkpoint preflight checks the FSDP world size against the GPU allocation, every rank's nonempty model, optimizer and scheduler/RNG shard, and nonempty paired state files. The runtime's experiment identity, configuration hash, and recorded step must match that actor checkpoint. These file and metadata checks precede native deserialization.

Configuration and launch-lock files are published only after a complete write. If matrix generation is interrupted, repeat the same command: complete matching files are retained and missing files are generated. A failed native batch concatenation restores the original batch references for a clean retry.

No base question is dropped or duplicated to fill a batch. The last collection batch retains its actual size. A short final actor-memory window publishes all its events and updates the actor once. At the next epoch, the two-batch schedule restarts. The frozen schedule records `questions_per_epoch`, `window_questions`, `tail_questions`, and `total_actor_updates`.

For 10,000 training questions and batch 256, the schedule contains 40 collection batches, with 16 questions in the final batch. Native SEARL performs 40 actor update calls; `pace` and EvoCond perform 20. The final two-batch update covers 272 questions. These counts describe calls into the native actor updater; optimizer minibatch steps remain part of the native training log.

Save and validate at declared actor barriers and the final short window. Recovery restores the paired actor, optimizer, data iterator, graph, registry, and teacher state. An interrupted incomplete window resumes from the preceding paired checkpoint, replaying that window under the same frozen inputs. Preparation is idempotent only for identical artifacts. A changed source or protocol uses a new run directory.

Recovery starts from a fresh runtime or a released actor-memory barrier. It checks the complete actor, registry, representation, and teacher binding before replacing the live inventory. Rejected recovery requests preserve live episodes, pending memory, and the current inventory. An explicit checkpoint map must be nonempty and cover every selected job; restore arguments apply to training and evaluation. Artifact paths use forward slashes relative to their declared root, including when a manifest moves between Windows and Linux.

Native batch assembly validates sequence lengths, response masks, metadata, and window readiness before padding or appending to the pending window. A rejected batch leaves the pending window available for correction and retry.

### Test Evaluation

Use one greedy answer per scheduled base question, `do_sample=false`, temperature 0, top-p 1, and identical execution limits across conditions. Select checkpoints using validation before test evaluation. A checkpoint map explicitly names every condition/seed checkpoint; test access requires `--allow-test`.

Evaluate each benchmark separately. Shared memory is frozen on test, while newly created tools remain private to their current task. The v2 judge uses the declared reference set as the denominator: missing or failed predictions become empty answers and score zero; duplicate predictions and identity mismatches raise errors. Corrupt judge responses do not become model failures. The same Qwen3-32B snapshot, parser, and judge generation settings score all conditions. Four-seed aggregation matches the dataset content hash, input format, question mapping, answer mapping, and reference count. Moving identical input files keeps their identity; changing their interpreted fields creates a different evaluation.

Four training seeds produce four accuracy values per condition and benchmark. Report their mean and sample standard deviation; repeated training rollouts are not extra test questions. Report per-benchmark values first, then explicitly distinguish benchmark macro-average from question-weighted average. `summary.py` requires the complete benchmark-by-seed grid, matching test fingerprints and denominators, and the same judge settings. Question-paired intervals remain separate from between-seed variation.

[benchmark.py](benchmark.py) also defines native evaluation interfaces for [ToolHop](https://huggingface.co/datasets/bytedance-research/ToolHop), [ToolSandbox](https://github.com/apple-aiml-research/ToolSandbox), and [BigCodeBench](https://github.com/bigcode-project/bigcodebench). Implement `NativeEvaluator.run` with public task inputs and `NativeEvaluator.grade` with the benchmark's private reference and official scoring rules. The interface checks the complete case collection and protocol hash before execution, supplies independent copies to adapters, and retains validated receipts against the scheduled denominator. Binary correctness and ToolSandbox's continuous milestone similarity keep their native scales. Plan output publishes `report.json` after the protocol and complete public requests; an interrupted export uses a fresh output directory.

Judge identity includes the scoring source and wrapper hashes as well as the model, parser, metric, and generation settings. The summary hashes the manifest bytes it actually parses. Reports are published as complete files, and an existing output is preserved.

The [ARPO protocol](https://arxiv.org/html/2507.19849v1) motivates declaring cold-start and sampling budgets explicitly; [AgentFlow, Section 4.3](https://arxiv.org/html/2510.05592#S4.SS3) separates training-strategy changes from fixed execution modules. EvoCond's comparison applies the same separation to actor cadence, condition representation, and adoption decisions.

### Memory Fatigue

Structural memory fatigue is a decline in the incremental terminal utility of structural guidance relative to semantic retrieval as training and relation accumulation proceed. Measure both utility and its components: raw task success, full cost, and the clipped cost used by the declared reward. A zero net gain can hide offsetting rescues and harms. A negative net gain can also arise from additional cost without any lost successes. Paired transitions and cost differences identify which observed changes contribute to the average.

The distinction matters during actor learning: successful execution with memory includes both actor capability and memory contribution. Actor progress can raise absolute success while memory's additional contribution declines. EvoCond learns adoption relative to semantic retrieval on the current actor and inventory, while relation feedback retains observational support. Structural guidance can therefore lose influence without deleting a capability needed by another task.

#### Training Trace

`phases/phaseN.json` now records `memory_sizes.read_snapshot`, `memory_sizes.pending_publication`, and `memory_value`. Inventory counts refer to distinct capabilities, versions, relations, and retained edge events. `memory_value` compares semantic presentation with a prespecified uniform distribution over distinct nonsemantic actual payload classes at the randomized point. Identical payloads count once. With logging probability `p`, the contrast weight is `(target_probability - semantic_probability) / p`; a fixed reward midpoint centers the estimate. This known-propensity IPS diagnostic requires no fitted reward model. The adoption learner retains its cross-fitted DR objective.

Every preselected question stays in the denominator. Unreached points, unsupported histories, and a single actual class contribute zero. No collected observations yields an unavailable mean rather than a zero-valued memory effect. Reports include the smallest logging probability, largest contrast weight, sample dispersion, and measured success and costs. Relation-support receipts separately record deduplicated queried edges, conditional mass, query effective support, and signed edge counts; their statistics cover only randomized points with recorded support. Legacy logs without these fields keep support unavailable. Training traces summarize the random-point effect under each logged continuation; the frozen panel below evaluates full presentation policies on the same probe tasks. Completed phases are included; an unfinished tail is counted separately. Neither output selects a deployment policy.

Zero-contrast observations still carry a checked terminal reward and cost receipt. In absolute one-point policy evaluation, their realized reward remains in the full-question average. Each evaluation window uses one frozen snapshot and unique base questions; fitting questions from any branch of a mixed teacher are excluded from its evaluation set.

Live inspection first captures the completed phase reports, then streams a fixed byte prefix of the intervention log. `intervention_sha256` hashes those exact bytes and `intervention_bytes` identifies their length; later appends enter the next inspection. An unfinished final record is counted in `incomplete_tail_bytes`, while missing or extra observations for a completed phase remain errors. The reader releases each phase's raw observations after summarizing it. Output files are published only after the complete report has been written, and existing reports are preserved.

#### Frozen Actor-by-Memory Panel

Freeze two actor snapshots and two relation banks before the probe evaluation. Keep capability identities, tool versions, task set, candidate-generation rule, payload budget, execution protocol, and cost scale common. Relation banks differ in retained relation evidence. In all four cells, run each scheduled base question under `semantic` and `structural` presentation. Structural presentation uses the same fixed rule in every cell; trained teachers do not enter the attribution target.

| Cell | Actor | Relation evidence | Target |
|---|---|---|---|
| `actor0_bank0` | Earlier checkpoint | Earlier bank | Earlier structural lift |
| `actor0_bank1` | Earlier checkpoint | Later bank | Bank effect at fixed actor |
| `actor1_bank0` | Later checkpoint | Earlier bank | Actor effect at fixed bank |
| `actor1_bank1` | Later checkpoint | Later bank | Later structural lift |

Let `B00`, `B01`, `B10`, and `B11` be structural-minus-semantic net utility. Compute paired task contrasts before averaging: actor effect `B10-B00`, bank effect `B01-B00`, interaction `B11-B10-B01+B00`, and total change `B11-B00`. The three effects sum to the total change. Further condition-erasure and budget controls locate the mechanism behind each effect. Repeat on seeds 42, 52, 3047, 3407 and use `--panels` for the mean and sample standard deviation over training runs.

The four-seed output retains absolute success, cost, clipped cost, and utility for every presentation arm. Each cell's `paired_success.structural_vs_semantic` contains the following observed transitions:

| Semantic result | Structural result | Count |
|---|---|---|
| Failure | Failure | `both_failed` |
| Failure | Success | `rescued` |
| Success | Failure | `harmed` |
| Success | Success | `both_succeeded` |

`rescue_share` and `harm_share` divide their counts by all scheduled questions; their difference is the success-rate lift. `rescue_rate` divides by semantic failures, while `harm_rate` divides by semantic successes. A stronger semantic execution reduces the available errors and expands the initially successful group. Even unchanged conditional rescue and harm rates can therefore produce a reversal in net success lift. Compare these counts and rates across the frozen actor-by-bank cells before attributing a smaller gain to memory quality.

The panel adds `opportunity_changes` for `actor_at_old_bank`, `actor_at_new_bank`, and `total_change`. Each comparison splits the observed utility change into `opportunity_mix`, `rescue_rate`, `harm_rate`, and `cost`; these sum to `total`. The opportunity term changes the proportions of semantic failures and successes, while the two response terms change guidance outcomes within those groups. Symmetric midpoint weights make every term reverse sign when the comparison is reversed. The cost contribution is the negative change in the clipped cost penalty.

This accounting locates a decline that comes from fewer rescue opportunities, changed guidance responses, or greater cost. It describes observed groups, whose membership can change with the actor. An empty opportunity group leaves the three opportunity/response terms undefined; observed total and cost changes remain available. Four-seed summaries retain `per_seed` values and `defined_seeds`, and publish a complete mean only when all four seeds define the term. The teacher continues to learn current relative utility from randomized presentation.

Each cell's `benefit_components` accounts for the same full task set: `net_utility_lift = rescue_share - harm_share - cost_penalty`. The signed `cost_penalty` is the frozen reward coefficient times the alternative-minus-reference clipped cost; cost savings make it negative. Full costs remain in the outcome summaries. Optional adopted outcomes receive the same accounting against both semantic and structural execution. Four-seed summaries retain each component's per-seed mean and between-run variation. This separates fewer rescue opportunities, more guidance damage, and greater cost without changing the adoption target.

If an opportunity group is empty, its conditional rate is `null`. Four-seed summaries keep the full `per_seed` counts and `defined_seeds`; a conditional metric missing in any seed has no four-seed mean or standard deviation. Unconditional success lift and net utility retain all four seeds. Optional adopted outcomes add both `adopted_vs_semantic` and `adopted_vs_structural` transition reports.

Paired transitions require both observed outcomes for the same task under the declared decoding and execution protocol. The single-branch randomized training trace estimates marginal contrasts and does not reconstruct a joint transition table. These conditional rates explain the evaluation; the adoption learner continues to optimize net utility over the complete question denominator.

Populate `configs/fatigue.example.json` with actual checkpoint, bank, common-protocol, and cell-file SHA256 values. Set integer `questions` to the full scheduled probe count and `base_ids_sha256` to `control.dynamic.digest(sorted(scheduled_base_ids))`, derived from the frozen probe reference. Paths are relative to the manifest. Each cell file is a JSON object with these fields:

```json
{
  "format": "evocond_fatigue_cell_v1",
  "actor": "<ACTOR_SHA256>",
  "bank": "<RELATION_BANK_SHA256>",
  "seed": 42,
  "protocol_sha256": "<CANONICAL_PROTOCOL_SHA256>",
  "records": []
}
```

`protocol_sha256` is `control.dynamic.digest(manifest["protocol"])`. Export one record per scheduled task with `base_id`, `semantic`, and `structural`. Each outcome has a Boolean `success` and measured `costs` with numeric `actor`, `tools`, and `retrieval` entries. Failed, timed-out, or missing answers remain explicit failed outcomes with their incurred costs. All four files must cover the predeclared reference, including failures; matching subsets cannot reduce the denominator. The importer rejects missing cells, unmatched tasks, repeated checkpoints, corrupt checksums, and nonfinite costs. UTF-8 files with or without a byte-order mark are accepted, while checksums identify the original file bytes.

To measure adoption recovery, add `adopted` outcomes for every task in every cell and a `teacher_sha256` identifying that cell's frozen, current-basis teacher. The report gives adopted-minus-semantic and adopted-minus-structural utility independently of the fixed-structure decomposition. Record call changes alongside payload changes when testing whether recovered utility comes from actual execution decisions.

The comparison targets the learning trajectory of the same actor family. [REM](https://arxiv.org/abs/2610.27377v1) studies in-loop recall and recipient capability; [Anti-Persistence](https://arxiv.org/abs/2610.27406v1) reconstructs temporary guidance from retained trajectories; [the continual-learning study](https://arxiv.org/abs/2610.25858v1) exposes interference under drifting workflows. [CMP](https://arxiv.org/abs/2610.02070v3) identifies query-level utility through randomized retrieval exposure, and [MemCoRL](https://aclanthology.org/2026.acl-long.1804.pdf) alternates retrieval and utilization training. EvoCond studies conditional tool relations whose adoption value changes as the same actor learns, using the crossed panel to distinguish opportunity changes, observed harms, and cost.

## Acknowledgments

The dynamic tool environment and distributed training runtime build on [SEARL](https://github.com/circles-post/SEARL) and [verl](https://github.com/volcengine/verl). Third-party source retains its original copyright notices and licenses.


<details>
<summary>Apache 2.0 license for the adapted SEARL and verl sources</summary>

```text

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

</details>

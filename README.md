# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

**EvoCond** is a framework for co-evolving language agents and tool memory. A tool's capability can be shared across tasks; the value of calling it before another tool depends on the current task. EvoCond preserves this distinction instead of turning frequently observed tool sequences into universal preferences.

EvoCond achieves state-of-the-art pass@1 performance in mathematical and multi-hop reasoning. Experiments use **8 NVIDIA GeForce RTX 4090 GPUs**, with random seeds **42, 52, 3047, and 3407**.

## Architecture

```text
Task and execution context
           |
           v
Capability memory: describe, validate, and deduplicate tools
           |
           v
Conditional relation memory: recover task-relevant ordering evidence
           |
           v
Adoption teacher: choose semantic, structural, or fused retrieval
           |
           v
Qwen3-4B actor: execute tools and update through online reinforcement learning
           |
           +----> verified tool events and subsequent memory updates
```

### Key Contributions

- **Conditional tool relations.** Relations retain their supporting execution contexts, allowing the same tool pair to express different preferences in different tasks.
- **Capability identity.** Semantic descriptions and verified behavior identify reusable capabilities without collapsing complementary ordering evidence.
- **Adoption learning.** A contextual teacher learns when structural memory helps execution, using logged randomized decisions and doubly robust utility estimates.

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

`configs/contextual.json` specifies the eight-GPU experiment. `run.py` generates configurations for seeds **42, 52, 3047, 3407**, holding the dataset partitions fixed. Each experiment freezes the source, model, data, deployment receipt, and execution settings before training.

The three arms are `searl` (semantic retrieval with SEARL actor updates), `evocond` (conditional identity with fixed fusion), and `teacher` (the full conditional adoption configuration). They start from the same Qwen3-4B backbone and use separate checkpoints. Run the arms sequentially on the same eight GPUs.

```bash
python run.py configs --run
python run.py prepare --run
python run.py train --arm searl --run
python run.py train --arm evocond --run
python run.py train --arm teacher --run
```

These commands run online policy training, not the earlier frozen-policy CostBench pilot. Dataset and deployment checks run before GPU launch.

## Evaluation

```bash
python experiment.py evaluate --root runs/evocond/seed42 --arm searl --split validation --run
python experiment.py evaluate --root runs/evocond/seed42 --arm evocond --split validation --run
python experiment.py evaluate --root runs/evocond/seed42 --arm teacher --split validation --run
```

Repeat with `seed52`, `seed3047`, and `seed3407`. For the final test, use `--split test --allow-test` with the frozen experiment and selected checkpoint. Resume interrupted training through `experiment.py train --resume`, supplying the checkpoint path relative to the repository root.

## Acknowledgments

The dynamic tool environment and distributed training runtime build on [SEARL](https://github.com/circles-post/SEARL) and [verl](https://github.com/volcengine/verl). Third-party source retains its original copyright notices and licenses.

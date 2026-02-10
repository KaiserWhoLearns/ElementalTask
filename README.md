## Project Structure

```
ElementalTask/
├── configs/                    # Model checkpoint configurations
│   ├── olmo2_checkpoints.json  # OLMo-2 1B/7B checkpoint definitions
│   └── ...
│
├── dataset/                    # Task datasets (CSVs)
│   ├── simple.csv              # Atomic ICL tasks (uppercase, lowercase, first_letter, etc.)
│   ├── simple_spaced.csv       # Spaced variant of atomic tasks
│   ├── compositional.csv       # Compositional tasks (chained operations)
│   ├── compositional_spaced.csv
│   ├── math_expressions.csv    # Math/arithmetic tasks
│   ├── textfrct/               # TextFRCT benchmark tasks
│   └── ...
│
├── tasks/                      # Task framework
│   ├── base_task.py            # BaseTask abstract class and TaskConfig
│   ├── registry.py             # Task discovery, registration, and listing
│   └── implementations/        # Task implementations
│       ├── simple_icl_task.py          # Atomic ICL tasks (10 categories)
│       ├── compositional_task.py       # Compositional tasks (chained ops)
│       ├── textfrct_task.py            # TextFRCT benchmark tasks
│       ├── math_task.py                # Math expression tasks
│       ├── basic_arithmetic_task.py    # Basic arithmetic
│       ├── copying_task.py             # Exact copying
│       ├── token_reversal_task.py      # Token reversal
│       ├── string_analogy_task.py      # String analogies
│       ├── ignoring_context_task.py    # Ignoring irrelevant context
│       ├── ioi_task.py                 # Indirect Object Identification
│       └── part_of_speech_task.py      # Part of speech tagging
│
├── function_vecs/              # Function vector extraction & analysis
│   ├── extract_function_vecs.py    # FV extraction (simple + detailed API)
│   └── experiments/                # FV analysis experiments
│       └── analyze_real_tasks.py
├── scripts/                    # Evaluation & analysis scripts
│   ├── eval_across_checkpoints.py          # Main checkpoint evaluation
│   ├── eval_array_job_v2.sh                # SLURM array job for batch eval
│   ├── eval_new_compositions.sh            # Eval for new compositional tasks
│   ├── eval_checkpoints.sh                 # Single-node eval script
│   ├── run_unified_eval.py                 # Unified evaluation runner
│   └── trajectory_analysis/                # Developmental trajectory analysis
│       └── predict_compositional_from_components.py
│
├── analysis/                   # Plotting & visualization
├── results/                    # Evaluation outputs
├── plots/                      # Generated figures
├── logs/                       # SLURM job logs
└── tests/                      # Unit tests
    └── test_basic_icl_tasks.py
```

## Past Notes
    - Olmo 1
    - Olmo 2
    - LLM360
    - Emmy's Adhoc ckpts

* Tasks (phase I - sanity check)

    - Exact copying/semantic copying
        - Copy the input:
            Input: xyzabc
    - Arithmetic
    - Synonyms/antonyms
    - Parallel structures/templates
    - Reversal/token ops
        - Reverse the word cat: tac
    - Factual recall
        - facebook/kilt_tasks

* Complex Tasks: How can we construct a complex task with the elemental tasks we proposed?
    - extractiveQA = Understanding + Reasoning + Exact Copy
    - Opendomain QA = Memorization + Understanding + Reasoning + Lingusitics
    - Natural Langauge Inference = Understanding + Reasoning


```
Example usage:
  # List all available checkpoints
  python scripts/measure_ckpt_interp_perf.py --model_id allenai/OLMo-1B-hf --list_checkpoints_only

  # Evaluate 10 uniformly sampled checkpoints
  python scripts/measure_ckpt_interp_perf.py --model_id allenai/OLMo-1B-hf --use_vllm

  # Evaluate specific checkpoints
  python scripts/measure_ckpt_interp_perf.py --model_id allenai/OLMo-1B-hf --checkpoints step10000-tokens42B step50000-tokens210B --use_vllm
```





Usage Examples (Generate images):
```
  # Generate all plot types
  python analysis/plotting.py --csv_path output/olmo2_ckpt_interp_results_firstckpts.csv

  # Generate only performance curves
  python analysis/plotting.py --csv_path
  output/olmo2_ckpt_interp_results.csv --plot_type curves

  # Generate grouped task curves
  python analysis/plotting.py --csv_path
  output/olmo2_ckpt_interp_results.csv --plot_type grouped

  # Custom output directory and figure size
  python analysis/plotting.py --csv_path
  output/olmo2_ckpt_interp_results.csv --output_dir my_plots --figsize
   16 10
```

### Function vector and basis extraction

The code to extract function vectors from tasks and form a basis lives in the dir `function_vecs`. Note that the code in this section assumes that the task has an ICL representation that makes sense.

The function vector is an L2-normalized vector which corresponds to how the top-k most informative attention heads contribute differently in the residual stream across ICL examples versus control examples (ones where the input -> output mapping is shuffled).

#### Simple vs Complex Extraction

We provide two interfaces for function vector extraction:

**Simple Interface** - One-stop function for quick extraction:
- Automatically configures model loading, head selection, and extraction parameters
- Works with existing task registry and provides sensible defaults
- Best for: Rapid prototyping, standard use cases, getting started

**Detailed Interface** - Full control over extraction pipeline:
- Manual configuration of models, heads, sampling, and extraction parameters  
- Supports custom task configurations and advanced head selection strategies
- Best for: Research experiments, custom tasks, fine-tuned control

#### Quick Start: Simplified Interface

```python
from function_vecs.extract_function_vecs import extract_function_vector_simple

# Simple usage - uses all defaults
function_vec = extract_function_vector_simple("simple_icl", num_samples=10)
print(f"Function vector shape: {function_vec.function_vec.shape}")
print(f"Task name: {function_vec.task_name}")

# With custom model and device
function_vec = extract_function_vector_simple(
    task_name="simple_icl",
    model_name="distilgpt2",
    num_samples=5,
    device="cpu",
    layer_idx=5  # Use specific layer
)
```

Available tasks in registry: `["simple_icl", "simple", "textfrct", "math"]`

#### Advanced Usage: Detailed Interface

```python
from function_vecs.extract_function_vecs import extract_task_function_vec, ExtractConfig, Headset
from tasks.base_task import TaskConfig
from tasks.registry import get_task

# 1. Define your task or import one

# Create task with custom in-memory data
task_config = TaskConfig(
    name="uppercase_conversion",
    data_format="memory", 
    in_memory_data=[
        {"input": "a", "output": "A"},
        {"input": "b", "output": "B"},
        {"input": "c", "output": "C"}
    ]
)
task_from_config = get_task("simple_icl", task_config)  # Use simple_icl as base class

# Use existing task from registry
simple_icl_config = TaskConfig(
    name="simple_icl", 
    data_path="dataset/simple.csv",
    input_column="question",
    output_column="answer"
)
task = get_task("simple_icl", simple_icl_config)

# 2. Configure extraction
extract_config = ExtractConfig(
    model_name="gpt2",
    num_samples_per_task=10,
    batch_size=4,
    layers=[11],  # Focus on specific layers
    device="auto"
)

# 3. Create headset (specify which attention heads to analyze)
head_set = Headset(mode="topk", heads=[(11, 0), (11, 1), (11, 2)])

# 4. Extract function vectors
function_vec_from_config = extract_task_function_vec(task_from_config, extract_config, head_set)
print(f"Custom task: {function_vec_from_config}")

function_vec_from_task = extract_task_function_vec(task, extract_config, head_set)
print(f"Registry task: {function_vec_from_task}")
```

#### Forming a basis from multiple tasks

```python
from function_vecs.extract_function_vecs import (
    extract_function_vector_simple, 
    stack_function_vecs, 
    build_skill_basis
)

# Extract function vectors for multiple tasks
tasks = ["simple_icl"]  # Add more tasks as they become available
function_vecs = []

for task_name in tasks:
    print(f"Extracting function vector for {task_name}...")
    vec = extract_function_vector_simple(task_name, num_samples=10)
    function_vecs.append(vec)

# Stack vectors into a matrix
task_matrix = stack_function_vecs(function_vecs)
print(f"Task matrix shape: {task_matrix.V.shape}")

# Build skill basis using SVD
skill_basis = build_skill_basis(task_matrix, method="svd", k=6)
print(f"Skill basis dimensions: {skill_basis.U.shape}")
print(f"Explained variance ratios: {skill_basis.S / skill_basis.S.sum()}")

# Analyze task relationships in the skill space
task_projections = skill_basis.Vt  # Tasks projected onto skill dimensions
print(f"Task projections shape: {task_projections.shape}")
```



## Developmental TODOs

* Data Preparation
  * Write everything to `data` dir, with the task name as the sub-directory
  * A universal data format
    * Locally as `HF dataset .jsonl`, with `lm_input` indicate the input to the langauge model, and `reference` be the expected output.
* Model Inference
  * Pass model name, checkpoint
  * Run inference (write to `outputs`) / evaluation
* Data Saving
  * Kaiser made in save directory somewhere

* minor TODO: current version of VLLM is incompatible, need to search backward until we find a good one? (speculative, not sure but seems like the issue is recent) - maybe fix VLLM later for faster generation... (Millicent)

#
Test
```
python models/evaluate_models.py \
  --model_id LLM360/Crystal \
  --max_new_tokens 5 \
  --chkpt main 

python models/evaluate_models.py \
  --task_name FRCT_CV1_ScrambledWords \
  --max_new_tokens 5 \
  --chkpt main 

```


# Function Vectors Extraction

This directory contains tools for extracting **function vectors** from language models. Function vectors are L2-normalized vectors that capture how attention heads contribute to the residual stream differently for in-context learning (ICL) examples vs. control examples (where input→output mappings are shuffled). They represent task-specific information learned by the model.

## Table of Contents

- [Quick Start](#quick-start)
- [Two Extraction Interfaces](#two-extraction-interfaces)
  - [Simple Interface (Recommended)](#1-simple-interface-recommended)
  - [Advanced Interface](#2-advanced-interface-full-control)
- [Building a Skill Basis](#building-a-skill-basis-from-multiple-tasks)
- [Discovering Available Tasks](#discovering-available-tasks)
- [Configuration Parameters](#key-configuration-parameters)
- [How It Works](#how-it-works)
- [Running Tests](#running-tests)

---

## Quick Start

```python
from function_vecs.extract_function_vecs import extract_function_vector_simple

# Extract function vector with one line
function_vec = extract_function_vector_simple("basic_arithmetic", num_samples=10)

print(f"Function vector shape: {function_vec.function_vec.shape}")
print(f"Task name: {function_vec.task_name}")
print(f"L2 norm: {function_vec.function_vec.dot(function_vec.function_vec):.6f}")  # ~1.0
```

---

## Two Extraction Interfaces

### 1. Simple Interface (Recommended)

The simple interface provides one-stop function extraction with automatic configuration. Best for rapid prototyping, standard use cases, and getting started.

#### Basic Usage

```python
from function_vecs.extract_function_vecs import extract_function_vector_simple

# Minimal usage - uses all defaults
function_vec = extract_function_vector_simple("basic_arithmetic")

# With custom parameters
function_vec = extract_function_vector_simple(
    task_name="simple_icl",
    model_name="gpt2",          # or "distilgpt2", "EleutherAI/gpt-j-6B", etc.
    num_samples=20,              # Number of examples to use
    device="cuda",               # "auto", "cuda", or "cpu"
    layer_idx=11                 # Specific layer (None = use last layer)
)

print(f"Task: {function_vec.task_name}")
print(f"Shape: {function_vec.function_vec.shape}")
print(f"Normalization: {function_vec.normalization}")  # "l2"
```

#### Available Tasks

Tasks are auto-discovered from the task registry. Available tasks include:
- `basic_arithmetic` - Basic arithmetic operations
- `simple_icl` - Simple in-context learning tasks
- `simple` - Simple task examples
- `textfrct` - TextFRCT dataset tasks
- `part_of_speech` - Part of speech identification
- `token_reversal` - Token reversal operations
- `math` - Mathematical reasoning tasks
- `ioi_task` - Indirect object identification

See [Discovering Available Tasks](#discovering-available-tasks) to list all tasks dynamically.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_name` | str | Required | Name of task from registry |
| `task_config` | TaskConfig | None | Optional custom config |
| `model_name` | str | `"gpt2"` | HuggingFace model identifier |
| `num_samples` | int | `10` | Number of examples to use |
| `device` | str | `"auto"` | Device: "auto", "cuda", or "cpu" |
| `layer_idx` | int | None | Layer to extract from (None = last) |

#### Returns

A `TaskFunctionVec` object with:
- `task_name`: Name of the task
- `function_vec`: The extracted function vector (numpy array)
- `normalization`: Normalization method used (default: "l2")

---

### 2. Advanced Interface (Full Control)

The advanced interface provides manual configuration of models, heads, sampling, and extraction parameters. Best for research experiments, custom tasks, and fine-tuned control.

#### Complete Example

```python
from function_vecs.extract_function_vecs import (
    extract_task_function_vec,
    ExtractConfig,
    Headset,
    extract_informative_heads
)
from tasks.registry import get_task
from tasks.base_task import TaskConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda").eval()

# Step 2: Get task from registry
task_config = TaskConfig(
    name="basic_arithmetic",
    data_path="dataset/simple.csv",  # Optional: custom data path
    input_column="question",
    output_column="answer"
)
task = get_task("simple_arithmetic", task_config)

# Step 3: Configure extraction parameters
config = ExtractConfig(
    model_name=model_name,
    device="cuda",
    num_samples_per_task=20,
    batch_size=8,
    layers=[11],                      # Which layers to analyze
    topk_heads=10,                    # Number of heads to select
    head_selection="topk",            # "topk" or "soft"
    seed=42
)

# Step 4: Select informative heads
# Option A: Automatic selection using AIE (Attention Importance Estimation)
headset = extract_informative_heads(config, [task])
print(f"Selected heads: {headset.heads}")

# Option B: Manual specification
headset = Headset(
    mode="topk",
    heads=[(11, 0), (11, 1), (11, 2), (11, 5)]  # (layer, head) tuples
)

# Step 5: Extract function vector
function_vec = extract_task_function_vec(
    task=task,
    config=config,
    head_set=headset,
    model=model,
    tokenizer=tokenizer
)

print(f"Task: {function_vec.task_name}")
print(f"Function vector shape: {function_vec.function_vec.shape}")
```

#### Creating Custom Tasks

You can create tasks with in-memory data:

```python
from tasks.base_task import TaskConfig
from tasks.registry import get_task

# Create task with custom in-memory data
task_config = TaskConfig(
    name="uppercase_conversion",
    data_format="memory",
    in_memory_data=[
        {"input": "a", "output": "A"},
        {"input": "b", "output": "B"},
        {"input": "c", "output": "C"},
        {"input": "hello", "output": "HELLO"}
    ],
    input_column="input",
    output_column="output"
)
task = get_task("simple_icl", task_config)  # Use simple_icl as base class

# Now extract function vector for this custom task
function_vec = extract_task_function_vec(task, config, headset)
```

---

## Building a Skill Basis from Multiple Tasks

Extract function vectors from multiple tasks and create a shared skill basis using SVD. This reveals the underlying "skill dimensions" shared across tasks.

```python
from function_vecs.extract_function_vecs import (
    extract_function_vector_simple,
    stack_function_vecs,
    build_skill_basis
)

# Step 1: Extract function vectors for multiple tasks
task_names = ["basic_arithmetic", "simple_icl", "token_reversal", "part_of_speech"]
function_vecs = []

for task_name in task_names:
    print(f"Extracting function vector for {task_name}...")
    vec = extract_function_vector_simple(
        task_name,
        model_name="gpt2",
        num_samples=20
    )
    function_vecs.append(vec)

# Step 2: Stack vectors into a matrix
task_matrix = stack_function_vecs(function_vecs)
print(f"Task matrix shape (d_model x num_tasks): {task_matrix.V.shape}")
print(f"Tasks: {task_matrix.task_names}")

# Step 3: Build skill basis using SVD
skill_basis = build_skill_basis(
    task_matrix,
    method="svd",
    k=6  # Number of skill dimensions (-1 for auto-selection based on 95% energy)
)

print(f"Skill basis U shape: {skill_basis.U.shape}")  # (d_model, k)
print(f"Singular values: {skill_basis.S}")
print(f"Explained variance ratios: {skill_basis.S / skill_basis.S.sum()}")

# Step 4: Analyze task relationships in skill space
task_projections = skill_basis.Vt  # (k, num_tasks)
print(f"Task projections shape: {task_projections.shape}")

# Each column of Vt represents how much each task loads on each skill dimension
for i, task_name in enumerate(skill_basis.task_names):
    print(f"{task_name}: {skill_basis.Vt[:, i]}")
```

### Interpreting the Skill Basis

- **U**: Skill basis vectors in model space (d_model x k)
- **S**: Singular values indicating importance of each skill dimension
- **Vt**: Task loadings on skill dimensions (k x num_tasks)

The skill basis reveals:
1. Which underlying capabilities are shared across tasks
2. How tasks relate to each other in the skill space
3. The dimensionality of the "skill manifold"

---

## Discovering Available Tasks

### From Python

```python
from function_vecs.extract_function_vecs import discover_all_tasks

# List all available tasks with descriptions
task_names = discover_all_tasks()
```

### From Command Line

```bash
python function_vecs/extract_function_vecs.py
```

### Using Task Registry

```python
from tasks.registry import list_tasks, get_task_info

# List all task names
tasks = list_tasks()
print(f"Available tasks: {tasks}")

# Get detailed info about all tasks
task_info = get_task_info()
for task_name, info in task_info.items():
    print(f"{task_name}: {info['class']} - {info['docstring'][:100]}")

# Get info about a specific task
info = get_task_info("simple_arithmetic")
print(info)
```

---

## Key Configuration Parameters

### ExtractConfig

Complete configuration for the extraction process:

```python
@dataclass
class ExtractConfig:
    # Model configuration
    model_name: str = "EleutherAI/gpt-j-6B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    seed: int = 42
    layers: Optional[List[int]] = None  # If None, use all layers

    # Sampling configuration
    num_samples_per_task: int = 20
    num_shuffled_controls_per_task: int = 10

    # Head selection configuration
    head_selection: Literal["topk", "soft"] = "topk"
    topk_heads: int = 10
    cached_headset_path: Optional[str] = None

    # Basis configuration
    basis_method: Literal["svd", "pca"] = "svd"
    basis_dim: int = 20
    eps: float = 0.01  # for eps-rank
```

### Headset

Specifies which attention heads to analyze:

```python
@dataclass
class Headset:
    mode: Literal["topk", "soft"]  # Selection mode
    heads: List[Tuple[int, int]]    # List of (layer, head) tuples
    weights: Optional[np.ndarray]   # Optional weights for "soft" mode
```

**Two modes:**
- `topk`: Sum contributions from top-k selected heads equally
- `soft`: Weighted combination using provided weights

---

## How It Works

The function vector extraction pipeline:

1. **Sample ICL Prompts**: Get in-context learning examples from the task
   - Uses task's test split
   - Formats prompts using task-specific templates

2. **Generate Control Prompts**: Create shuffled control examples
   - Breaks input→output mapping by shuffling
   - Same format as ICL prompts but with incorrect mappings

3. **Compute Head Importance** (if using `extract_informative_heads`):
   - Uses AIE (Attention Importance Estimation)
   - Measures performance drop when each head is swapped with control
   - Selects top-k most informative heads

4. **Extract Per-Head Contributions**:
   - For each prompt, compute how each attention head contributes to residual stream
   - Use hooks to capture pre/post projection tensors
   - Decompose output projection into per-head contributions

5. **Average Across Examples**:
   - Mean contribution of each head across all ICL examples
   - Results in (d_model, num_heads) matrix

6. **Collapse to Function Vector**:
   - Sum (or weighted sum) across selected heads
   - Results in single d_model-dimensional vector

7. **Normalize**:
   - Apply L2 normalization (default)
   - Final function vector has unit norm

### Mathematical Details

For a given attention head h in layer l:
- Let O_h be the head's output before the output projection
- Let W_o^h be the corresponding columns of the output projection
- Head contribution: c_h = O_h @ W_o^h

Function vector:
```
v = normalize(Σ_h∈H c_h)
```

where H is the set of selected heads.

---

## Running Tests

The test suite validates both interfaces:

```bash
# Run all tests
pytest tests/test_simple_interface.py -v
pytest tests/test_function_vecs_revised.py -v
pytest tests/test_basic_icl_tasks.py -v

# Run specific test
pytest tests/test_simple_interface.py::test_simple_interface_existing_task -v

# Run with output
python tests/test_simple_interface.py
```

### Test Examples

The test files provide working examples:

**Simple Interface** ([tests/test_simple_interface.py](../tests/test_simple_interface.py)):
```python
# Basic usage
function_vec = extract_function_vector_simple(
    task_name="simple_arithmetic",
    model_name="distilgpt2",
    num_samples=3,
    device="cpu"
)

# Verify it's L2 normalized
assert abs(function_vec.function_vec.dot(function_vec.function_vec) - 1.0) < 1e-5
```

**Advanced Interface** ([tests/test_function_vecs_revised.py](../tests/test_function_vecs_revised.py)):
- Full extraction pipeline
- Custom head selection
- Multiple tasks and basis construction

---

## Architecture

### Key Files

- [`extract_function_vecs.py`](extract_function_vecs.py) - Main extraction logic
- [`model_internal_getters.py`](model_internal_getters.py) - Model architecture introspection
- [`activation_patching.py`](activation_patching.py) - Activation intervention tools

### Key Classes

- `ExtractConfig` - Configuration for extraction
- `Headset` - Specification of attention heads to analyze
- `TaskFunctionVec` - Container for extracted function vector
- `TaskMatrix` - Stack of multiple function vectors
- `SkillBasis` - SVD-based skill space representation

### Key Functions

- `extract_function_vector_simple()` - Simple one-stop interface
- `extract_task_function_vec()` - Advanced extraction with full control
- `extract_informative_heads()` - Automatic head selection using AIE
- `stack_function_vecs()` - Combine vectors into matrix
- `build_skill_basis()` - Construct skill basis via SVD

---

## Tips and Best Practices

### Model Selection
- Start with smaller models for testing: `distilgpt2`, `gpt2`
- Use larger models for production: `gpt2-medium`, `gpt2-large`, `EleutherAI/gpt-j-6B`

### Number of Samples
- More samples = more stable estimates
- Recommended: 10-20 for quick experiments, 50+ for research
- Trade-off: computation time vs. stability

### Layer Selection
- Last layer typically has most semantic information
- Middle layers may capture more syntactic patterns
- Experiment with different layers for your task

### Head Selection
- `extract_informative_heads()` automatically finds important heads
- Manual selection useful when you have domain knowledge
- Start with 5-10 heads, adjust based on results

### Device Management
- Use `device="auto"` to automatically select CUDA if available
- For large models, ensure sufficient GPU memory
- CPU works but is slower

### Batch Size
- Larger batches = faster but more memory
- Adjust based on your GPU memory and model size
- Default of 8 works well for most cases

---

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce `batch_size` or use smaller model

### Issue: Task not found
**Solution**: Run `discover_all_tasks()` to see available tasks, or check task name spelling

### Issue: Import errors
**Solution**: Ensure all dependencies are installed and task implementations are valid

### Issue: Function vector has unexpected shape
**Solution**: Check that model loaded correctly and layer_idx is valid

---

## Examples Gallery

### Example 1: Quick Extraction
```python
from function_vecs.extract_function_vecs import extract_function_vector_simple

vec = extract_function_vector_simple("basic_arithmetic")
print(f"Extracted {vec.function_vec.shape[0]}-dimensional vector for {vec.task_name}")
```

### Example 2: Compare Multiple Models
```python
models = ["distilgpt2", "gpt2", "gpt2-medium"]
vectors = {}

for model_name in models:
    vec = extract_function_vector_simple(
        "basic_arithmetic",
        model_name=model_name,
        num_samples=20
    )
    vectors[model_name] = vec.function_vec

# Compare vectors (cosine similarity, etc.)
```

### Example 3: Multi-Task Analysis
```python
tasks = ["basic_arithmetic", "simple_icl", "token_reversal"]
vecs = [extract_function_vector_simple(t, num_samples=20) for t in tasks]

# Build skill basis
task_matrix = stack_function_vecs(vecs)
skill_basis = build_skill_basis(task_matrix, k=-1)  # Auto-select k

print(f"Found {skill_basis.U.shape[1]} skill dimensions")
print(f"Capturing {skill_basis.S.sum():.2%} of variance")
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{elemental_tasks_function_vecs,
  title={Function Vector Extraction for Language Models},
  author={[Your Name/Team]},
  year={2025},
  url={https://github.com/[your-repo]/ElementalTask}
}
```

---

## Related Documentation

- [Main Project README](../README.md)
- [Task System Documentation](../tasks/README.md)
- [Model Evaluation Scripts](../scripts/)

---

## Questions?

For issues or questions:
1. Check the test files for working examples
2. Review the function docstrings in `extract_function_vecs.py`
3. Open an issue on the project repository

# Unified Task Evaluation System

This directory contains a unified task evaluation framework that consolidates the various approaches previously used in the project. The system provides a common interface for evaluating models on different tasks while supporting multiple data formats and model backends.

## Features

- **Unified Interface**: Common API for all tasks regardless of data source
- **Multiple Data Formats**: Support for CSV, JSON, and JSONL files
- **Multiple Model Backends**: vLLM, Transformers, OpenAI API, Together API
- **Flexible Task Definition**: Easy to create new tasks via configuration files
- **Comprehensive Evaluation**: Per-category metrics and detailed result saving
- **Extensible Design**: Easy to add new task types and evaluation metrics

## Quick Start

### 1. Basic Usage

```python
from tasks import create_task_from_config, TaskEvaluator, ModelConfig, EvaluationConfig
from tasks.simple_icl_task import SimpleICLTask

# Create a task
task = create_task_from_config('tasks/configs/simple_icl_tasks.json', SimpleICLTask)

# Configure model
model_config = ModelConfig(
    model_id="allenai/OLMo-1B-hf",
    backend="vllm",
    temperature=0.0,
    max_tokens=10
)

# Configure evaluation
eval_config = EvaluationConfig(output_dir="results")

# Run evaluation
evaluator = TaskEvaluator(model_config, eval_config)
results = evaluator.evaluate_task(task)
```

### 2. Command Line Usage

```bash
# Evaluate on Simple ICL tasks with vLLM
python run_unified_eval.py \
    --task_type simple_icl \
    --model_id allenai/OLMo-1B-hf \
    --backend vllm \
    --output_dir results

# Evaluate with OpenAI API
python run_unified_eval.py \
    --task_type simple_icl \
    --model_id gpt-4o-mini-2024-07-18 \
    --backend openai \
    --api_key YOUR_API_KEY \
    --output_dir results
```

## Architecture

### Core Components

1. **BaseTask**: Abstract base class defining the task interface
2. **SimpleTask**: Basic implementation with exact match evaluation
3. **SimpleICLTask**: Specialized task for in-context learning with category-based demonstrations
4. **TaskEvaluator**: Main evaluation engine supporting multiple model backends
5. **Configuration Classes**: Type-safe configuration for tasks, models, and evaluation

### Task Configuration

Tasks are defined using JSON configuration files:

```json
{
  "name": "my_task",
  "description": "Description of the task",
  "data_path": "path/to/data.csv",
  "data_format": "csv",
  "input_column": "question",
  "output_column": "answer",
  "num_demonstrations": 5,
  "evaluation_metrics": ["accuracy"]
}
```

### Supported Backends

- **vLLM**: High-performance inference for local models
- **Transformers**: Standard HuggingFace transformers
- **OpenAI**: OpenAI API models (GPT-4, etc.)
- **Together**: Together AI API models

## Creating New Tasks

### 1. Using Configuration Files

Create a JSON config file and use the `SimpleTask` class:

```python
from tasks.base_task import create_task_from_config, SimpleTask

task = create_task_from_config('my_task_config.json', SimpleTask)
```

### 2. Creating Custom Task Classes

Inherit from `BaseTask` and override key methods:

```python
from tasks.base_task import BaseTask

class MyCustomTask(BaseTask):
    def build_prompt(self, instance):
        # Custom prompt building logic
        return f"Question: {instance['input']}\nAnswer:"
    
    def evaluate(self, predictions, split="test", **kwargs):
        # Custom evaluation logic
        ground_truth = self.get_ground_truth(split)
        accuracy = compute_accuracy(predictions, ground_truth)
        return {"accuracy": accuracy}
```

## Examples

### Simple ICL Task

The `SimpleICLTask` demonstrates category-based in-context learning:

```python
# Each category gets specific demonstrations
category_demonstrations = {
    "uppercase": ["a -> A", "c -> C"],
    "lowercase": ["A -> a", "C -> c"],
    "translate_eng_fr": ["hello -> bonjour", "goodbye -> au revoir"]
}

# Prompts are built with category-specific examples
# For uppercase: "a -> A\nc -> C\nb ->"
```

### Evaluation Results

The system provides comprehensive evaluation metrics:

```python
{
    "accuracy": 0.8966,           # Overall accuracy
    "correct": 104,               # Total correct predictions
    "total": 116,                 # Total examples
    "accuracy_uppercase": 0.8750, # Per-category accuracy
    "accuracy_lowercase": 0.8750,
    # ... more per-category metrics
}
```

## File Structure

```
tasks/
├── __init__.py              # Task registry and imports
├── base_task.py             # Base classes and interfaces
├── evaluator.py             # Main evaluation engine
├── simple_icl_task.py       # ICL task implementation
├── textfrct_task.py         # TextFRCT integration (optional)
└── configs/
    └── simple_icl_tasks.json # Task configuration example
```

## Migration from Legacy Code

The unified system replaces the previous scattered approaches:

### Before (Multiple Approaches)
- `run_interp.ipynb`: Manual ICL with hardcoded examples
- `run.ipynb`: TextFRCT with OpenAI API
- `models/evaluate_models.py`: vLLM with task registry
- `scripts/evaluate_textfrct.py`: TextFRCT with multiple backends

### After (Unified System)
- Single `TaskEvaluator` class
- Common task interface
- Unified configuration format
- Consistent result format
- Support for all previous backends

### Migration Steps

1. **Replace notebook evaluations**:
   ```python
   # Old: Manual prompt building and evaluation
   # New: Use SimpleICLTask and TaskEvaluator
   ```

2. **Replace script-based evaluations**:
   ```bash
   # Old: Multiple different scripts
   # New: Single run_unified_eval.py with different --task_type
   ```

3. **Update model configurations**:
   ```python
   # Old: Various parameter formats
   # New: ModelConfig dataclass with validation
   ```

## Testing

Run the test suite to verify the system:

```bash
python test_unified_system.py
```

This will:
- Show example prompts for different categories
- Run mock evaluation with simulated predictions
- Display per-category accuracy metrics

## Future Extensions

The system is designed to be easily extensible:

1. **New Task Types**: Add new task classes for specific evaluation needs
2. **New Backends**: Add support for additional model serving frameworks
3. **New Metrics**: Extend evaluation with domain-specific metrics
4. **Batch Processing**: Add support for evaluating multiple models/tasks
5. **Caching**: Add result caching for expensive evaluations

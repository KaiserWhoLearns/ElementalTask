"""Base task class and configuration. When implementing a new task, please put it under tasks/implementations and inherit from BaseTask."""

import json
import csv
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path


@dataclass
class TaskConfig:
    """Configuration for a task."""
    name: str # the task name 
    description: Optional[str] = None # optional description
    data_path: Optional[str] = None # path to the main data file (CSV, JSON, etc.) for non-memory tasks
    data_format: str = "csv"  # 'csv', 'json', 'jsonl', 'memory'
    input_column: str = "input" # name of the input column in the data
    output_column: str = "output" # name of the output column in the data
    demonstrations_path: Optional[str] = None # path to demonstrations file (JSON, CSV, TXT)
    num_demonstrations: int = 5 # number of demonstrations to include in prompt. Will be max(num available, num_demonstrations)
    prompt_template: Optional[str] = None # optional prompt template with placeholders
    evaluation_metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    metadata: Dict[str, Any] = field(default_factory=dict)
    # For in-memory data (when migrating from hardcoded tasks)
    in_memory_data: Optional[List[Dict[str, Any]]] = None # option to just pass in-memory data (e.g. just a list)
    in_memory_demonstrations: Optional[Dict[str, List[str]]] = None


class BaseTask(ABC):
    """Base class for all tasks"""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.data = None
        self.demonstrations = None
    # track which example indices have been used for ICL sampling
    self._icl_used_indices = set()
        self._load_data()
        self._load_demonstrations()
    
    def _load_data(self):
        """Load the main task data from the specified path and format or use in-memory data."""
        # Check if we have in-memory data first
        if self.config.in_memory_data is not None:
            self.data = pd.DataFrame(self.config.in_memory_data)
            return
            
        # If no data_path provided, let subclass handle data loading
        if not self.config.data_path:
            # Subclasses should override this method to provide their own data
            # If they don't override it properly, self.data will remain None
            return
            
        # Otherwise load from file
        data_path = Path(self.config.data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        if self.config.data_format.lower() == 'csv':
            self.data = pd.read_csv(data_path)
        elif self.config.data_format.lower() == 'json':
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.data = pd.DataFrame(data)
        elif self.config.data_format.lower() == 'jsonl':
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            self.data = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data format: {self.config.data_format}")
    
    def _load_demonstrations(self):
        """Load demonstration examples from file, in-memory data, or use defaults."""
        # Check for in-memory demonstrations first
        if self.config.in_memory_demonstrations is not None:
            self.demonstrations = self.config.in_memory_demonstrations
            return
            
        # Otherwise try to load from file
        if self.config.demonstrations_path:
            demo_path = Path(self.config.demonstrations_path)
            if demo_path.exists():
                if demo_path.suffix == '.json':
                    with open(demo_path, 'r', encoding='utf-8') as f:
                        self.demonstrations = json.load(f)
                elif demo_path.suffix == '.csv':
                    self.demonstrations = pd.read_csv(demo_path)
                else:
                    # Try to load as text file with examples
                    with open(demo_path, 'r', encoding='utf-8') as f:
                        self.demonstrations = f.read().strip()
    @property
    def supports_icl(self) -> bool:
        """Check if this task supports ICL format"""
        return hasattr(self, 'get_icl_examples') or self.config.in_memory_data is not None

    def get_icl_examples(
        self,
        num_examples: int = 10,
        shuffle: bool = True,
        seed: Optional[int] = None,
        fresh: bool = True,
    ) -> List[Dict[str, str]]:
        """Return ICL-formatted examples (dicts with 'input' and 'output').

        Args:
            num_examples: number of examples to return.
            shuffle: whether to shuffle available examples before selection.
            seed: optional random seed for reproducibility.
            fresh: if True, prefer examples that haven't been returned before
                   (tracks indices in ``self._icl_used_indices``).
        """
        rows = self.get_split("test")
        if not rows:
            return []

        n = len(rows)
        indices = list(range(n))

        if shuffle:
            import random
            if seed is not None:
                random.seed(seed)
            random.shuffle(indices)

        if fresh:
            # choose indices not seen before; if insufficient, wrap around deterministically
            available = [i for i in indices if i not in self._icl_used_indices]
            if len(available) < num_examples:
                # include previously used indices to fill up
                available = available + [i for i in indices if i not in available]
            selected = available[:min(num_examples, n)]
            self._icl_used_indices.update(selected)
        else:
            selected = indices[:min(num_examples, n)]

        examples: List[Dict[str, str]] = []
        for i in selected:
            item = rows[i]
            examples.append({
                "input": item.get(self.config.input_column, ""),
                "output": str(item.get(self.config.output_column, "")),
            })
        return examples

    def reset_icl_tracking(self) -> None:
        """Reset internal tracking of used ICL examples so sampling starts fresh."""
        self._icl_used_indices.clear()

    def get_split(self, split: str = "test") -> List[Dict[str, Any]]:
        """Get data split as a list of dictionaries."""
        if split not in ["train", "val", "test", "all"]:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val', 'test', 'all']")
        
        # For now, return all data as test split
        # Subclasses can override this for proper train/val/test splits
        if split == "all" or split == "test":
            return self.data.to_dict('records')
        else:
            # Return empty for train/val if not implemented
            return []
    
    def build_prompt(self, instance: Dict[str, Any]) -> str:
        """Build a prompt for a given instance."""
        prompt = ""
        
        # Add demonstrations if available
        if self.demonstrations is not None:
            prompt += self._format_demonstrations()
            prompt += "\n\n"
        
        # Add the instance input
        if self.config.prompt_template:
            prompt += self.config.prompt_template.format(**instance)
        else:
            prompt += f"Input: {instance[self.config.input_column]}\nOutput:"
        
        return prompt
    
    def _format_demonstrations(self) -> str:
        """Format demonstration examples into a string."""
        if isinstance(self.demonstrations, str):
            return self.demonstrations
        elif isinstance(self.demonstrations, pd.DataFrame):
            demo_text = "Here are some examples:\n"
            for _, row in self.demonstrations.head(self.config.num_demonstrations).iterrows():
                demo_text += f"Input: {row[self.config.input_column]}\nOutput: {row[self.config.output_column]}\n\n"
            return demo_text.strip()
        elif isinstance(self.demonstrations, list):
            demo_text = "Here are some examples:\n"
            for demo in self.demonstrations[:self.config.num_demonstrations]:
                demo_text += f"Input: {demo[self.config.input_column]}\nOutput: {demo[self.config.output_column]}\n\n"
            return demo_text.strip()
        elif isinstance(self.demonstrations, dict):
            # Handle category-based demonstrations (like in the notebook)
            return ""  # Will be handled by subclasses that need category-specific demos
        else:
            return ""
    
    @abstractmethod
    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """Evaluate predictions against ground truth."""
        pass
    
    def preprocess_prediction(self, prediction: str) -> str:
        """Preprocess a model prediction before evaluation."""
        # Default implementation: strip whitespace and take first line
        return prediction.strip().split('\n')[0] if prediction else ""
    
    def get_ground_truth(self, split: str = "test") -> List[str]:
        """Get ground truth answers for a split."""
        data = self.get_split(split)
        return [item[self.config.output_column] for item in data]


class SimpleTask(BaseTask):
    """A simple task implementation with basic exact match evaluation."""
    
    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """Evaluate predictions with exact match accuracy."""
        ground_truth = self.get_ground_truth(split)
        
        if len(predictions) != len(ground_truth):
            raise ValueError(f"Prediction count ({len(predictions)}) doesn't match ground truth count ({len(ground_truth)})")
        
        # Preprocess predictions
        processed_predictions = [self.preprocess_prediction(pred) for pred in predictions]
        
        # Calculate exact match accuracy
        correct = sum(1 for pred, gt in zip(processed_predictions, ground_truth) 
                     if pred.lower().strip() == gt.lower().strip())
        accuracy = correct / len(ground_truth)
        
        results = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(ground_truth)
        }
        
        # Add per-category accuracy if available
        data = self.get_split(split)
        if any("category" in item for item in data):
            category_results = {}
            for pred, gt, item in zip(processed_predictions, ground_truth, data):
                category = item.get("category", "unknown")
                if category not in category_results:
                    category_results[category] = {"correct": 0, "total": 0}
                
                category_results[category]["total"] += 1
                if pred.lower().strip() == gt.lower().strip():
                    category_results[category]["correct"] += 1
            
            # Calculate per-category accuracy
            for category, stats in category_results.items():
                results[f"accuracy_{category}"] = stats["correct"] / stats["total"]
        
        return results


# Factory function for creating tasks from config files
def create_task_from_config(config_path: str, task_class: type = SimpleTask) -> BaseTask:
    """Create a task instance from a configuration file."""
    config_path = Path(config_path)
    
    if config_path.suffix == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    config = TaskConfig(**config_dict)
    return task_class(config)


# Factory function for creating tasks from hardcoded data (for migration)
def create_task_from_data(
    name: str,
    data: List[Dict[str, Any]],
    demonstrations: Optional[Dict[str, List[str]]] = None,
    description: str = "Task created from hardcoded data",
    task_class: type = SimpleTask,
    **kwargs
) -> BaseTask:
    """Create a task instance from in-memory data."""
    config = TaskConfig(
        name=name,
        description=description,
        data_format="memory",
        in_memory_data=data,
        in_memory_demonstrations=demonstrations,
        **kwargs
    )
    return task_class(config)


# Utility function to convert notebook-style data to standardized format
def convert_notebook_data_to_standard(
    csv_path: str,
    category_demonstrations: Dict[str, List[str]],
    output_csv_path: Optional[str] = None,
    output_demo_json_path: Optional[str] = None
) -> tuple[str, str]:
    """
    Convert notebook-style hardcoded data to standardized CSV and JSON files.
    
    Args:
        csv_path: Path to the existing CSV data file
        category_demonstrations: Dictionary mapping categories to demonstration examples
        output_csv_path: Path for output CSV (defaults to input path)
        output_demo_json_path: Path for demonstration JSON file
        
    Returns:
        Tuple of (csv_path, demo_json_path)
    """
    import pandas as pd
    import json
    from pathlib import Path
    
    # Read the CSV data
    data = pd.read_csv(csv_path)
    
    # Set default output paths
    if output_csv_path is None:
        output_csv_path = csv_path
    if output_demo_json_path is None:
        base_path = Path(csv_path).parent
        output_demo_json_path = base_path / f"{Path(csv_path).stem}_demonstrations.json"
    
    # Save demonstrations to JSON
    with open(output_demo_json_path, 'w', encoding='utf-8') as f:
        json.dump(category_demonstrations, f, indent=2, ensure_ascii=False)
    
    # Ensure CSV has the right format (it likely already does)
    if output_csv_path != csv_path:
        data.to_csv(output_csv_path, index=False)
    
    print(f"Converted data saved to:")
    print(f"  CSV: {output_csv_path}")
    print(f"  Demonstrations: {output_demo_json_path}")
    
    return str(output_csv_path), str(output_demo_json_path)

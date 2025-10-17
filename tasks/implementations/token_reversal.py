"""Simple token reversal task — reverse the characters of input words."""

from typing import Dict, List, Any, Optional
import pandas as pd

from tasks.base_task import BaseTask, TaskConfig


class TokenReversalTask(BaseTask):
    """Task for reversing input words or strings."""
    TASK_NAME = "token_reversal"  # Auto-discovery key

    def __init__(self, config: TaskConfig):
        super().__init__(config)

    def _load_data(self):
        """
        Load data using BaseTask logic if available (in-memory or from file).
        If nothing is loaded (no data_path and no in_memory_data), fall back to defaults.
        """
        # Try BaseTask's loader first (handles in_memory_data / file paths if provided)
        try:
            super()._load_data()
        except Exception:
            # If BaseTask loader raises due to missing files etc., we still fall back to defaults
            self.data = None

        if self.data is not None:
            return

        # Default token reversal examples (~20 examples)
        default_examples: List[Dict[str, str]] = [
            {"input": "cat", "output": "tac"},
            {"input": "apple", "output": "elppa"},
            {"input": "mirror", "output": "rorrim"},
            {"input": "desk", "output": "ksed"},
            {"input": "light", "output": "thgil"},
            {"input": "blue", "output": "eulb"},
            {"input": "forest", "output": "tserof"},
            {"input": "dream", "output": "maerd"},
            {"input": "stone", "output": "enots"},
            {"input": "house", "output": "esuoh"},
            {"input": "river", "output": "revir"},
            {"input": "garden", "output": "nedrag"},
            {"input": "planet", "output": "tenalp"},
            {"input": "rocket", "output": "tekcor"},
            {"input": "orange", "output": "egnaro"},
            {"input": "bottle", "output": "elttob"},
            {"input": "window", "output": "wodniw"},
            {"input": "silver", "output": "revils"},
            {"input": "guitar", "output": "ratiug"},
            {"input": "candle", "output": "eldnac"},
        ]
        self.data = pd.DataFrame(default_examples)

    # Use BaseTask.get_split and BaseTask.build_prompt as-is.

    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """Evaluate predictions with exact match accuracy (case-insensitive, trimmed)."""
        ground_truth = self.get_ground_truth(split)

        if len(predictions) != len(ground_truth):
            return {
                "accuracy": 0.0,
                "error": f"Prediction count ({len(predictions)}) does not match ground truth count ({len(ground_truth)})",
            }

        processed_predictions = [self.preprocess_prediction(p) for p in predictions]
        correct = sum(
            1 for pred, gt in zip(processed_predictions, ground_truth)
            if pred.lower().strip() == str(gt).lower().strip()
        )
        accuracy = correct / len(ground_truth) if ground_truth else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(ground_truth),
        }


def create_token_reversal_task(
    examples: Optional[List[Dict[str, str]]] = None,
    name: str = "token_reversal"
) -> TokenReversalTask:
    """
    Create a TokenReversalTask instance.

    Uses default columns `input` and `output` to match BaseTask defaults.
    Provides a prompt_template so BaseTask.build_prompt can be used directly.
    """
    config = TaskConfig(
        name=name,
        description="Token reversal evaluation task",
        data_format="memory" if examples is not None else "csv",  # 'memory' if examples provided; else BaseTask may load from file if configured later
        in_memory_data=examples,
        input_column="input",
        output_column="output",
        prompt_template="Reverse the characters in the given word.\nInput: {input}\nOutput:",
        evaluation_metrics=["accuracy"],
        metadata={"task_type": "string_transformation"},
    )
    return TokenReversalTask(config)

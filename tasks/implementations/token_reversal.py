"""Simple token reversal task — reverse the characters of input words."""

from typing import Dict, List, Any, Optional
import pandas as pd

from tasks.base_task import BaseTask, TaskConfig


class TokenReversalTask(BaseTask):
    """Task for reversing input words or strings."""
    TASK_NAME = "token_reversal"  # Auto-discovery key

    def __init__(self, config: TaskConfig):
        super().__init__(config)

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
    name: str = "token_reversal",
) -> TokenReversalTask:
    """
    Create a TokenReversalTask instance.

    Since this is an ICL task, we pass examples directly as in-memory data and rely on BaseTask.
    """
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

    config = TaskConfig(
        name=name,
        description="Token reversal evaluation task",
        data_format="memory",
        in_memory_data=examples if examples is not None else default_examples,
        input_column="input",
        output_column="output",
        prompt_template="Input: {input}\nOutput:",
        evaluation_metrics=["accuracy"],
        metadata={"task_type": "string_transformation"},
    )
    return TokenReversalTask(config)
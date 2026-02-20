"""Multi-step arithmetic — chained computation primitive.

Tests the model's ability to perform sequential arithmetic operations,
a prerequisite for GSM8k-style math reasoning.

Categories:
  - two_step:    "3 + 4, then multiply by 2" → "14"
  - three_step:  "Start with 10, subtract 3, then multiply by 4" → "28"

All operands and results are small positive integers (≤200) to keep the
task about *chaining* rather than large-number arithmetic.

Format (ICL):
    Input: 3 + 4, then multiply by 2
    Output: 14
"""

import random
from typing import Dict, List, Any

from tasks.base_task import BaseTask, TaskConfig


class MultistepArithmeticTask(BaseTask):
    """Multi-step (chained) arithmetic task."""

    TASK_NAME = "multistep_arithmetic"

    CATEGORY_DATA: Dict[str, List[Dict[str, str]]] = {
        "two_step": [
            {"input": "3 + 4, then multiply by 2", "output": "14"},
            {"input": "10 - 3, then add 5", "output": "12"},
            {"input": "6 * 3, then subtract 8", "output": "10"},
            {"input": "20 / 4, then add 7", "output": "12"},
            {"input": "8 + 7, then divide by 3", "output": "5"},
            {"input": "5 * 5, then subtract 10", "output": "15"},
            {"input": "9 - 2, then multiply by 4", "output": "28"},
            {"input": "12 / 3, then add 11", "output": "15"},
            {"input": "7 + 8, then subtract 6", "output": "9"},
            {"input": "4 * 6, then divide by 8", "output": "3"},
            {"input": "15 - 9, then multiply by 5", "output": "30"},
            {"input": "18 / 6, then add 9", "output": "12"},
            {"input": "2 + 13, then divide by 5", "output": "3"},
            {"input": "11 * 2, then subtract 14", "output": "8"},
            {"input": "16 - 7, then multiply by 3", "output": "27"},
            {"input": "30 / 5, then add 4", "output": "10"},
            {"input": "8 + 2, then multiply by 7", "output": "70"},
            {"input": "14 - 6, then divide by 2", "output": "4"},
            {"input": "3 * 9, then subtract 17", "output": "10"},
            {"input": "24 / 8, then add 6", "output": "9"},
        ],

        "three_step": [
            {"input": "Start with 10, subtract 3, then multiply by 4", "output": "28"},
            {"input": "Start with 2, add 6, then divide by 4", "output": "2"},
            {"input": "Start with 5, multiply by 3, then subtract 7", "output": "8"},
            {"input": "Start with 20, divide by 5, then add 8", "output": "12"},
            {"input": "Start with 3, add 9, then multiply by 2", "output": "24"},
            {"input": "Start with 18, subtract 6, then divide by 3", "output": "4"},
            {"input": "Start with 4, multiply by 5, then subtract 11", "output": "9"},
            {"input": "Start with 15, divide by 3, then add 10", "output": "15"},
            {"input": "Start with 7, add 5, then divide by 6", "output": "2"},
            {"input": "Start with 8, multiply by 4, then subtract 20", "output": "12"},
            {"input": "Start with 30, divide by 6, then multiply by 3", "output": "15"},
            {"input": "Start with 1, add 11, then divide by 4", "output": "3"},
            {"input": "Start with 9, subtract 4, then multiply by 6", "output": "30"},
            {"input": "Start with 16, divide by 2, then add 7", "output": "15"},
            {"input": "Start with 6, multiply by 7, then subtract 32", "output": "10"},
            {"input": "Start with 25, subtract 10, then divide by 5", "output": "3"},
            {"input": "Start with 11, add 4, then multiply by 2", "output": "30"},
            {"input": "Start with 36, divide by 9, then add 5", "output": "9"},
            {"input": "Start with 3, multiply by 8, then subtract 14", "output": "10"},
            {"input": "Start with 14, subtract 8, then multiply by 5", "output": "30"},
        ],
    }

    CATEGORY_DEMOS: Dict[str, List[str]] = {
        "two_step": [
            "Input: 5 + 3, then multiply by 4\nOutput: 32",
            "Input: 12 - 4, then add 3\nOutput: 11",
            "Input: 7 * 2, then subtract 5\nOutput: 9",
            "Input: 20 / 5, then add 2\nOutput: 6",
            "Input: 9 + 1, then divide by 2\nOutput: 5",
        ],
        "three_step": [
            "Input: Start with 6, add 4, then multiply by 3\nOutput: 30",
            "Input: Start with 12, subtract 2, then divide by 5\nOutput: 2",
            "Input: Start with 4, multiply by 3, then add 8\nOutput: 20",
            "Input: Start with 24, divide by 6, then subtract 1\nOutput: 3",
            "Input: Start with 7, add 3, then divide by 2\nOutput: 5",
        ],
    }

    def __init__(self, config: TaskConfig):
        super().__init__(config)

    def _load_data(self):
        import pandas as pd

        if self.config.in_memory_data:
            self.data = pd.DataFrame(self.config.in_memory_data)
            return

        rows: List[Dict[str, str]] = []
        for category, examples in self.CATEGORY_DATA.items():
            for ex in examples:
                rows.append({
                    "input": ex["input"],
                    "output": ex["output"],
                    "category_name": category,
                })
        self.data = pd.DataFrame(rows)

    def build_prompt(self, instance: Dict[str, Any], num_shots: int = 5) -> str:
        category = instance.get("category_name", "two_step")
        demos = self.CATEGORY_DEMOS.get(category, [])

        inst_input = instance.get("input", "")
        demos = [d for d in demos if inst_input not in d][:num_shots]

        prompt = ""
        for d in demos:
            prompt += d + "\n\n"
        prompt += f"Input: {inst_input}\nOutput:"
        return prompt

    def _extract_number(self, text: str) -> str:
        """Extract the first number (integer) from model output."""
        import re
        text = text.strip()
        match = re.search(r'-?\d+', text)
        return match.group() if match else text

    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        ground_truth = self.get_ground_truth(split)
        task_data = self.get_split(split)

        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Prediction count ({len(predictions)}) != ground truth ({len(ground_truth)})"
            )

        processed = [self._extract_number(self.preprocess_prediction(p)) for p in predictions]

        correct = sum(
            1 for p, g in zip(processed, ground_truth)
            if p.strip() == g.strip()
        )
        results: Dict[str, Any] = {
            "accuracy": correct / len(ground_truth),
            "correct": correct,
            "total": len(ground_truth),
        }

        cat_stats: Dict[str, Dict[str, int]] = {}
        for p, g, item in zip(processed, ground_truth, task_data):
            cat = item.get("category_name", "unknown")
            cat_stats.setdefault(cat, {"correct": 0, "total": 0})
            cat_stats[cat]["total"] += 1
            if p.strip() == g.strip():
                cat_stats[cat]["correct"] += 1

        for cat, s in cat_stats.items():
            results[f"accuracy_{cat}"] = s["correct"] / s["total"]

        return results

    def get_ground_truth(self, split: str = "test") -> List[str]:
        rows = self.get_split(split)
        return [str(r.get("output", "")) for r in rows]


def create_multistep_arithmetic_task(
    category: str = None,
    name: str = "multistep_arithmetic",
) -> MultistepArithmeticTask:
    """Create a MultistepArithmeticTask, optionally filtered to one category."""
    data = None
    if category and category in MultistepArithmeticTask.CATEGORY_DATA:
        data = [
            {**ex, "category_name": category}
            for ex in MultistepArithmeticTask.CATEGORY_DATA[category]
        ]
        name = f"multistep_arithmetic:{category}"

    config = TaskConfig(
        name=name,
        description="Multi-step chained arithmetic",
        data_format="memory",
        in_memory_data=data,
        input_column="input",
        output_column="output",
        evaluation_metrics=["accuracy"],
        metadata={"task_type": "multistep_arithmetic", "category": category},
    )
    return MultistepArithmeticTask(config)

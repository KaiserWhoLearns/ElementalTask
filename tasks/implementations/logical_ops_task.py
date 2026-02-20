"""Logical operators — elemental reasoning primitives.

Tests the model's capacity for basic logical operations:
  - negation:     Negate an adjective / statement ("happy" → "not happy")
  - conjunction:  Evaluate AND of truth values ("True AND False" → "False")
  - conditional:  Modus ponens — given a rule and a fact, derive the conclusion

These are building blocks for reasoning-heavy benchmarks.

Format (ICL):
    Input: happy
    Output: not happy
    (varies by category — see demos)
"""

import random
from typing import Dict, List, Any

from tasks.base_task import BaseTask, TaskConfig


class LogicalOpsTask(BaseTask):
    """Elemental logical-operators task."""

    TASK_NAME = "logical_ops"

    CATEGORY_DATA: Dict[str, List[Dict[str, str]]] = {
        "negation": [
            {"input": "happy", "output": "not happy"},
            {"input": "tall", "output": "not tall"},
            {"input": "fast", "output": "not fast"},
            {"input": "warm", "output": "not warm"},
            {"input": "light", "output": "not light"},
            {"input": "clean", "output": "not clean"},
            {"input": "loud", "output": "not loud"},
            {"input": "sharp", "output": "not sharp"},
            {"input": "thick", "output": "not thick"},
            {"input": "smooth", "output": "not smooth"},
            {"input": "dry", "output": "not dry"},
            {"input": "heavy", "output": "not heavy"},
            {"input": "bright", "output": "not bright"},
            {"input": "deep", "output": "not deep"},
            {"input": "wide", "output": "not wide"},
            {"input": "soft", "output": "not soft"},
            {"input": "new", "output": "not new"},
            {"input": "full", "output": "not full"},
            {"input": "rich", "output": "not rich"},
            {"input": "safe", "output": "not safe"},
        ],

        "conjunction": [
            {"input": "True AND True", "output": "True"},
            {"input": "True AND False", "output": "False"},
            {"input": "False AND True", "output": "False"},
            {"input": "False AND False", "output": "False"},
            {"input": "True OR True", "output": "True"},
            {"input": "True OR False", "output": "True"},
            {"input": "False OR True", "output": "True"},
            {"input": "False OR False", "output": "False"},
            {"input": "NOT True", "output": "False"},
            {"input": "NOT False", "output": "True"},
            {"input": "True AND True AND True", "output": "True"},
            {"input": "True AND True AND False", "output": "False"},
            {"input": "False OR False OR True", "output": "True"},
            {"input": "False OR False OR False", "output": "False"},
            {"input": "True AND (True OR False)", "output": "True"},
            {"input": "False AND (True OR True)", "output": "False"},
            {"input": "(True OR False) AND True", "output": "True"},
            {"input": "(False AND True) OR True", "output": "True"},
            {"input": "(False AND False) OR False", "output": "False"},
            {"input": "NOT (True AND False)", "output": "True"},
        ],

        "conditional": [
            {"input": 'Rule: "If it rains, the ground gets wet."\nFact: "It rains."', "output": "The ground gets wet."},
            {"input": 'Rule: "If the light is green, cars may go."\nFact: "The light is green."', "output": "Cars may go."},
            {"input": 'Rule: "If the temperature drops below 0, water freezes."\nFact: "The temperature drops below 0."', "output": "Water freezes."},
            {"input": 'Rule: "If you study hard, you pass the exam."\nFact: "You study hard."', "output": "You pass the exam."},
            {"input": 'Rule: "If the alarm rings, everyone evacuates."\nFact: "The alarm rings."', "output": "Everyone evacuates."},
            {"input": 'Rule: "If the power goes out, the lights turn off."\nFact: "The power goes out."', "output": "The lights turn off."},
            {"input": 'Rule: "If a number is even, it is divisible by 2."\nFact: "The number is even."', "output": "It is divisible by 2."},
            {"input": 'Rule: "If the store closes, no one can buy anything."\nFact: "The store closes."', "output": "No one can buy anything."},
            {"input": 'Rule: "If the dog barks, the cat hides."\nFact: "The dog barks."', "output": "The cat hides."},
            {"input": 'Rule: "If it snows, schools close."\nFact: "It snows."', "output": "Schools close."},
            {"input": 'Rule: "If you press the button, the door opens."\nFact: "You press the button."', "output": "The door opens."},
            {"input": 'Rule: "If the battery dies, the phone turns off."\nFact: "The battery dies."', "output": "The phone turns off."},
            {"input": 'Rule: "If the bridge is closed, drivers must detour."\nFact: "The bridge is closed."', "output": "Drivers must detour."},
            {"input": 'Rule: "If the cake is done, the timer beeps."\nFact: "The cake is done."', "output": "The timer beeps."},
            {"input": 'Rule: "If the sun sets, it gets dark."\nFact: "The sun sets."', "output": "It gets dark."},
            {"input": 'Rule: "If the key fits, the lock opens."\nFact: "The key fits."', "output": "The lock opens."},
            {"input": 'Rule: "If you water the plant, it grows."\nFact: "You water the plant."', "output": "It grows."},
            {"input": 'Rule: "If the wind blows, the leaves scatter."\nFact: "The wind blows."', "output": "The leaves scatter."},
            {"input": 'Rule: "If demand increases, prices rise."\nFact: "Demand increases."', "output": "Prices rise."},
            {"input": 'Rule: "If the code compiles, the program runs."\nFact: "The code compiles."', "output": "The program runs."},
        ],
    }

    CATEGORY_DEMOS: Dict[str, List[str]] = {
        "negation": [
            "Input: cold\nOutput: not cold",
            "Input: old\nOutput: not old",
            "Input: slow\nOutput: not slow",
            "Input: dark\nOutput: not dark",
            "Input: quiet\nOutput: not quiet",
        ],
        "conjunction": [
            "Input: True AND True\nOutput: True",
            "Input: True AND False\nOutput: False",
            "Input: False OR True\nOutput: True",
            "Input: NOT True\nOutput: False",
            "Input: False OR False\nOutput: False",
        ],
        "conditional": [
            'Input: Rule: "If the bell rings, class starts."\nFact: "The bell rings."\nOutput: Class starts.',
            'Input: Rule: "If traffic is heavy, arrive early."\nFact: "Traffic is heavy."\nOutput: Arrive early.',
            'Input: Rule: "If you eat too much, you feel sick."\nFact: "You eat too much."\nOutput: You feel sick.',
            'Input: Rule: "If the match is lit, it burns."\nFact: "The match is lit."\nOutput: It burns.',
            'Input: Rule: "If the door is locked, use the key."\nFact: "The door is locked."\nOutput: Use the key.',
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
        category = instance.get("category_name", "negation")
        demos = self.CATEGORY_DEMOS.get(category, [])

        inst_input = instance.get("input", "")
        demos = [d for d in demos if inst_input not in d][:num_shots]

        prompt = ""
        for d in demos:
            prompt += d + "\n\n"

        if category == "conditional":
            prompt += f"Input: {inst_input}\nOutput:"
        else:
            prompt += f"Input: {inst_input}\nOutput:"
        return prompt

    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        ground_truth = self.get_ground_truth(split)
        task_data = self.get_split(split)

        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Prediction count ({len(predictions)}) != ground truth ({len(ground_truth)})"
            )

        processed = [self.preprocess_prediction(p) for p in predictions]

        correct = sum(
            1 for p, g in zip(processed, ground_truth)
            if p.lower().strip() == g.lower().strip()
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
            if p.lower().strip() == g.lower().strip():
                cat_stats[cat]["correct"] += 1

        for cat, s in cat_stats.items():
            results[f"accuracy_{cat}"] = s["correct"] / s["total"]

        return results

    def get_ground_truth(self, split: str = "test") -> List[str]:
        rows = self.get_split(split)
        return [str(r.get("output", "")) for r in rows]


def create_logical_ops_task(
    category: str = None,
    name: str = "logical_ops",
) -> LogicalOpsTask:
    """Create a LogicalOpsTask, optionally filtered to one category."""
    data = None
    if category and category in LogicalOpsTask.CATEGORY_DATA:
        data = [
            {**ex, "category_name": category}
            for ex in LogicalOpsTask.CATEGORY_DATA[category]
        ]
        name = f"logical_ops:{category}"

    config = TaskConfig(
        name=name,
        description="Logical operators (negation, conjunction, conditional)",
        data_format="memory",
        in_memory_data=data,
        input_column="input",
        output_column="output",
        evaluation_metrics=["accuracy"],
        metadata={"task_type": "logical_ops", "category": category},
    )
    return LogicalOpsTask(config)

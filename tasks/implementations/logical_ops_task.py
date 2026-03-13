"""Logical operators — elemental reasoning primitives.

Tests the model's capacity for basic logical operations in text form:
    - negation:     Verify whether a candidate is a correct logical negation
    - conjunction:  Evaluate conjunction/disjunction statements from text facts
    - conditional:  Decide whether a conclusion follows from a rule + fact

These are building blocks for reasoning-heavy benchmarks.

Format (ICL):
    Input: [text logic problem]
    Output: True or False
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
            {"input": "Statement: All robots can move.\nCandidate negation: Some robots cannot move.\nIs the candidate a correct logical negation?", "output": "True"},
            {"input": "Statement: All apples are red.\nCandidate negation: No apples are red.\nIs the candidate a correct logical negation?", "output": "False"},
            {"input": "Statement: Some birds can swim.\nCandidate negation: No birds can swim.\nIs the candidate a correct logical negation?", "output": "True"},
            {"input": "Statement: Some books are heavy.\nCandidate negation: Some books are not heavy.\nIs the candidate a correct logical negation?", "output": "False"},
            {"input": "Statement: The lamp is on.\nCandidate negation: The lamp is not on.\nIs the candidate a correct logical negation?", "output": "True"},
            {"input": "Statement: The lamp is on.\nCandidate negation: The lamp is off and broken.\nIs the candidate a correct logical negation?", "output": "False"},
            {"input": "Statement: Everyone passed the test.\nCandidate negation: Not everyone passed the test.\nIs the candidate a correct logical negation?", "output": "True"},
            {"input": "Statement: Everyone passed the test.\nCandidate negation: No one passed the test.\nIs the candidate a correct logical negation?", "output": "False"},
            {"input": "Statement: No cars are electric.\nCandidate negation: Some cars are electric.\nIs the candidate a correct logical negation?", "output": "True"},
            {"input": "Statement: No cars are electric.\nCandidate negation: All cars are electric.\nIs the candidate a correct logical negation?", "output": "False"},
            {"input": "Statement: The gate is closed.\nCandidate negation: The gate is open.\nIs the candidate a correct logical negation?", "output": "False"},
            {"input": "Statement: The gate is closed.\nCandidate negation: The gate is not closed.\nIs the candidate a correct logical negation?", "output": "True"},
        ],

        "conjunction": [
            {"input": "Fact A is True. Fact B is True.\nClaim: A AND B.\nIs the claim true?", "output": "True"},
            {"input": "Fact A is True. Fact B is False.\nClaim: A AND B.\nIs the claim true?", "output": "False"},
            {"input": "Fact A is False. Fact B is True.\nClaim: A OR B.\nIs the claim true?", "output": "True"},
            {"input": "Fact A is False. Fact B is False.\nClaim: A OR B.\nIs the claim true?", "output": "False"},
            {"input": "Fact A is True. Fact B is False.\nClaim: NOT B.\nIs the claim true?", "output": "True"},
            {"input": "Fact A is True. Fact B is False.\nClaim: NOT A.\nIs the claim true?", "output": "False"},
            {"input": "Fact A is True. Fact B is True. Fact C is False.\nClaim: A AND B AND C.\nIs the claim true?", "output": "False"},
            {"input": "Fact A is True. Fact B is True. Fact C is False.\nClaim: A AND (B OR C).\nIs the claim true?", "output": "True"},
            {"input": "Fact A is False. Fact B is False. Fact C is True.\nClaim: (A OR B) AND C.\nIs the claim true?", "output": "False"},
            {"input": "Fact A is False. Fact B is False. Fact C is True.\nClaim: A OR (B OR C).\nIs the claim true?", "output": "True"},
            {"input": "Fact A is True. Fact B is False.\nClaim: (A AND B) OR A.\nIs the claim true?", "output": "True"},
            {"input": "Fact A is True. Fact B is False.\nClaim: (A OR B) AND B.\nIs the claim true?", "output": "False"},
        ],

        "conditional": [
            {"input": 'Rule: If it rains, the ground gets wet.\nFact: It rains.\nConclusion: The ground gets wet.\nDoes the conclusion logically follow?', "output": "True"},
            {"input": 'Rule: If it rains, the ground gets wet.\nFact: It does not rain.\nConclusion: The ground gets wet.\nDoes the conclusion logically follow?', "output": "False"},
            {"input": 'Rule: If the switch is on, the lamp lights up.\nFact: The switch is on.\nConclusion: The lamp lights up.\nDoes the conclusion logically follow?', "output": "True"},
            {"input": 'Rule: If the switch is on, the lamp lights up.\nFact: The lamp lights up.\nConclusion: The switch is on.\nDoes the conclusion logically follow?', "output": "False"},
            {"input": 'Rule: If a number is even, it is divisible by 2.\nFact: 14 is even.\nConclusion: 14 is divisible by 2.\nDoes the conclusion logically follow?', "output": "True"},
            {"input": 'Rule: If a number is even, it is divisible by 2.\nFact: 15 is not even.\nConclusion: 15 is not divisible by 2.\nDoes the conclusion logically follow?', "output": "False"},
            {"input": 'Rule: If a file is deleted, it is unavailable.\nFact: The file is deleted.\nConclusion: The file is unavailable.\nDoes the conclusion logically follow?', "output": "True"},
            {"input": 'Rule: If a file is deleted, it is unavailable.\nFact: The file is unavailable.\nConclusion: The file is deleted.\nDoes the conclusion logically follow?', "output": "False"},
            {"input": 'Rule: If a student studies, they pass this quiz.\nFact: Mina studies.\nConclusion: Mina passes this quiz.\nDoes the conclusion logically follow?', "output": "True"},
            {"input": 'Rule: If a student studies, they pass this quiz.\nFact: Leo does not pass this quiz.\nConclusion: Leo did not study.\nDoes the conclusion logically follow?', "output": "False"},
            {"input": 'Rule: If the alarm rings, everyone evacuates.\nFact: The alarm rings.\nConclusion: Everyone evacuates.\nDoes the conclusion logically follow?', "output": "True"},
            {"input": 'Rule: If the alarm rings, everyone evacuates.\nFact: Everyone evacuates.\nConclusion: The alarm rings.\nDoes the conclusion logically follow?', "output": "False"},
        ],
    }

    CATEGORY_DEMOS: Dict[str, List[str]] = {
        "negation": [
            "Input: Statement: All chairs are wooden.\nCandidate negation: Some chairs are not wooden.\nIs the candidate a correct logical negation?\nOutput: True",
            "Input: Statement: Some cups are clean.\nCandidate negation: No cups are clean.\nIs the candidate a correct logical negation?\nOutput: True",
            "Input: Statement: The window is open.\nCandidate negation: The window is not open.\nIs the candidate a correct logical negation?\nOutput: True",
            "Input: Statement: Everyone arrived early.\nCandidate negation: No one arrived early.\nIs the candidate a correct logical negation?\nOutput: False",
            "Input: Statement: The road is clear.\nCandidate negation: The road is blocked.\nIs the candidate a correct logical negation?\nOutput: False",
        ],
        "conjunction": [
            "Input: Fact A is True. Fact B is True.\nClaim: A AND B.\nIs the claim true?\nOutput: True",
            "Input: Fact A is True. Fact B is False.\nClaim: A AND B.\nIs the claim true?\nOutput: False",
            "Input: Fact A is False. Fact B is True.\nClaim: A OR B.\nIs the claim true?\nOutput: True",
            "Input: Fact A is True. Fact B is False.\nClaim: NOT B.\nIs the claim true?\nOutput: True",
            "Input: Fact A is False. Fact B is False.\nClaim: A OR B.\nIs the claim true?\nOutput: False",
        ],
        "conditional": [
            "Input: Rule: If the bell rings, class starts.\nFact: The bell rings.\nConclusion: Class starts.\nDoes the conclusion logically follow?\nOutput: True",
            "Input: Rule: If traffic is heavy, travel time increases.\nFact: Travel time increased.\nConclusion: Traffic was heavy.\nDoes the conclusion logically follow?\nOutput: False",
            "Input: Rule: If the key fits, the lock opens.\nFact: The key fits.\nConclusion: The lock opens.\nDoes the conclusion logically follow?\nOutput: True",
            "Input: Rule: If a file is saved, it can be reopened.\nFact: The file can be reopened.\nConclusion: The file was saved.\nDoes the conclusion logically follow?\nOutput: False",
            "Input: Rule: If the sensor triggers, the alarm sounds.\nFact: The sensor triggers.\nConclusion: The alarm sounds.\nDoes the conclusion logically follow?\nOutput: True",
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

"""BoolQ yes/no reading-comprehension benchmark task.

Dataset: google/boolq
Splits: train | validation  (no public test labels)

Format: a passage followed by a yes/no question.
  Answer: "yes" if instance["answer"] is True, else "no".
"""

from typing import Any, Dict, List

from datasets import load_dataset

from ..base_task import TaskConfig
from .base_benchmark import BaseBenchmarkTask, MCChoice


class BoolQTask(BaseBenchmarkTask):
    """BoolQ yes/no reading-comprehension questions."""

    TASK_NAME = "boolq"

    def __init__(
        self,
        eval_mode: str = "logprob",
        num_shots: int = 5,
        split: str = "validation",
    ):
        task_config = TaskConfig(
            name="boolq",
            description="BoolQ yes/no reading comprehension",
            data_format="memory",
        )
        super().__init__(task_config, eval_mode=eval_mode, num_shots=num_shots, split=split)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_hf_dataset(self):
        ds = load_dataset("google/boolq")

        eval_split = self.hf_split if self.hf_split in ds else "validation"
        self.data = [dict(row) for row in ds[eval_split]]
        self._few_shot_pool = [dict(row) for row in ds["train"]]

    # ------------------------------------------------------------------
    # Task interface
    # ------------------------------------------------------------------

    def get_choices(self, instance: Dict[str, Any]) -> List[MCChoice]:
        return [
            MCChoice(label="yes", text="Yes"),
            MCChoice(label="no", text="No"),
        ]

    def get_correct_label(self, instance: Dict[str, Any]) -> str:
        return "yes" if instance["answer"] else "no"

    def format_question(self, instance: Dict[str, Any]) -> str:
        return instance["question"]

    def format_context(self, instance: Dict[str, Any]) -> str:
        return f"Passage: {instance['passage']}"

    def _format_single(self, instance: Dict[str, Any], include_answer: bool = False) -> str:
        """Override for a passage-then-question layout."""
        passage = instance["passage"]
        question = instance["question"]
        answer_prefix = "Answer:"
        body = (
            f"Passage: {passage}\n"
            f"Question: {question} (yes or no)\n"
            f"{answer_prefix}"
        )
        if include_answer:
            return f"{body} {self.get_correct_label(instance)}"
        return body

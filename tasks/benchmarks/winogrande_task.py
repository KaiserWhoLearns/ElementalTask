"""WinoGrande coreference / commonsense benchmark task.

Dataset: allenai/winogrande
Configuration: "winogrande_xl" (largest, 40k training examples)

Splits: train | validation | test
  - test split has no labels on HuggingFace, so we use validation for evaluation
  and a portion of train for the few-shot pool.

Format: fill-in-the-blank sentence — the blank (_) must be resolved to one
of two options:
    "He carried the _ to his car because ..."  → option1 / option2

Standard eval: 0-shot (WinoGrande benchmark paper uses 0-shot), but we
default to 5-shot for consistency with the rest of the evaluation suite.
"""

from typing import Any, Dict, List

from datasets import load_dataset

from ..base_task import TaskConfig
from .base_benchmark import BaseBenchmarkTask, MCChoice


class WinograndeTask(BaseBenchmarkTask):
    """WinoGrande binary coreference resolution.

    Each instance presents a sentence with a blank (`_`) and two candidate
    fillers. The model must pick the correct one.

    Labels are "1" and "2" (matching the `answer` field in the dataset).
    """

    TASK_NAME = "winogrande"

    def __init__(
        self,
        winogrande_config: str = "winogrande_xl",
        eval_mode: str = "logprob",
        num_shots: int = 5,
        split: str = "validation",
    ):
        self.winogrande_config = winogrande_config
        task_config = TaskConfig(
            name="winogrande",
            description="WinoGrande binary coreference resolution",
            data_format="memory",
        )
        super().__init__(task_config, eval_mode=eval_mode, num_shots=num_shots, split=split)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_hf_dataset(self):
        ds = load_dataset("allenai/winogrande", self.winogrande_config)

        eval_split = self.hf_split if self.hf_split in ds else "validation"
        self.data = [dict(row) for row in ds[eval_split]]

        # Filter train rows that actually have labels (answer != "")
        self._few_shot_pool = [
            dict(row) for row in ds["train"]
            if str(row.get("answer", "")).strip() in ("1", "2")
        ]

    # ------------------------------------------------------------------
    # Task interface
    # ------------------------------------------------------------------

    def get_choices(self, instance: Dict[str, Any]) -> List[MCChoice]:
        return [
            MCChoice(label="1", text=instance["option1"]),
            MCChoice(label="2", text=instance["option2"]),
        ]

    def get_correct_label(self, instance: Dict[str, Any]) -> str:
        return str(instance["answer"]).strip()

    def format_question(self, instance: Dict[str, Any]) -> str:
        """The sentence itself is the question; blank shown as ___."""
        sent = instance["sentence"]
        opt1 = instance["option1"]
        opt2 = instance["option2"]
        # Show the fill-in-blank sentence and the two options
        return f"{sent}\nOptions: (1) {opt1}  (2) {opt2}"

    def _format_single(self, instance: Dict[str, Any], include_answer: bool = False) -> str:
        """Override to use a more natural fill-in-blank format."""
        sent = instance["sentence"]
        opt1 = instance["option1"]
        opt2 = instance["option2"]

        question = (
            f"Complete the sentence by choosing option 1 or 2.\n"
            f"Sentence: {sent}\n"
            f"(1) {opt1}\n"
            f"(2) {opt2}"
        )
        answer_prefix = "Answer:"
        if include_answer:
            return f"{question}\n{answer_prefix} {self.get_correct_label(instance)}"
        return f"{question}\n{answer_prefix}"

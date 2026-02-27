"""GSM8K grade-school math benchmark task.

Dataset: openai/gsm8k
Configuration: "main"
Splits: train | test

This is an open-ended (generative) task — the model must produce the numeric
answer. Log-prob mode is not applicable (no fixed choices), so eval_mode is
always "generative".

The gold answer is extracted from the `#### <number>` marker at the end of
the `answer` field, e.g.  "...So the total is 42. #### 42".
"""

import re
from typing import Any, Dict, List

from datasets import load_dataset

from ..base_task import TaskConfig
from .base_benchmark import BaseBenchmarkTask, MCChoice


class GSM8KTask(BaseBenchmarkTask):
    """GSM8K grade-school math reasoning (open-ended)."""

    TASK_NAME = "gsm8k"

    def __init__(
        self,
        eval_mode: str = "generative",   # always generative; logprob N/A
        num_shots: int = 8,              # standard 8-shot as in the GSM8K paper
        split: str = "test",
    ):
        if eval_mode != "generative":
            print(
                "Warning: GSM8K has no fixed answer choices; "
                "forcing eval_mode='generative'."
            )
            eval_mode = "generative"

        task_config = TaskConfig(
            name="gsm8k",
            description="GSM8K grade-school math reasoning",
            data_format="memory",
        )
        super().__init__(task_config, eval_mode=eval_mode, num_shots=num_shots, split=split)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_hf_dataset(self):
        ds = load_dataset("openai/gsm8k", "main")

        eval_split = self.hf_split if self.hf_split in ds else "test"
        self.data = [dict(row) for row in ds[eval_split]]
        self._few_shot_pool = [dict(row) for row in ds["train"]]

    # ------------------------------------------------------------------
    # Task interface
    # ------------------------------------------------------------------

    def get_choices(self, instance: Dict[str, Any]) -> List[MCChoice]:
        """GSM8K is open-ended — no fixed answer choices."""
        return []

    def get_correct_label(self, instance: Dict[str, Any]) -> str:
        """Extract the numeric answer that follows '####' in the answer field."""
        raw = instance.get("answer", "")
        m = re.search(r"####\s*(-?[\d,]+)", raw)
        if m:
            return m.group(1).replace(",", "")
        # Fallback: last number in the answer
        nums = re.findall(r"-?[\d,]+", raw)
        return nums[-1].replace(",", "") if nums else raw.strip()

    def format_question(self, instance: Dict[str, Any]) -> str:
        return instance["question"]

    def _format_single(self, instance: Dict[str, Any], include_answer: bool = False) -> str:
        """Format with chain-of-thought when the gold answer is available."""
        question = instance["question"]
        answer_prefix = "Answer:"
        if include_answer:
            # Include the full reasoning chain during few-shot
            full_answer = instance.get("answer", self.get_correct_label(instance))
            return f"Question: {question}\n{answer_prefix} {full_answer}"
        return f"Question: {question}\n{answer_prefix}"

    def normalize_prediction(self, text: str) -> str:
        """Extract the final numeric answer from model output."""
        text = text.strip()
        # Look for #### marker (chain-of-thought style)
        m = re.search(r"####\s*(-?[\d,]+)", text)
        if m:
            return m.group(1).replace(",", "")
        # Fall back to last number mentioned
        nums = re.findall(r"-?[\d,]+", text)
        return nums[-1].replace(",", "") if nums else text

    def check_answer(self, prediction: str, instance: Dict[str, Any]) -> bool:
        pred = self.normalize_prediction(prediction)
        gold = self.get_correct_label(instance)
        try:
            return float(pred.replace(",", "")) == float(gold.replace(",", ""))
        except ValueError:
            return pred.strip() == gold.strip()

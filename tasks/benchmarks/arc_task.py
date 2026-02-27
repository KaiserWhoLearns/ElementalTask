"""ARC (AI2 Reasoning Challenge) benchmark task.

Dataset: allenai/ai2_arc
Configurations:
  - "ARC-Easy"      (easier subset)
  - "ARC-Challenge" (harder subset — used in most papers)

Splits: train | validation | test
Standard eval: test split.
"""

from typing import Any, Dict, List

from datasets import load_dataset

from ..base_task import TaskConfig
from .base_benchmark import BaseBenchmarkTask, MCChoice


class ARCTask(BaseBenchmarkTask):
    """ARC multiple-choice science questions.

    Each instance has 3–5 lettered answer choices.
    Default eval on the 'test' split; few-shot pool from 'train'.
    """

    TASK_NAME = "arc"

    def __init__(
        self,
        config: str | TaskConfig = "ARC-Challenge",
        eval_mode: str = "logprob",
        num_shots: int = 25,
        split: str = "test",
    ):
        """
        Args:
            config: HuggingFace ARC configuration — "ARC-Easy" or "ARC-Challenge".
            eval_mode: "generative" or "logprob".
            num_shots: Number of ICL shots (default 25, matching the ARC paper).
            split: Dataset split to evaluate on.
        """
        if isinstance(config, str):
            self.arc_config = config
            task_config = TaskConfig(
                name=f"arc_{config.lower().replace('-', '_')}",
                description=f"ARC {config} multiple-choice science questions",
                data_format="memory",
            )
        else:
            # TaskConfig passed directly
            self.arc_config = "ARC-Challenge"
            task_config = config

        super().__init__(task_config, eval_mode=eval_mode, num_shots=num_shots, split=split)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_hf_dataset(self):
        ds = load_dataset("allenai/ai2_arc", self.arc_config)

        self.data = [dict(row) for row in ds[self.hf_split]]
        train_split = "train" if "train" in ds else self.hf_split
        self._few_shot_pool = [dict(row) for row in ds[train_split]]

    # ------------------------------------------------------------------
    # Task interface
    # ------------------------------------------------------------------

    def get_choices(self, instance: Dict[str, Any]) -> List[MCChoice]:
        labels = instance["choices"]["label"]
        texts = instance["choices"]["text"]
        return [MCChoice(label=str(l), text=str(t)) for l, t in zip(labels, texts)]

    def get_correct_label(self, instance: Dict[str, Any]) -> str:
        return str(instance["answerKey"]).strip()

    def format_question(self, instance: Dict[str, Any]) -> str:
        return instance["question"]

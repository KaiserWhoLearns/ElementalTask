"""Base class for multiple-choice and generative benchmark tasks.

Supports two evaluation modes:
  - "generative": prompt ends with "Answer:" and model generates a letter/word.
                  Answer is extracted from the first token(s) of the output.
  - "logprob":    each answer choice is scored by its log-probability given the
                  prompt; the argmax is taken as the prediction. Requires the
                  evaluator's compute_target_metrics() method.
"""

import re
import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

from ..base_task import BaseTask, TaskConfig


@dataclass
class MCChoice:
    """A single multiple-choice option."""
    label: str   # "A", "B", "C", "D", "1", "2", "yes", "no", …
    text: str    # the option text


class BaseBenchmarkTask(BaseTask):
    """Base class for real benchmark tasks (ARC, WinoGrande, BoolQ, GSM8K, …).

    Sub-classes must implement:
        get_choices(instance)     -> List[MCChoice]  (empty for open-ended tasks)
        get_correct_label(instance) -> str
        _load_hf_dataset()        -> populates self.data (list of dicts) and
                                      self._few_shot_pool (list of dicts for ICL)

    Optional overrides:
        format_question(instance) -> str  (default: instance["question"])
        format_context(instance)  -> str  (default: "")
        normalize_prediction(text) -> str (default: strip + lower first char)
    """

    # Labels used in generative prompts, e.g. "(A)", "(B)", …
    LETTER_LABELS = ["A", "B", "C", "D", "E"]

    def __init__(
        self,
        config: TaskConfig,
        eval_mode: str = "generative",
        num_shots: int = 5,
        split: str = "validation",
    ):
        """
        Args:
            config: TaskConfig (name, description, etc.)
            eval_mode: "generative" or "logprob"
            num_shots: default number of ICL shots in prompts
            split: HuggingFace dataset split for evaluation ("validation" / "test")
        """
        assert eval_mode in ("generative", "logprob"), \
            f"eval_mode must be 'generative' or 'logprob', got '{eval_mode}'"
        self.eval_mode = eval_mode
        self.num_shots = num_shots
        self.hf_split = split
        self._few_shot_pool: List[Dict[str, Any]] = []

        # BaseTask.__init__ calls _load_data(); we override that to load from HF.
        super().__init__(config)

    # ------------------------------------------------------------------
    # HuggingFace data loading (override _load_data from BaseTask)
    # ------------------------------------------------------------------

    def _load_data(self):
        """Load dataset from HuggingFace. Populates self.data and self._few_shot_pool."""
        try:
            self._load_hf_dataset()
        except Exception as e:
            print(f"Warning: Failed to load {self.config.name} from HuggingFace: {e}")
            print("  Install with: pip install datasets")
            self.data = []
            self._few_shot_pool = []

    @abstractmethod
    def _load_hf_dataset(self):
        """Load from HuggingFace datasets. Must set self.data and self._few_shot_pool.

        self.data          — list of dicts, evaluation split
        self._few_shot_pool — list of dicts, used for ICL examples (train split)
        """
        ...

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_choices(self, instance: Dict[str, Any]) -> List[MCChoice]:
        """Return the answer choices for this instance.

        For open-ended tasks (e.g. GSM8K) return an empty list.
        """
        ...

    @abstractmethod
    def get_correct_label(self, instance: Dict[str, Any]) -> str:
        """Return the correct label string (e.g. "A", "yes", "42")."""
        ...

    # ------------------------------------------------------------------
    # Formatting helpers (can be overridden)
    # ------------------------------------------------------------------

    def format_question(self, instance: Dict[str, Any]) -> str:
        """Return the question text. Override if the field name differs."""
        return instance.get("question", instance.get("sentence", ""))

    def format_context(self, instance: Dict[str, Any]) -> str:
        """Return optional context/passage. Empty by default."""
        return ""

    def format_choices_block(self, choices: List[MCChoice]) -> str:
        """Format choices as '(A) text\\n(B) text\\n…'"""
        return "\n".join(f"({c.label}) {c.text}" for c in choices)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _format_single(self, instance: Dict[str, Any], include_answer: bool = False) -> str:
        """Format one example (question + optional choices + optional answer)."""
        parts = []

        ctx = self.format_context(instance)
        if ctx:
            parts.append(ctx)

        parts.append(f"Question: {self.format_question(instance)}")

        choices = self.get_choices(instance)
        if choices:
            parts.append(self.format_choices_block(choices))

        answer_prefix = "Answer:"
        if include_answer:
            label = self.get_correct_label(instance)
            parts.append(f"{answer_prefix} {label}")
        else:
            parts.append(answer_prefix)

        return "\n".join(parts)

    def build_prompt(self, instance: Dict[str, Any], num_shots: Optional[int] = None) -> str:
        """Build a full ICL prompt for the given instance.

        Args:
            instance: evaluation example
            num_shots: number of ICL shots (default: self.num_shots)

        Returns:
            Prompt string ending with "Answer:" (ready for the model to complete)
        """
        k = num_shots if num_shots is not None else self.num_shots
        parts = []

        # Few-shot examples from the training pool
        import random
        pool = self._few_shot_pool
        shots = random.sample(pool, min(k, len(pool))) if pool else []
        for shot in shots:
            parts.append(self._format_single(shot, include_answer=True))

        # Test instance (no answer)
        parts.append(self._format_single(instance, include_answer=False))

        return "\n\n".join(parts)

    # Compatible with BaseTask.build_prompt(query, num_shots) signature used by
    # scripts/print_tasks_overview.py etc.
    def get_icl_examples(self, num_examples: int = 5, seed: int = 42, **kwargs) -> List[Dict[str, str]]:
        """Return ICL examples in BaseTask format (list of {input, output})."""
        import random
        rng = random.Random(seed)
        pool = self._few_shot_pool or (self.data or [])
        shots = rng.sample(pool, min(num_examples, len(pool)))
        return [
            {
                "input": self._format_single(s, include_answer=False),
                "output": self.get_correct_label(s),
            }
            for s in shots
        ]

    # ------------------------------------------------------------------
    # Answer extraction / evaluation
    # ------------------------------------------------------------------

    def normalize_prediction(self, text: str) -> str:
        """Extract the answer from raw model output (generative mode)."""
        text = text.strip()
        if not text:
            return ""

        # Check for explicit "(X)" or "X)" at the start
        m = re.match(r"^\(?\s*([A-Ea-e1-9])\s*\)?", text)
        if m:
            return m.group(1).upper()

        # For yes/no tasks
        lower = text.lower()
        if lower.startswith("yes"):
            return "yes"
        if lower.startswith("no"):
            return "no"

        # For numeric answers (GSM8K)
        m = re.search(r"####\s*(-?[\d,]+)", text)
        if m:
            return m.group(1).replace(",", "")
        m = re.search(r"(-?[\d,]+)", text)
        if m:
            return m.group(1).replace(",", "")

        return text[0].upper() if text else ""

    def check_answer(self, prediction: str, instance: Dict[str, Any]) -> bool:
        """Check whether a (generative) prediction matches the correct answer."""
        pred = self.normalize_prediction(prediction)
        gold = self.get_correct_label(instance).strip().upper()
        return pred.upper() == gold

    # ------------------------------------------------------------------
    # BaseTask abstract method implementation
    # ------------------------------------------------------------------

    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """Evaluate a list of predictions against ground truth for the given split.

        Args:
            predictions: model output strings (one per instance in the split)
            split: dataset split name (ignored; self.data is already the eval split)

        Returns:
            dict with at least {"accuracy": float}
        """
        instances = self.data or []
        if not instances:
            return {"accuracy": 0.0, "n": 0}

        n = min(len(predictions), len(instances))
        correct = sum(
            self.check_answer(pred, inst)
            for pred, inst in zip(predictions[:n], instances[:n])
        )
        return {"accuracy": correct / n if n else 0.0, "n": n}

    # ------------------------------------------------------------------
    # Log-prob scoring (used when eval_mode == "logprob")
    # ------------------------------------------------------------------

    def score_choices_logprob(
        self,
        evaluator,
        prompt: str,
        choices: List[MCChoice],
    ) -> Tuple[str, List[float]]:
        """Score each choice by log-probability and return the predicted label.

        Args:
            evaluator: TaskEvaluator instance (must have compute_target_metrics)
            prompt: the prompt string ending with "Answer:"
            choices: list of MCChoice objects

        Returns:
            (predicted_label, list_of_log_probs)
        """
        # Score each choice: log-prob of " {label}" or " {text}" appended to prompt
        targets = [f" {c.label}" for c in choices]
        metrics = evaluator.compute_target_metrics([prompt] * len(targets), targets)

        # Use negative loss (= log-prob) as score; higher = better
        scores = [-m.get("loss", float("inf")) for m in metrics]

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return choices[best_idx].label, scores

    # ------------------------------------------------------------------
    # BaseTask compatibility
    # ------------------------------------------------------------------

    def get_split(self, split: str = "test") -> List[Dict[str, Any]]:
        if self.data is None:
            return []
        return list(self.data) if split in ("test", "all") else []

"""POS identification task — fill in the part of speech for a target word in a sentence."""

from typing import Dict, List, Any, Optional
from copy import deepcopy
import pandas as pd

from tasks.base_task import BaseTask, TaskConfig


class PartOfSpeechTask(BaseTask):
    """Task for identifying the part of speech of a target word in a sentence."""
    TASK_NAME = "part_of_speech"  # Auto-discovery key

    def __init__(self, config: TaskConfig):
        super().__init__(config)

    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """Evaluate predictions with exact-match accuracy (case-insensitive, trimmed)."""
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

        return {"accuracy": accuracy, "correct": correct, "total": len(ground_truth)}


def create_pos_task(
    dataset: Optional[List[Dict[str, str]]] = None,
    name: str = "part_of_speech",
    prompt_format: str = '{sentence} The part of speech for "{target_word}" is _',
) -> PartOfSpeechTask:
    """
    Create a PartOfSpeechTask instance.

    Expects items shaped like:
      {"sentence": str, "target_word": str, "answer": str}
    Optionally items may include a pre-built "prompt"; otherwise it's derived from `prompt_format`.

    Since this is an ICL task, we pass examples directly as in-memory data and rely on BaseTask.
    """

    # --- Default: 15 examples (from the conversation) ---
    default_examples: List[Dict[str, str]] = [
        {"sentence": "The cat is in the house.", "target_word": "cat", "answer": "noun"},
        {"sentence": "Alex arrived before noon.", "target_word": "Alex", "answer": "proper noun"},
        {"sentence": "They run every morning.", "target_word": "run", "answer": "verb"},
        {"sentence": "The happy child waved.", "target_word": "happy", "answer": "adjective"},
        {"sentence": "She moved quietly through the hallway.", "target_word": "quietly", "answer": "adverb"},
        {"sentence": "The keys are under the table.", "target_word": "under", "answer": "preposition"},
        {"sentence": "I called, but no one answered.", "target_word": "but", "answer": "conjunction"},
        {"sentence": "Hey, can you help me?", "target_word": "Hey", "answer": "interjection"},
        {"sentence": "Those apples look fresh.", "target_word": "Those", "answer": "determiner"},
        {"sentence": "We were planning a trip.", "target_word": "were", "answer": "auxiliary verb"},
        {"sentence": "She won first place.", "target_word": "first", "answer": "numeral"},
        {"sentence": "Please sit down.", "target_word": "down", "answer": "particle"},
        {"sentence": "It seems cold today.", "target_word": "It", "answer": "pronoun"},
        {"sentence": "We met during lunch.", "target_word": "during", "answer": "preposition"},
        {"sentence": "Bravo! That was amazing.", "target_word": "Bravo", "answer": "interjection"},
    ]

    # Prepare data: ensure "prompt" is present and add a "word" alias for template compatibility if needed
    raw = deepcopy(dataset if dataset is not None else default_examples)
    prepared: List[Dict[str, Any]] = []
    for ex in raw:
        ex = {**ex}
        # alias to support templates that might use {word}
        ex.setdefault("word", ex.get("target_word", ""))
        # derive prompt if not provided
        ex.setdefault("prompt", prompt_format.format(**ex))
        # BaseTask expects "input" and "output" columns for evaluation by default—map them here:
        ex["input"] = ex["prompt"]
        ex["output"] = ex["answer"]
        prepared.append(ex)

    config = TaskConfig(
        name=name,
        description='Identify the part of speech of the target word within a sentence.',
        data_format="memory",
        in_memory_data=prepared,
        input_column="input",     # uses constructed prompt string
        output_column="output",   # expected POS label
        prompt_template="{input}",  # pass-through: we've already formatted the prompt
        evaluation_metrics=["accuracy"],
        metadata={
            "task_type": "classification",
            "labels": sorted({ex["answer"].lower() for ex in prepared}),
            "prompt_format": prompt_format,
        },
    )
    return PartOfSpeechTask(config)

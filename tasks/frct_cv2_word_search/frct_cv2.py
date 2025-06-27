# tasks/frct_cv2_word_search/frct_cv2.py

import csv
from typing import Iterator, Dict, Any, List
from tasks.base_task import register_task, Task

@register_task
class FRCT_CV2_WordSearch(Task):
    """
    Word‐search style task: find all the four‐letter words hidden in the scrambled string.
    All Answer1–Answer5 columns are considered correct.
    """

    name = "frct_cv2_word_search"

    def __init__(
        self,
        csv_path: str = "tasks/frct_cv2_word_search/FRCT-LLM_cv2.csv",
    ):
        super().__init__(csv_path=csv_path)

    def get_split(self, split: str) -> Iterator[Dict[str, Any]]:
        if split != "test":
            raise ValueError(f"Split '{split}' not supported for task {self.name}")
        with open(self.csv_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row["Question"].strip()
                refs = [
                    row[f"Answer{i}"].strip()
                    for i in range(1, 6)
                    if row.get(f"Answer{i}") and row[f"Answer{i}"].strip()
                ]
                prompt = (
                    f"Find all the four-letter words hidden in: {question}. "
                    "Separate all words with a comma (','). Answers:"
                )
                yield {
                    "input":      prompt,
                    "output":     refs[0] if refs else "",
                    "references": refs,
                }

    def evaluate(
        self,
        outputs: List[str],
        split: str = "test",
        updated_dataset: List[Dict[str, Any]] = None,
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Override: accuracy == 1 iff the set of comma-split predictions
        exactly equals the set of references.
        """
        if updated_dataset is not None:
            examples = updated_dataset
        else:
            examples = list(self.get_split(split))
            if len(outputs) != len(examples):
                raise ValueError("Number of outputs != number of examples")

        correct = 0
        for pred, ex in zip(outputs, examples):
            # parse model output into a set of cleaned tokens
            pred_tokens = {tok.strip() for tok in pred.split(",") if tok.strip()}
            ref_tokens  = set(ex["references"])
            if pred_tokens == ref_tokens:
                correct += 1

        return {"accuracy": correct / len(examples)}

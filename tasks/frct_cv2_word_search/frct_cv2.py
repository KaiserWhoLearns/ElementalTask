# tasks/frct_cv2_word_search/frct_cv2.py

import csv
from typing import Iterator, Dict, Any

from tasks.base_task import register_task, Task

@register_task
class FRCT_CV2_WordSearch(Task):
    """
    Word‐search style task from the FRCT CV2 CSV.
    All Answer1–Answer5 columns are considered correct.
    """

    name = "frct_cv2_word_search"

    def __init__(
        self,
        csv_path: str = "tasks/frct_cv2_word_search/FRCT-LLM_cv2.csv",
    ):
        super().__init__(csv_path=csv_path)

    def get_split(self, split: str) -> Iterator[Dict[str, Any]]:
        """
        Only a single 'test' split.
        Yields dicts:
          {
            "input":      "Find all the words hidden in: {Question}. Answers:",
            "output":     <first answer> (for legacy code),
            "references": [ans1, ans2, …]  # all valid answers
          }
        """
        if split != "test":
            raise ValueError(f"Split '{split}' not supported for task {self.name}")

        with open(self.csv_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row["Question"].strip()
                # gather every non-empty Answer{i} up to 5
                refs = [
                    row[f"Answer{i}"].strip()
                    for i in range(1, 6)
                    if row.get(f"Answer{i}") and row[f"Answer{i}"].strip()
                ]
                prompt = f"Find all the four-letter words hidden in: {question}. Separate all words with a comma (','). Answers:"
                yield {
                    "input":      prompt,
                    "output":     refs[0] if refs else "",
                    "references": refs,
                }

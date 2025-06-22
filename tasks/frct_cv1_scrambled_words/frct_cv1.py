# tasks/frct_cv1_scrambled_words/frct_cv1.py

import csv
from typing import Iterator, Dict, Any

from tasks.base_task import register_task, Task

@register_task
class FRCT_CV1_ScrambledWords(Task):
    """
    Unscramble words from the FRCT CV1 CSV.
    All Answer1–Answer4 columns are considered correct.
    """

    name = "frct_cv1_scrambled_words"

    def __init__(
        self,
        csv_path: str = "tasks/frct_cv1_scrambled_words/FRCT-LLM.csv",
    ):
        # base __init__ will set self.csv_path for us
        super().__init__(csv_path=csv_path)

    def get_split(self, split: str) -> Iterator[Dict[str, Any]]:
        """
        Only a single 'test' split.
        Yields dicts:
          {
            "input":      "Unscramble the word: {Question}. Answer:",
            "output":     <first answer>,         # optional legacy field
            "references": [ans1, ans2, …]        # all valid answers
          }
        """
        if split != "test":
            raise ValueError(f"Split '{split}' not supported for task {self.name}")

        with open(self.csv_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row["Question"].strip()
                # collect every non-empty Answer{i}
                refs = [
                    row[f"Answer{i}"].strip()
                    for i in range(1, 5)
                    if row.get(f"Answer{i}") and row[f"Answer{i}"].strip()
                ]
                prompt = f"Unscramble the word: {question}. Answer:"
                yield {
                    "input":      prompt,
                    "output":     refs[0] if refs else "",
                    "references": refs,
                }

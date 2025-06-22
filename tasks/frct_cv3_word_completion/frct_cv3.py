# tasks/frct_cv3_word_completion/frct_cv3.py

import csv
from typing import Iterator, Dict, Any

from tasks.base_task import register_task, Task

@register_task
class FRCT_CV3_WordCompletion(Task):
    """
    Fill in the blanks for partially completed words from the FRCT CV3 CSV.
    All Answer1–Answer5 columns are considered correct.
    """

    name = "frct_cv3_word_completion"

    def __init__(
        self,
        csv_path: str = "tasks/frct_cv3_word_completion/FRCT-LLM_cv3.csv",
    ):
        # base Task.__init__ will automatically stash self.csv_path
        super().__init__(csv_path=csv_path)

    def get_split(self, split: str) -> Iterator[Dict[str, Any]]:
        """
        Only a single 'test' split.
        Yields dicts:
          {
            "input":      "Complete the word: {Question}. Answer:",
            "output":     <first answer>,         # legacy field
            "references": [ans1, ans2, …]        # all valid completions
          }
        """
        if split != "test":
            raise ValueError(f"Split '{split}' not supported for task {self.name}")

        with open(self.csv_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row["Question"].strip()
                # collect every non-empty Answer{i} up to 5
                refs = [
                    row[f"Answer{i}"].strip()
                    for i in range(1, 6)
                    if row.get(f"Answer{i}") and row[f"Answer{i}"].strip()
                ]
                prompt = f"Complete the English word: {question}. Answer:"
                yield {
                    "input":      prompt,
                    "output":     refs[0] if refs else "",
                    "references": refs,
                }

import csv
from typing import Iterator, Dict, Any

from tasks.base_task import register_task, Task

@register_task
class FRCT_FA2_Opposites(Task):
    """
    Find a word that means the opposite of the given word.
    """

    name = "frct_fa2_opposites"

    def __init__(
        self,
        csv_path: str = "tasks/frct_fa2_opposites/FRCT-LLM_fa2.csv",
    ):
        # stores self.csv_path
        super().__init__(csv_path=csv_path)

    def get_split(self, split: str) -> Iterator[Dict[str, Any]]:
        """
        Only a single 'test' split.
        Yields:
          {
            "input":      "Think of words with opposite meanings to this word: {Question}. Answer:",
            "output":     <first opposite>,       # legacy
            "references": [ant1, ant2, …]        # all valid opposites
          }
        """
        if split != "test":
            raise ValueError(f"Split '{split}' not supported for task {self.name}")

        with open(self.csv_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row["Question"].strip()
                # collect every non-empty Answer{i} up to 6
                refs = [
                    row[f"Answer{i}"].strip()
                    for i in range(1, 7)
                    if row.get(f"Answer{i}") and row[f"Answer{i}"].strip()
                ]
                prompt = f"Think of words with opposite meanings to this word: {question}. Answer:"
                yield {
                    "input":      prompt,
                    "output":     refs[0] if refs else "",
                    "references": refs,
                }
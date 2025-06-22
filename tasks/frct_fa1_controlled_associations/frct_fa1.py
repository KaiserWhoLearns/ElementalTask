import csv
from typing import Iterator, Dict, Any

from tasks.base_task import register_task, Task

@register_task
class FRCT_FA1_ControlledAssociations(Task):
    """
    To think of words having meanings which are the same as or similar to a given word.
    """

    name = "frct_fa1_controlled_associations"

    def __init__(
        self,
        csv_path: str = "tasks/frct_fa1_controlled_associations/FRCT-LLM_fa1.csv",
    ):
        # stores self.csv_path
        super().__init__(csv_path=csv_path)

    def get_split(self, split: str) -> Iterator[Dict[str, Any]]:
        """
        Only a single 'test' split.
        Yields:
          {
            "input":      "Think of words with similar meanings to this word: {Question}. Answer:",
            "output":     <first controlled_associations>,       # legacy
            "references": [syn1, syn2, …]        # all valid controlled_associationss
          }
        """
        if split != "test":
            raise ValueError(f"Split '{split}' not supported for task {self.name}")

        with open(self.csv_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row["Question"].strip()
                # split the single “Answer” col on commas
                raw = row["Answer"]
                refs = [s.strip() for s in raw.split(",") if s.strip()]
                prompt = f"Think of words with similar meanings to this word: {word}. Answer:"
                yield {
                    "input":      prompt,
                    "output":     refs[0] if refs else "",
                    "references": refs,
                }

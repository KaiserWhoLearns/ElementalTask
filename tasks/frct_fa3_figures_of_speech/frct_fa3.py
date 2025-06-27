import csv
from typing import Iterator, Dict, Any

from tasks.base_task import register_task, Task

@register_task
class FRCT_FA3_FiguresOfSpeech(Task):
    """
    Find words or phrases that could be used in making figures of speech.
    """

    name = "frct_fa3_figures_of_speech"

    def __init__(
        self,
        csv_path: str = "tasks/frct_fa3_figures_of_speech/FRCT-LLM_fa3.csv",
    ):
        # stores self.csv_path
        super().__init__(csv_path=csv_path)

    def get_split(self, split: str) -> Iterator[Dict[str, Any]]:
        """
        Only a single 'test' split.
        Yields:
          {i hate
            "input":      "Think of words or phrases that could be used in making figures of speech which compare one object with another given this word: {Question}. Answer:",
            "output":     <first word related to the question>,       # legacy
            "references": [ant1, ant2, …]        # all valid words or phrases
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
                    for i in range(1, 8)
                    if row.get(f"Answer{i}") and row[f"Answer{i}"].strip()
                ]
                prompt = f"Think of words or phrases that could be used in making figures of speech which compare one object with another given this word: {question}. Answer:"
                yield {
                    "input":      prompt,
                    "output":     refs[0] if refs else "",
                    "references": refs,
                }
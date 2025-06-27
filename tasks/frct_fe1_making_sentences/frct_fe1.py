import csv
from typing import Iterator, Dict, Any, List

from tasks.base_task import register_task, Task

@register_task
class FRCT_FE1_MakingSentences(Task):
    """
    Make sentences based on a given letter.
    Note: the evaluation should be open ended, but we give reference sentences as an example.
    """

    name = "frct_fe1_making_sentences"

    def __init__(
        self,
        csv_path: str = "tasks/frct_fe1_making_sentences/FRCT-LLM_fe1.csv",
    ):
        # stores self.csv_path
        super().__init__(csv_path=csv_path)

    def get_split(self, split: str) -> Iterator[Dict[str, Any]]:
        """
        Only a single 'test' split.
        Yields:
          {
            "input":      "Make a sentence containing words that begin only with the specified letter: {Question}. Answer:",
            "output":     <first sentence>       # legacy
            "references": [ant1, ant2, …]        # a valid sentence (not exhaustive)
          }
        """
        if split != "test":
            raise ValueError(f"Split '{split}' not supported for task {self.name}")

        with open(self.csv_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row["Question"].strip()
                refs = [row["Answer"]]
                prompt = f"Make a sentence containing words that begin only with the specified letter: {question}. Answer:"
                yield {
                    "input":      prompt,
                    "output":     refs[0] if refs else "",
                    "references": refs,
                    "letter": question[0].lower() if question else ""
                }

    def evaluate(
        self,
        outputs: List[str],
        split: str = "test",
        updated_dataset: List[Dict[str, Any]] = None,
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Evaluation is done mechanically, checking that the first letter of
        each word matches the initial letter given. The reference is unused here.
        """
        if updated_dataset is not None:
            examples = updated_dataset
        else:
            examples = list(self.get_split(split))
            if len(outputs) != len(examples):
                raise ValueError("Number of outputs != number of examples")

        correct = 0
        for pred, ex in zip(outputs, examples):
            p = pred[0].strip()
            words = p.split()
            input_letter = ex["letter"][0].lower()
            letters = [word[0].lower() for word in words if word]

            correct += all(letter == input_letter for letter in letters)
        
        acc = correct / len(examples)
        return {"accuracy": acc}


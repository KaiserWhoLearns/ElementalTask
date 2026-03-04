"""Coreference / pronoun resolution — elemental reading comprehension.

Tests the model's ability to resolve pronouns and other referring expressions
to their antecedents. This is a prerequisite for WinoGrande-style benchmarks
and general reading comprehension.

Categories:
  - pronoun_simple:  Unambiguous pronoun resolution (he/she/they)
  - pronoun_hard:    Winograd-style where world knowledge is needed

All cases are *unambiguous* — there is exactly one correct answer.

Format (ICL):
    Sentence: "Alice told Bob that she would be late."
    Who does "she" refer to?
    Answer: Alice
"""

import random
from typing import Dict, List, Any

from tasks.base_task import BaseTask, TaskConfig


class CoreferenceTask(BaseTask):
    """Elemental coreference / pronoun resolution task."""

    TASK_NAME = "coreference"

    CATEGORY_DATA: Dict[str, List[Dict[str, str]]] = {
        "pronoun_simple": [
            {"input": 'Sentence: "Alice told Bob that she would be late."\nWho does "she" refer to?', "output": "Alice"},
            {"input": 'Sentence: "David asked Sarah to help him with the project."\nWho does "him" refer to?', "output": "David"},
            {"input": 'Sentence: "Maria finished her homework before dinner."\nWho does "her" refer to?', "output": "Maria"},
            {"input": 'Sentence: "The teacher praised James because he scored the highest."\nWho does "he" refer to?', "output": "James"},
            {"input": 'Sentence: "Emma called her mother and told her the good news."\nWho does the first "her" refer to?', "output": "Emma"},
            {"input": 'Sentence: "Tom lent his car to his brother Mike."\nWho does "his" refer to?', "output": "Tom"},
            {"input": 'Sentence: "When Lisa entered the room, she noticed the window was open."\nWho does "she" refer to?', "output": "Lisa"},
            {"input": 'Sentence: "The doctor told the patient that he needed more rest."\nWho does "he" refer to?', "output": "the patient"},
            {"input": 'Sentence: "Kevin and his dog went for a walk in the park."\nWho does "his" refer to?', "output": "Kevin"},
            {"input": 'Sentence: "Rachel sent a letter to her grandmother."\nWho does "her" refer to?', "output": "Rachel"},
            {"input": 'Sentence: "Mr. Johnson asked his secretary to reschedule the meeting."\nWho does "his" refer to?', "output": "Mr. Johnson"},
            {"input": 'Sentence: "Sophie baked a cake and brought it to the party."\nWhat does "it" refer to?', "output": "the cake"},
            {"input": 'Sentence: "After Mark finished his speech, he received a standing ovation."\nWho does "he" refer to?', "output": "Mark"},
            {"input": 'Sentence: "Diana wrote a novel and published it last year."\nWhat does "it" refer to?', "output": "the novel"},
            {"input": 'Sentence: "The manager told the employee that she had been promoted."\nWho does "she" refer to?', "output": "the employee"},
            {"input": 'Sentence: "Paul took his children to the zoo on Saturday."\nWho does "his" refer to?', "output": "Paul"},
            {"input": 'Sentence: "Olivia practiced the piano until she could play the piece perfectly."\nWho does "she" refer to?', "output": "Olivia"},
            {"input": 'Sentence: "The chef prepared a special dish and served it to the guests."\nWhat does "it" refer to?', "output": "the dish"},
            {"input": 'Sentence: "Laura thanked her friend for helping her move."\nWho does the first "her" refer to?', "output": "Laura"},
            {"input": 'Sentence: "Ben fixed the car and drove it to work."\nWhat does "it" refer to?', "output": "the car"},
        ],

        "pronoun_hard": [
            {"input": 'Sentence: "The trophy didn\'t fit in the suitcase because it was too big."\nWhat was too big?', "output": "the trophy"},
            {"input": 'Sentence: "The trophy didn\'t fit in the suitcase because it was too small."\nWhat was too small?', "output": "the suitcase"},
            {"input": 'Sentence: "The city council refused the demonstrators a permit because they feared violence."\nWho feared violence?', "output": "the city council"},
            {"input": 'Sentence: "The city council refused the demonstrators a permit because they advocated violence."\nWho advocated violence?', "output": "the demonstrators"},
            {"input": 'Sentence: "Sam pulled the wagon and Chris pushed it, even though he was very tired."\nWho was tired?', "output": "Sam"},
            {"input": 'Sentence: "The painting was sold to the collector because it was very valuable."\nWhat was valuable?', "output": "the painting"},
            {"input": 'Sentence: "The plant didn\'t grow well in the pot because it was too dry."\nWhat was too dry?', "output": "the pot"},
            {"input": 'Sentence: "The cat caught the mouse because it was too slow."\nWhat was too slow?', "output": "the mouse"},
            {"input": 'Sentence: "The cat caught the mouse because it was very quick."\nWhat was quick?', "output": "the cat"},
            {"input": 'Sentence: "Joan made sure to thank Susan for all the help she had given."\nWho gave help?', "output": "Susan"},
            {"input": 'Sentence: "The table won\'t fit through the door because it is too wide."\nWhat is too wide?', "output": "the table"},
            {"input": 'Sentence: "The table won\'t fit through the door because it is too narrow."\nWhat is too narrow?', "output": "the door"},
            {"input": 'Sentence: "Bob collapsed on the sidewalk, and Jim called an ambulance for him."\nWho needed the ambulance?', "output": "Bob"},
            {"input": 'Sentence: "The doctor told the nurse that she had made an error in the chart."\nWho made an error?', "output": "the nurse"},
            {"input": 'Sentence: "The truck hit the pole and it broke into pieces."\nWhat broke?', "output": "the pole"},
            {"input": 'Sentence: "The actress used to be the director\'s favorite, but she replaced her."\nWho was replaced?', "output": "the actress"},
            {"input": 'Sentence: "The ball broke the window because it was fragile."\nWhat was fragile?', "output": "the window"},
            {"input": 'Sentence: "The ball broke the window because it was moving fast."\nWhat was moving fast?', "output": "the ball"},
            {"input": 'Sentence: "The guests enjoyed the meal that the chef prepared because it was delicious."\nWhat was delicious?', "output": "the meal"},
            {"input": 'Sentence: "The lock couldn\'t be opened by the key because it was rusty."\nWhat was rusty?', "output": "the lock"},
        ],
    }

    CATEGORY_DEMOS: Dict[str, List[str]] = {
        "pronoun_simple": [
            'Sentence: "John asked Mary if she could help."\nWho does "she" refer to?\nAnswer: Mary',
            'Sentence: "Carlos lost his wallet at the store."\nWho does "his" refer to?\nAnswer: Carlos',
            'Sentence: "The professor graded the papers and returned them to the students."\nWhat does "them" refer to?\nAnswer: the papers',
            'Sentence: "Anna brought her dog to the vet."\nWho does "her" refer to?\nAnswer: Anna',
            'Sentence: "The boy kicked the ball and it flew over the fence."\nWhat does "it" refer to?\nAnswer: the ball',
        ],
        "pronoun_hard": [
            'Sentence: "The bottle didn\'t fit in the bag because it was too large."\nWhat was too large?\nAnswer: the bottle',
            'Sentence: "The town council denied the miners a parade permit because they worried about damage."\nWho worried?\nAnswer: the town council',
            'Sentence: "The book fell off the shelf because it was uneven."\nWhat was uneven?\nAnswer: the shelf',
            'Sentence: "Bill passed the salt to John because he needed it for his soup."\nWho needed the salt?\nAnswer: John',
            'Sentence: "The vase broke when the mover dropped it because it was slippery."\nWhat was slippery?\nAnswer: the vase',
        ],
    }

    def __init__(self, config: TaskConfig):
        super().__init__(config)

    def _load_data(self):
        import pandas as pd

        if self.config.in_memory_data:
            self.data = pd.DataFrame(self.config.in_memory_data)
            return

        rows: List[Dict[str, str]] = []
        for category, examples in self.CATEGORY_DATA.items():
            for ex in examples:
                rows.append({
                    "input": ex["input"],
                    "output": ex["output"],
                    "category_name": category,
                })
        self.data = pd.DataFrame(rows)

    def build_prompt(self, instance: Dict[str, Any], num_shots: int = 5) -> str:
        category = instance.get("category_name", "pronoun_simple")
        demos = self.CATEGORY_DEMOS.get(category, [])

        inst_input = instance.get("input", "")
        demos = [d for d in demos if inst_input not in d][:num_shots]

        prompt = ""
        for d in demos:
            prompt += d + "\n\n"
        prompt += f"{inst_input}\nAnswer:"
        return prompt

    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        ground_truth = self.get_ground_truth(split)
        task_data = self.get_split(split)

        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Prediction count ({len(predictions)}) != ground truth ({len(ground_truth)})"
            )

        processed = [self.preprocess_prediction(p) for p in predictions]

        correct = sum(
            1 for p, g in zip(processed, ground_truth)
            if p.lower().strip() == g.lower().strip()
        )
        results: Dict[str, Any] = {
            "accuracy": correct / len(ground_truth),
            "correct": correct,
            "total": len(ground_truth),
        }

        cat_stats: Dict[str, Dict[str, int]] = {}
        for p, g, item in zip(processed, ground_truth, task_data):
            cat = item.get("category_name", "unknown")
            cat_stats.setdefault(cat, {"correct": 0, "total": 0})
            cat_stats[cat]["total"] += 1
            if p.lower().strip() == g.lower().strip():
                cat_stats[cat]["correct"] += 1

        for cat, s in cat_stats.items():
            results[f"accuracy_{cat}"] = s["correct"] / s["total"]

        return results

    def get_ground_truth(self, split: str = "test") -> List[str]:
        rows = self.get_split(split)
        return [str(r.get("output", "")) for r in rows]


def create_coreference_task(
    category: str = None,
    name: str = "coreference",
) -> CoreferenceTask:
    """Create a CoreferenceTask, optionally filtered to one category."""
    data = None
    if category and category in CoreferenceTask.CATEGORY_DATA:
        data = [
            {**ex, "category_name": category}
            for ex in CoreferenceTask.CATEGORY_DATA[category]
        ]
        name = f"coreference:{category}"

    config = TaskConfig(
        name=name,
        description="Coreference / pronoun resolution",
        data_format="memory",
        in_memory_data=data,
        input_column="input",
        output_column="output",
        evaluation_metrics=["accuracy"],
        metadata={"task_type": "coreference", "category": category},
    )
    return CoreferenceTask(config)

"""Fact extraction from context — elemental reading comprehension.

Tests the model's ability to extract a specific fact from a short passage
given a question. This is the core skill underlying virtually all QA benchmarks.

Categories:
  - extract_entity:   "Who/What" questions about named entities
  - extract_number:   "How many/much" questions about quantities
  - extract_location: "Where" questions about places

Format (ICL):
    Passage: "The cat sat on the red mat in the kitchen."
    Question: Where is the mat?
    Answer: the kitchen

Each category tests a different extraction target type while keeping
the passages simple and unambiguous.
"""

import random
from typing import Dict, List, Any

from tasks.base_task import BaseTask, TaskConfig


class FactExtractionTask(BaseTask):
    """Elemental fact extraction from short passages."""

    TASK_NAME = "fact_extraction"

    # ── static data ──────────────────────────────────────────────────

    CATEGORY_DATA: Dict[str, List[Dict[str, str]]] = {
        "extract_entity": [
            {"input": "Passage: \"Alice gave five apples to Bob at the park.\"\nQuestion: Who received the apples?", "output": "Bob"},
            {"input": "Passage: \"Dr. Smith performed the surgery on Tuesday morning.\"\nQuestion: Who performed the surgery?", "output": "Dr. Smith"},
            {"input": "Passage: \"The painting was created by Leonardo in 1503.\"\nQuestion: Who created the painting?", "output": "Leonardo"},
            {"input": "Passage: \"Mary told John that the meeting was cancelled.\"\nQuestion: Who told John about the meeting?", "output": "Mary"},
            {"input": "Passage: \"The award was given to Professor Chen for her research.\"\nQuestion: Who received the award?", "output": "Professor Chen"},
            {"input": "Passage: \"Tom built a wooden birdhouse for his daughter Emma.\"\nQuestion: Who was the birdhouse for?", "output": "Emma"},
            {"input": "Passage: \"Sarah found a lost wallet and returned it to Mr. Davis.\"\nQuestion: Who found the wallet?", "output": "Sarah"},
            {"input": "Passage: \"The letter was written by Queen Victoria in 1887.\"\nQuestion: Who wrote the letter?", "output": "Queen Victoria"},
            {"input": "Passage: \"Coach Martinez led the team to three consecutive victories.\"\nQuestion: Who led the team?", "output": "Coach Martinez"},
            {"input": "Passage: \"The vaccine was developed by Dr. Park and her team.\"\nQuestion: Who developed the vaccine?", "output": "Dr. Park"},
            {"input": "Passage: \"James lent his bicycle to his neighbor Kevin.\"\nQuestion: Who lent the bicycle?", "output": "James"},
            {"input": "Passage: \"The novel was dedicated to the author's mother, Helen.\"\nQuestion: Who was the novel dedicated to?", "output": "Helen"},
            {"input": "Passage: \"Officer Wilson responded to the emergency call first.\"\nQuestion: Who responded first?", "output": "Officer Wilson"},
            {"input": "Passage: \"The recipe was passed down from Grandma Rose.\"\nQuestion: Who passed down the recipe?", "output": "Grandma Rose"},
            {"input": "Passage: \"Daniel taught piano lessons to young students.\"\nQuestion: Who taught piano?", "output": "Daniel"},
            {"input": "Passage: \"The bridge was designed by engineer Clara Nguyen.\"\nQuestion: Who designed the bridge?", "output": "Clara Nguyen"},
            {"input": "Passage: \"Principal Adams announced the new school policy.\"\nQuestion: Who announced the policy?", "output": "Principal Adams"},
            {"input": "Passage: \"The statue was sculpted by Marcus from a single block of marble.\"\nQuestion: Who sculpted the statue?", "output": "Marcus"},
            {"input": "Passage: \"Lisa organized the charity event last weekend.\"\nQuestion: Who organized the event?", "output": "Lisa"},
            {"input": "Passage: \"The discovery was made by researcher Yuki Tanaka.\"\nQuestion: Who made the discovery?", "output": "Yuki Tanaka"},
        ],

        "extract_number": [
            {"input": "Passage: \"John gave 5 apples to Mary on Tuesday.\"\nQuestion: How many apples did John give?", "output": "5"},
            {"input": "Passage: \"The project took 14 months to complete.\"\nQuestion: How many months did the project take?", "output": "14"},
            {"input": "Passage: \"There were 32 students in the classroom.\"\nQuestion: How many students were there?", "output": "32"},
            {"input": "Passage: \"She scored 97 points on the final exam.\"\nQuestion: How many points did she score?", "output": "97"},
            {"input": "Passage: \"The recipe calls for 3 cups of flour.\"\nQuestion: How many cups of flour are needed?", "output": "3"},
            {"input": "Passage: \"He ran 26 miles during the marathon.\"\nQuestion: How many miles did he run?", "output": "26"},
            {"input": "Passage: \"The library has 12000 books in its collection.\"\nQuestion: How many books does the library have?", "output": "12000"},
            {"input": "Passage: \"The flight lasted 8 hours and arrived at noon.\"\nQuestion: How many hours did the flight last?", "output": "8"},
            {"input": "Passage: \"She planted 50 tulip bulbs in the garden.\"\nQuestion: How many tulip bulbs were planted?", "output": "50"},
            {"input": "Passage: \"The building has 15 floors and a rooftop terrace.\"\nQuestion: How many floors does the building have?", "output": "15"},
            {"input": "Passage: \"He saved 200 dollars each month for a year.\"\nQuestion: How much did he save each month?", "output": "200"},
            {"input": "Passage: \"The concert was attended by 4500 fans.\"\nQuestion: How many fans attended?", "output": "4500"},
            {"input": "Passage: \"They ordered 6 pizzas for the party.\"\nQuestion: How many pizzas were ordered?", "output": "6"},
            {"input": "Passage: \"The bridge spans 480 meters across the river.\"\nQuestion: How many meters does the bridge span?", "output": "480"},
            {"input": "Passage: \"There were 7 candidates in the election.\"\nQuestion: How many candidates were there?", "output": "7"},
            {"input": "Passage: \"She read 42 books last summer.\"\nQuestion: How many books did she read?", "output": "42"},
            {"input": "Passage: \"The team won 18 games this season.\"\nQuestion: How many games did the team win?", "output": "18"},
            {"input": "Passage: \"He bought 3 tickets for the show.\"\nQuestion: How many tickets did he buy?", "output": "3"},
            {"input": "Passage: \"The experiment used 120 subjects.\"\nQuestion: How many subjects were used?", "output": "120"},
            {"input": "Passage: \"The cake needs 4 eggs and 2 cups of sugar.\"\nQuestion: How many eggs does the cake need?", "output": "4"},
        ],

        "extract_location": [
            {"input": "Passage: \"The cat sat on the red mat in the kitchen.\"\nQuestion: Where is the mat?", "output": "the kitchen"},
            {"input": "Passage: \"She left her keys on the table by the front door.\"\nQuestion: Where are the keys?", "output": "on the table"},
            {"input": "Passage: \"The meeting will be held in Conference Room B.\"\nQuestion: Where will the meeting be held?", "output": "Conference Room B"},
            {"input": "Passage: \"He parked his car in the underground garage.\"\nQuestion: Where did he park?", "output": "the underground garage"},
            {"input": "Passage: \"The children were playing in the backyard.\"\nQuestion: Where were the children playing?", "output": "the backyard"},
            {"input": "Passage: \"The museum is located on Fifth Avenue.\"\nQuestion: Where is the museum?", "output": "Fifth Avenue"},
            {"input": "Passage: \"She found the lost ring under the sofa cushion.\"\nQuestion: Where was the ring found?", "output": "under the sofa cushion"},
            {"input": "Passage: \"The concert takes place at Madison Square Garden.\"\nQuestion: Where does the concert take place?", "output": "Madison Square Garden"},
            {"input": "Passage: \"He stored the documents in the filing cabinet.\"\nQuestion: Where are the documents stored?", "output": "the filing cabinet"},
            {"input": "Passage: \"The birds built a nest in the old oak tree.\"\nQuestion: Where did the birds build a nest?", "output": "the old oak tree"},
            {"input": "Passage: \"The festival is celebrated in the town square.\"\nQuestion: Where is the festival celebrated?", "output": "the town square"},
            {"input": "Passage: \"She hung the painting above the fireplace.\"\nQuestion: Where is the painting?", "output": "above the fireplace"},
            {"input": "Passage: \"The treasure was buried on a remote island.\"\nQuestion: Where was the treasure buried?", "output": "a remote island"},
            {"input": "Passage: \"He keeps his tools in the shed behind the house.\"\nQuestion: Where does he keep his tools?", "output": "the shed"},
            {"input": "Passage: \"The flowers were growing along the fence.\"\nQuestion: Where were the flowers growing?", "output": "along the fence"},
            {"input": "Passage: \"She studies at the library every evening.\"\nQuestion: Where does she study?", "output": "the library"},
            {"input": "Passage: \"The wedding was held in a garden by the lake.\"\nQuestion: Where was the wedding held?", "output": "a garden by the lake"},
            {"input": "Passage: \"He left his umbrella at the restaurant.\"\nQuestion: Where did he leave his umbrella?", "output": "the restaurant"},
            {"input": "Passage: \"The conference is being held in Tokyo this year.\"\nQuestion: Where is the conference?", "output": "Tokyo"},
            {"input": "Passage: \"She placed the vase on the windowsill.\"\nQuestion: Where is the vase?", "output": "the windowsill"},
        ],
    }

    # Demonstrations for ICL (separate from test data)
    CATEGORY_DEMOS: Dict[str, List[str]] = {
        "extract_entity": [
            'Passage: "The book was written by Charles Dickens in 1859."\nQuestion: Who wrote the book?\nAnswer: Charles Dickens',
            'Passage: "Emily gave the report to her manager David."\nQuestion: Who received the report?\nAnswer: David',
            'Passage: "The cake was baked by Aunt Martha for the birthday."\nQuestion: Who baked the cake?\nAnswer: Aunt Martha',
            'Passage: "Professor Lee published the findings in Nature."\nQuestion: Who published the findings?\nAnswer: Professor Lee',
            'Passage: "The house was sold to Mr. and Mrs. Thompson."\nQuestion: Who bought the house?\nAnswer: Mr. and Mrs. Thompson',
        ],
        "extract_number": [
            'Passage: "The train departed at 9 AM with 150 passengers."\nQuestion: How many passengers were on the train?\nAnswer: 150',
            'Passage: "She bought 4 tickets for the movie."\nQuestion: How many tickets did she buy?\nAnswer: 4',
            'Passage: "The marathon was 42 kilometers long."\nQuestion: How many kilometers was the marathon?\nAnswer: 42',
            'Passage: "He ate 3 slices of pizza for dinner."\nQuestion: How many slices did he eat?\nAnswer: 3',
            'Passage: "There are 24 hours in a day."\nQuestion: How many hours are in a day?\nAnswer: 24',
        ],
        "extract_location": [
            'Passage: "The dog was sleeping under the porch."\nQuestion: Where was the dog?\nAnswer: under the porch',
            'Passage: "She put the groceries on the counter."\nQuestion: Where are the groceries?\nAnswer: on the counter',
            'Passage: "The game will be played at the stadium."\nQuestion: Where will the game be played?\nAnswer: the stadium',
            'Passage: "He hid the present in the closet."\nQuestion: Where is the present?\nAnswer: the closet',
            'Passage: "The plane landed at Heathrow Airport."\nQuestion: Where did the plane land?\nAnswer: Heathrow Airport',
        ],
    }

    def __init__(self, config: TaskConfig):
        super().__init__(config)

    def _load_data(self):
        """Build data from hardcoded examples."""
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

    # ── prompt building ──────────────────────────────────────────────

    def build_prompt(self, instance: Dict[str, Any], num_shots: int = 5) -> str:
        category = instance.get("category_name", "extract_entity")
        demos = self.CATEGORY_DEMOS.get(category, [])

        # Filter demos that overlap with the current instance
        inst_input = instance.get("input", "")
        demos = [d for d in demos if inst_input not in d][:num_shots]

        prompt = ""
        for d in demos:
            prompt += d + "\n\n"

        prompt += f"{inst_input}\nAnswer:"
        return prompt

    # ── evaluation ───────────────────────────────────────────────────

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

        # Per-category
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


def create_fact_extraction_task(
    category: str = None,
    name: str = "fact_extraction",
) -> FactExtractionTask:
    """Create a FactExtractionTask, optionally filtered to one category."""
    data = None
    if category and category in FactExtractionTask.CATEGORY_DATA:
        data = [
            {**ex, "category_name": category}
            for ex in FactExtractionTask.CATEGORY_DATA[category]
        ]
        name = f"fact_extraction:{category}"

    config = TaskConfig(
        name=name,
        description="Fact extraction from short passages",
        data_format="memory",
        in_memory_data=data,
        input_column="input",
        output_column="output",
        evaluation_metrics=["accuracy"],
        metadata={"task_type": "fact_extraction", "category": category},
    )
    return FactExtractionTask(config)

"""Blended compositions task combining multiple reasoning primitives.

These items are designed to look more downstream-like than pure string transforms:
- coref_tracking_query: resolve pronouns, then do entity-state tracking
- opplan_solve: choose operation sequence, then compute numeric result
- decipher_apply_reason: infer token mapping and apply it to a new query
- extract_verify: extract facts from text and verify a claim
"""

from typing import Any, Dict, List

from tasks.base_task import BaseTask, TaskConfig


class BlendedCompositionsTask(BaseTask):
    """Task with hand-authored blended reasoning compositions."""

    TASK_NAME = "blended_compositions"

    CATEGORY_DATA: Dict[str, List[Dict[str, str]]] = {
        "coref_tracking_query": [
            {
                "input": "Mia has 8 tokens. Jay has 5. He gives 2 to Mia. Then she gives 3 to Jay. How many tokens does Mia have?",
                "output": "7",
            },
            {
                "input": "Lena has 10 marbles. Omar has 4. She gives 3 to Omar. Then he gives 1 to Lena. How many marbles does Omar have?",
                "output": "6",
            },
            {
                "input": "Nora has 7 books. Paul has 9. He gives 2 to Nora. Then she gives 4 to Paul. How many books does Nora have?",
                "output": "5",
            },
            {
                "input": "Ivy has 12 cards. Ben has 3. She gives 5 to Ben. Then he gives 2 to Ivy. How many cards does Ben have?",
                "output": "6",
            },
            {
                "input": "Kara has 6 apples. Dean has 11. He gives 4 to Kara. Then she gives 1 to Dean. How many apples does Kara have?",
                "output": "9",
            },
            {
                "input": "Tia has 14 coins. Ron has 2. She gives 6 to Ron. Then he gives 3 to Tia. How many coins does Tia have?",
                "output": "11",
            },
        ],
        "opplan_solve": [
            {
                "input": "A shelf has 6 boxes with 4 notebooks each. Then 5 notebooks are removed. What is the final number of notebooks?",
                "output": "19",
            },
            {
                "input": "A school bought 9 packs of 8 pencils. Then 14 pencils were used. How many pencils remain?",
                "output": "58",
            },
            {
                "input": "A bakery made 7 trays with 5 muffins each. Then it baked 12 more muffins. How many muffins are there now?",
                "output": "47",
            },
            {
                "input": "A box has 40 candies. They are split equally into 5 bags. How many candies per bag?",
                "output": "8",
            },
            {
                "input": "A farmer has 36 eggs. She packs them into cartons of 6. Then 2 cartons are sold. How many eggs remain?",
                "output": "24",
            },
            {
                "input": "A runner completed 4 laps of 3 km each, then ran 2 more km. How many km total?",
                "output": "14",
            },
        ],
        "decipher_apply_reason": [
            {
                "input": "Known pairs: black sheep = dag kip; white dog = tin bud; black cow = dag stam. Query: white sheep = ?\nOptions: 1) dag kip 2) tin kip 3) stam dag 4) bud tin",
                "output": "2",
            },
            {
                "input": "Known pairs: red fish = zor mek; blue fish = lan mek; red bird = zor pil. Query: blue bird = ?\nOptions: 1) zor mek 2) lan pil 3) pil lan 4) mek zor",
                "output": "2",
            },
            {
                "input": "Known pairs: small car = nim tor; large car = gor tor; small boat = nim vek. Query: large boat = ?\nOptions: 1) gor vek 2) nim tor 3) vek gor 4) tor gor",
                "output": "1",
            },
            {
                "input": "Known pairs: hot tea = fa lun; cold tea = si lun; hot soup = fa rem. Query: cold soup = ?\nOptions: 1) si rem 2) fa rem 3) lun si 4) rem fa",
                "output": "1",
            },
            {
                "input": "Known pairs: north road = ta mur; south road = ko mur; north gate = ta fen. Query: south gate = ?\nOptions: 1) ta fen 2) ko fen 3) mur ko 4) fen ta",
                "output": "2",
            },
            {
                "input": "Known pairs: old tree = vra sal; young tree = len sal; old house = vra dom. Query: young house = ?\nOptions: 1) len dom 2) vra dom 3) sal len 4) dom vra",
                "output": "1",
            },
        ],
        "extract_verify": [
            {
                "input": "Passage: Nora gave 3 apples to Ben. Ben then gave 1 apple to Li.\nClaim: Ben received apples before giving any away.\nDoes the claim follow?",
                "output": "True",
            },
            {
                "input": "Passage: Ravi arrived after Mina, but before Joel.\nClaim: Joel arrived before Mina.\nDoes the claim follow?",
                "output": "False",
            },
            {
                "input": "Passage: The red folder is on the desk. The blue folder is in the drawer.\nClaim: The red folder is not in the drawer.\nDoes the claim follow?",
                "output": "True",
            },
            {
                "input": "Passage: Every metal key opens Gate A. Key K is metal.\nClaim: Key K opens Gate A.\nDoes the claim follow?",
                "output": "True",
            },
            {
                "input": "Passage: Only students with badges may enter. Omar enters the lab.\nClaim: Omar has a badge.\nDoes the claim follow?",
                "output": "True",
            },
            {
                "input": "Passage: If the alarm rings, the hall is evacuated. The hall is evacuated.\nClaim: The alarm rang.\nDoes the claim follow?",
                "output": "False",
            },
        ],
    }

    CATEGORY_DEMOS: Dict[str, List[str]] = {
        "coref_tracking_query": [
            "Input: Ava has 9 coins. Leo has 4. He gives 3 to Ava. Then she gives 2 to Leo. How many coins does Ava have?\nOutput: 10",
            "Input: Kim has 11 cards. Max has 5. She gives 4 to Max. Then he gives 1 to Kim. How many cards does Max have?\nOutput: 8",
            "Input: Uma has 7 beads. Raj has 10. He gives 2 to Uma. Then she gives 5 to Raj. How many beads does Uma have?\nOutput: 4",
            "Input: Zoe has 13 stickers. Ian has 3. She gives 6 to Ian. Then he gives 2 to Zoe. How many stickers does Ian have?\nOutput: 7",
            "Input: Pia has 8 marbles. Ned has 9. He gives 1 to Pia. Then she gives 3 to Ned. How many marbles does Pia have?\nOutput: 6",
        ],
        "opplan_solve": [
            "Input: There are 5 packs with 6 markers each. Then 4 markers are lost. What is the final number of markers?\nOutput: 26",
            "Input: A crate has 48 oranges. They are packed equally into 6 bags. How many oranges per bag?\nOutput: 8",
            "Input: A class has 8 rows with 3 chairs each, then adds 5 chairs. How many chairs now?\nOutput: 29",
            "Input: A factory makes 9 boxes with 7 bolts each, then ships 10 bolts away. How many bolts remain?\nOutput: 53",
            "Input: A store has 60 cans and groups them into 10 shelves equally. How many cans per shelf?\nOutput: 6",
        ],
        "decipher_apply_reason": [
            "Input: Known pairs: black cat = dag mif; white cat = tin mif; black bird = dag pil. Query: white bird = ?\nOptions: 1) dag pil 2) tin pil 3) pil dag 4) mif tin\nOutput: 2",
            "Input: Known pairs: hot drink = fa lun; cold drink = si lun; hot meal = fa rem. Query: cold meal = ?\nOptions: 1) si rem 2) fa rem 3) lun si 4) rem fa\nOutput: 1",
            "Input: Known pairs: old stone = vra tok; new stone = len tok; old wall = vra dom. Query: new wall = ?\nOptions: 1) len dom 2) vra dom 3) tok len 4) dom vra\nOutput: 1",
            "Input: Known pairs: east road = te mur; west road = ko mur; east gate = te fen. Query: west gate = ?\nOptions: 1) ko fen 2) te fen 3) mur ko 4) fen te\nOutput: 1",
            "Input: Known pairs: small cup = nim bal; large cup = gor bal; small plate = nim vak. Query: large plate = ?\nOptions: 1) gor vak 2) nim bal 3) vak gor 4) bal gor\nOutput: 1",
        ],
        "extract_verify": [
            "Input: Passage: Mira left before Taro, and Taro left before June.\nClaim: Mira left before June.\nDoes the claim follow?\nOutput: True",
            "Input: Passage: Every blue key opens Door B. Key Q is blue.\nClaim: Key Q opens Door B.\nDoes the claim follow?\nOutput: True",
            "Input: Passage: If the bell rings, class begins. Class begins.\nClaim: The bell rang.\nDoes the claim follow?\nOutput: False",
            "Input: Passage: The map is in the drawer. The pen is on the desk.\nClaim: The map is not on the desk.\nDoes the claim follow?\nOutput: True",
            "Input: Passage: Nia arrived after Omar and before Paul.\nClaim: Paul arrived before Omar.\nDoes the claim follow?\nOutput: False",
        ],
    }

    def __init__(self, config: TaskConfig):
        super().__init__(config)

    def _load_data(self):
        import pandas as pd

        rows: List[Dict[str, str]] = []
        for category, examples in self.CATEGORY_DATA.items():
            for ex in examples:
                rows.append({
                    "input": ex["input"],
                    "output": ex["output"],
                    "category_name": category,
                })

        data = pd.DataFrame(rows)

        # Support get_task("blended_compositions:cat1,cat2") via config.name fallback path.
        name = str(getattr(self.config, "name", ""))
        if ":" in name:
            _, spec = name.split(":", 1)
            wanted = [s.strip() for s in spec.split(",") if s.strip()]
            if wanted:
                data = data[data["category_name"].isin(wanted)]

        self.data = data

    def build_prompt(self, instance: Dict[str, Any], num_shots: int = 5) -> str:
        category = instance.get("category_name", "")
        demos = self.CATEGORY_DEMOS.get(category, [])

        inst_input = instance.get("input", "")
        selected = [d for d in demos if inst_input not in d][:num_shots]

        prompt = ""
        for d in selected:
            prompt += d + "\n\n"
        prompt += f"Input: {inst_input}\nOutput:"
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
            "accuracy": correct / len(ground_truth) if ground_truth else 0.0,
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
            results[f"accuracy_{cat}"] = s["correct"] / s["total"] if s["total"] else 0.0
            results[f"correct_{cat}"] = s["correct"]
            results[f"total_{cat}"] = s["total"]

        return results


def create_blended_compositions_task(
    category: str = None,
    name: str = "blended_compositions",
) -> BlendedCompositionsTask:
    """Create blended compositions task; optionally filter to a category."""
    data = None
    if category and category in BlendedCompositionsTask.CATEGORY_DATA:
        data = [
            {**ex, "category_name": category}
            for ex in BlendedCompositionsTask.CATEGORY_DATA[category]
        ]
        name = f"blended_compositions:{category}"

    config = TaskConfig(
        name=name,
        description="Blended downstream-like compositions across reasoning skills",
        data_format="memory",
        in_memory_data=data,
        input_column="input",
        output_column="output",
        evaluation_metrics=["accuracy"],
        metadata={"task_type": "blended_compositions", "category": category},
    )
    return BlendedCompositionsTask(config)

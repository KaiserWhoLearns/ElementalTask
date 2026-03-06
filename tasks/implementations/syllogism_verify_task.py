"""Syllogism Verification Task based on Lampinen & Dasgupta (2024) (https://arxiv.org/pdf/2207.07051).

Task: Given two premises and a conclusion, determine if the conclusion
logically follows from the premises.
Format:
    Input: [major premise] [minor premise] Therefore, [conclusion]
    Output: Valid / Invalid

Following Lampinen & Dasgupta (2024), we provide three categories that differ on
consistency with typicality/world knowledge:
    - consistent: premises and conclusions align with world knowledge
    - violate: logically valid conclusions that violate world knowledge
    - nonce: nonsense words isolate pure logical reasoning from world knowledge
"""
from typing import Dict, List, Any

from tasks.base_task import BaseTask, TaskConfig


class SyllogismVerifyTask(BaseTask):
    """Syllogism verification task — determine Valid or Invalid."""

    TASK_NAME = "syllogism_verify"

    CATEGORY_DATA: Dict[str, List[Dict[str, str]]] = {
        "consistent": [
            # Valid
            {"input": "All mammals are warm-blooded. Dogs are mammals. Therefore, dogs are warm-blooded.", "output": "Valid"},
            {"input": "No insects have a backbone. All bees are insects. Therefore, no bees have a backbone.", "output": "Valid"},
            {"input": "All carpenters work with wood. Some craftspeople are carpenters. Therefore, some craftspeople work with wood.", "output": "Valid"},
            {"input": "All birds have feathers. Robins are birds. Therefore, robins have feathers.", "output": "Valid"},
            {"input": "No reptiles are warm-blooded. Snakes are reptiles. Therefore, snakes are not warm-blooded.", "output": "Valid"},
            {"input": "All vertebrates have a spine. Humans are vertebrates. Therefore, humans have a spine.", "output": "Valid"},
            {"input": "No liquids have a fixed shape. Water is a liquid. Therefore, water does not have a fixed shape.", "output": "Valid"},
            {"input": "No metals are transparent. Gold is a metal. Therefore, gold is not transparent.", "output": "Valid"},
            # Invalid
            {"input": "All mammals are warm-blooded. Cats are warm-blooded. Therefore, cats are mammals.", "output": "Invalid"},
            {"input": "All birds have feathers. Eagles have feathers. Therefore, eagles are the only animals with feathers.", "output": "Invalid"},
            {"input": "All doctors have medical degrees. All lawyers have degrees. Therefore, all doctors are lawyers.", "output": "Invalid"},
            {"input": "No insects have a backbone. Worms have no backbone. Therefore, worms are insects.", "output": "Invalid"},
            {"input": "All planets orbit a star. Earth orbits a star. Therefore, Earth is the only planet.", "output": "Invalid"},
            {"input": "All fish live in water. Sharks live in water. Therefore, sharks are fish.", "output": "Invalid"},
            {"input": "No metals are organic. Gold is not organic. Therefore, gold is a metal.", "output": "Invalid"},
        ],
        "violate": [
            # Valid (logically valid, but conclusion violates world knowledge)
            {"input": "All birds can fly. Penguins are birds. Therefore, penguins can fly.", "output": "Valid"},
            {"input": "No mammals live in water. Whales are mammals. Therefore, whales do not live in water.", "output": "Valid"},
            {"input": "All metals sink in water. Lithium is a metal. Therefore, lithium sinks in water.", "output": "Valid"},
            {"input": "All deserts are hot. Antarctica is a desert. Therefore, Antarctica is hot.", "output": "Valid"},
            {"input": "No large animals can jump. Kangaroos are large animals. Therefore, kangaroos cannot jump.", "output": "Valid"},
            {"input": "All snakes have legs. Cobras are snakes. Therefore, cobras have legs.", "output": "Valid"},
            {"input": "All oceanic creatures are fish. Dolphins are oceanic creatures. Therefore, dolphins are fish.", "output": "Valid"},
            {"input": "No plants grow in water. Lily pads are plants. Therefore, lily pads do not grow in water.", "output": "Valid"},
            # Invalid (logically invalid, premises also violate world knowledge)
            {"input": "No birds can swim. Penguins are birds. Therefore, penguins can swim.", "output": "Invalid"},
            {"input": "All reptiles are warm-blooded. Crocodiles are reptiles. Therefore, crocodiles are not warm-blooded.", "output": "Invalid"},
            {"input": "All spiders have six legs. Tarantulas are not spiders. Therefore, tarantulas do not have six legs.", "output": "Invalid"},
            {"input": "No herbivores eat plants. Cows are herbivores. Therefore, cows eat plants.", "output": "Invalid"},
            {"input": "All mammals live on land. Whales are mammals. Therefore, all whales do not live on land.", "output": "Invalid"},
            {"input": "No insects are beneficial to flowers. Bees help flowers bloom. Therefore, bees are not insects.", "output": "Invalid"},
            {"input": "All fish live in deserts. Salmon are fish. Therefore, no salmon live in deserts.", "output": "Invalid"},
        ],
        "nonce": [
            # Valid
            {"input": "All wugs are blickets. Daxes are wugs. Therefore, daxes are blickets.", "output": "Valid"},
            {"input": "No feps are glurps. All mooks are feps. Therefore, no mooks are glurps.", "output": "Valid"},
            {"input": "All zorbs have trunding. Some quivs are zorbs. Therefore, some quivs have trunding.", "output": "Valid"},
            {"input": "No blems are vortish. Snurfs are blems. Therefore, snurfs are not vortish.", "output": "Valid"},
            {"input": "All plonks are mirfy. Some flurbs are plonks. Therefore, some flurbs are mirfy.", "output": "Valid"},
            {"input": "No crumps can bleeve. All dorfs are crumps. Therefore, no dorfs can bleeve.", "output": "Valid"},
            {"input": "All snarks are wumble. Borogoves are snarks. Therefore, borogoves are wumble.", "output": "Valid"},
            {"input": "No splugs have frimble. Blarks are splugs. Therefore, blarks do not have frimble.", "output": "Valid"},
            # Invalid
            {"input": "All wugs are blickets. Zorbs are blickets. Therefore, zorbs are wugs.", "output": "Invalid"},
            {"input": "All feps are glurps. Mooks are not feps. Therefore, mooks are not glurps.", "output": "Invalid"},
            {"input": "All zorbs have trunding. All quivs have trunding. Therefore, all zorbs are quivs.", "output": "Invalid"},
            {"input": "No blems are vortish. Snurfs are not vortish. Therefore, snurfs are blems.", "output": "Invalid"},
            {"input": "All plonks are mirfy. Some flurbs are mirfy. Therefore, all flurbs are plonks.", "output": "Invalid"},
            {"input": "All grumkins are morble. Some pelzers are grumkins. Therefore, all pelzers are morble.", "output": "Invalid"},
            {"input": "All wortles are zeptic. Blumfs are not wortles. Therefore, blumfs are not zeptic.", "output": "Invalid"},
        ],
    }

    CATEGORY_DEMOS: Dict[str, List[Dict[str, str]]] = {
        "consistent": [
            {"input": "All prime ministers are politicians. Churchill was a prime minister. Therefore, Churchill was a politician.", "output": "Valid"},
            {"input": "All triangles have three sides. A shape has three sides. Therefore, the shape is a triangle.", "output": "Invalid"},
            {"input": "All even numbers are divisible by two. Six is an even number. Therefore, six is divisible by two.", "output": "Valid"},
            {"input": "No amphibians have scales. Frogs are amphibians. Therefore, frogs do not have scales.", "output": "Valid"},
            {"input": "No carnivores are herbivores. Wolves are not herbivores. Therefore, wolves are carnivores.", "output": "Invalid"},
        ],
        "violate": [
            {"input": "All fish live only in saltwater. Trout are fish. Therefore, trout live only in saltwater.", "output": "Valid"},
            {"input": "All flying animals are birds. Bats are flying animals. Therefore, bats are birds.", "output": "Valid"},
            {"input": "All trees lose their leaves. Some oaks lose their leaves. Therefore, all oaks are trees.", "output": "Invalid"},
            {"input": "No countries in Africa speak French. Senegal is a country in Africa. Therefore, Senegal does not speak French.", "output": "Valid"},
            {"input": "All elements are gases at room temperature. Helium is not a gas at room temperature. Therefore, helium is an element.", "output": "Invalid"},
        ],
        "nonce": [
            {"input": "All wugs are blickets. Daxes are wugs. Therefore, daxes are blickets.", "output": "Valid"},
            {"input": "All zorbs have trunding. Some quivs are zorbs. Therefore, some quivs have trunding.", "output": "Valid"},
            {"input": "All wugs are blickets. Zorbs are blickets. Therefore, zorbs are wugs.", "output": "Invalid"},
            {"input": "All feps are glurps. Mooks are not feps. Therefore, mooks are not glurps.", "output": "Invalid"},
            {"input": "No feps are glurps. All mooks are feps. Therefore, no mooks are glurps.", "output": "Valid"},
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
        category = instance.get("category_name", "consistent")
        demos = self.CATEGORY_DEMOS.get(category, [])

        inst_input = instance.get("input", "")
        demos = [d for d in demos if inst_input not in d["input"]][:num_shots]

        prompt = ""
        for d in demos:
            prompt += f"Input: {d['input']}\nOutput: {d['output']}\n\n"

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

        # also break down by validity label
        validity_stats: Dict[str, Dict[str, int]] = {}
        for p, g, item in zip(processed, ground_truth, task_data):
            label = item.get("output", "unknown")
            validity_stats.setdefault(label, {"correct": 0, "total": 0})
            validity_stats[label]["total"] += 1
            if p.lower().strip() == g.lower().strip():
                validity_stats[label]["correct"] += 1

        for label, s in validity_stats.items():
            results[f"accuracy_{label.lower()}"] = s["correct"] / s["total"]

        return results

    def get_ground_truth(self, split: str = "test") -> List[str]:
        rows = self.get_split(split)
        return [str(r.get("output", "")) for r in rows]


def create_syllogism_verify_task(
    category: str = None,
    name: str = "syllogism_verify",
) -> SyllogismVerifyTask:
    """Create a SyllogismVerifyTask, optionally filtered to one category."""
    data = None
    if category and category in SyllogismVerifyTask.CATEGORY_DATA:
        data = [
            {**ex, "category_name": category}
            for ex in SyllogismVerifyTask.CATEGORY_DATA[category]
        ]
        name = f"syllogism_verify:{category}"

    config = TaskConfig(
        name=name,
        description="Syllogism verification — Valid or Invalid (consistent, violate, nonce)",
        data_format="memory",
        in_memory_data=data,
        input_column="input",
        output_column="output",
        evaluation_metrics=["accuracy"],
        metadata={"task_type": "syllogism_verify", "category": category},
    )
    return SyllogismVerifyTask(config)

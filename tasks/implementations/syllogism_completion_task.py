"""Syllogism Completion Task based on Lampinen & Dasgupta (2024) (https://arxiv.org/pdf/2207.07051).
This test deductive reasoning, understanding of quantifiers, and logical operators.
It also requires some exact copying of spans.

Task: Given two premises, provide the conclusion
Format: 
    Input: [major premise] [minor premise] 
    Output: [conclusion]

Example: 
    Input: All men are mortal. Socrates is a man. 
    Output: Socrates is mortal.

Following Lampinen & Dasgupta (2024), we provide three categories that differ on consistency with typicality/world knowledge:
    - consistent
    - violate
    - nonce
"""
from typing import Dict, List, Any

from tasks.base_task import BaseTask, TaskConfig


class SyllogismCompletionTask(BaseTask):
    """Syllogism completion task"""

    TASK_NAME = "syllogism_completion"

    CATEGORY_DATA: Dict[str, List[Dict[str, str]]] = {
        "consistent": [
            {"input": "All mammals are warm-blooded. Dogs are mammals.", "output": "Dogs are warm-blooded."},
            {"input": "No insects have a backbone. All bees are insects.", "output": "No bees have a backbone."},
            {"input": "All carpenters work with wood. Some craftspeople are carpenters.", "output": "Some craftspeople work with wood."},
            {"input": "All birds have feathers. Robins are birds.", "output": "Robins have feathers."},
            {"input": "No reptiles are warm-blooded. Snakes are reptiles.", "output": "Snakes are not warm-blooded."},
            {"input": "All doctors have medical degrees. Some hospital workers are doctors.", "output": "Some hospital workers have medical degrees."},
            {"input": "No plants can move on their own. All roses are plants.", "output": "No roses can move on their own."},
            {"input": "All planets orbit a star. Earth is a planet.", "output": "Earth orbits a star."},
            {"input": "No fish can breathe air. Salmon are fish.", "output": "Salmon cannot breathe air."},
            {"input": "All squares have four equal sides. Some shapes are squares.", "output": "Some shapes have four equal sides."},
            {"input": "No metals are transparent. Gold is a metal.", "output": "Gold is not transparent."},
            {"input": "All vertebrates have a spine. Humans are vertebrates.", "output": "Humans have a spine."},
            {"input": "No unmarried persons are spouses. Some adults are unmarried.", "output": "Some adults are not spouses."},
            {"input": "All chefs cook food. Some restaurant workers are chefs.", "output": "Some restaurant workers cook food."},
            {"input": "No liquids have a fixed shape. Water is a liquid.", "output": "Water does not have a fixed shape."},
        ],
        "violate": [
            {"input": "All birds can fly. Penguins are birds.", "output": "Penguins can fly."},
            {"input": "No mammals live in water. Whales are mammals.", "output": "Whales do not live in water."},
            {"input": "All metals sink in water. Lithium is a metal.", "output": "Lithium sinks in water."},
            {"input": "No insects are beneficial to flowers. Bees are insects.", "output": "Bees are not beneficial to flowers."},
            {"input": "All spiders have six legs. Tarantulas are spiders.", "output": "Tarantulas have six legs."},
            {"input": "All deserts are hot. Antarctica is a desert.", "output": "Antarctica is hot."},
            {"input": "No large animals can jump. Kangaroos are large animals.", "output": "Kangaroos cannot jump."},
            {"input": "All primates lack opposable thumbs. Chimpanzees are primates.", "output": "Chimpanzees lack opposable thumbs."},
            {"input": "No plants grow in water. Lily pads are plants.", "output": "Lily pads do not grow in water."},
            {"input": "All snakes have legs. Cobras are snakes.", "output": "Cobras have legs."},
            {"input": "All liquids are transparent. Milk is a liquid.", "output": "Milk is transparent."},
            {"input": "No herbivores eat fruit. Gorillas are herbivores.", "output": "Gorillas do not eat fruit."},
            {"input": "All oceanic creatures are fish. Dolphins are oceanic creatures.", "output": "Dolphins are fish."},
            {"input": "No nocturnal animals hunt for food. Owls are nocturnal animals.", "output": "Owls do not hunt for food."},
            {"input": "All landlocked countries border an ocean. Mongolia is a landlocked country.", "output": "Mongolia borders an ocean."},
        ],
        "nonce": [
            {"input": "All flups can drov. Cravs are flups.", "output": "Cravs can drov."},
            {"input": "No wugs are blickets. All daxes are wugs.", "output": "No daxes are blickets."},
            {"input": "All zorbs have trunding. Some quivs are zorbs.", "output": "Some quivs have trunding."},
            {"input": "No blems are vortish. Snurfs are blems.", "output": "Snurfs are not vortish."},
            {"input": "All plonks are mirfy. Some flurbs are plonks.", "output": "Some flurbs are mirfy."},
            {"input": "No crumps can bleeve. All dorfs are crumps.", "output": "No dorfs can bleeve."},
            {"input": "All snarks are wumble. Borogoves are snarks.", "output": "Borogoves are wumble."},
            {"input": "No zibbles are trulant. Some quorbs are zibbles.", "output": "Some quorbs are not trulant."},
            {"input": "All mimsy things are outgrabe. Toves are mimsy.", "output": "Toves are outgrabe."},
            {"input": "No splugs have frimble. Blarks are splugs.", "output": "Blarks do not have frimble."},
            {"input": "All grumkins are morble. Some pelzers are grumkins.", "output": "Some pelzers are morble."},
            {"input": "No vexels are plumf. All snorbs are vexels.", "output": "No snorbs are plumf."},
            {"input": "All crombles have lurth. Snoggles are crombles.", "output": "Snoggles have lurth."},
            {"input": "No flurps are brimsy. Some quazzles are flurps.", "output": "Some quazzles are not brimsy."},
            {"input": "All wortles are zeptic. Blumfs are wortles.", "output": "Blumfs are zeptic."},
        ]
    }

    CATEGORY_DEMOS: Dict[str, List[Dict[str, str]]] = {
        "consistent": [
            {"input": "All prime ministers are politicians. Winston Churchill was a prime minister.", "output": "Winston Churchill was a politician."},
            {"input": "No carnivores are herbivores. Tigers are carnivores.", "output": "Tigers are not herbivores."},
            {"input": "All triangles have three sides. Some polygons are triangles.", "output": "Some polygons have three sides."},
            {"input": "All even numbers are divisible by two. Six is an even number.", "output": "Six is divisible by two."},
            {"input": "No amphibians have scales. Frogs are amphibians.", "output": "Frogs do not have scales."},
        ],
        "violate": [
            {"input": "All fish live only in saltwater. Trout are fish.", "output": "Trout live only in saltwater."},
            {"input": "No countries in Africa speak French. Senegal is a country in Africa.", "output": "Senegal does not speak French."},
            {"input": "All flying animals are birds. Bats are flying animals.", "output": "Bats are birds."},
            {"input": "No trees lose their leaves. Oak trees are trees.", "output": "Oak trees do not lose their leaves."},
            {"input": "All elements are gases at room temperature. Iron is an element.", "output": "Iron is a gas at room temperature."},
        ],
        "nonce": [
            {"input": "All wugs are blickets. Daxes are wugs.", "output": "Daxes are blickets."},
            {"input": "No feps are glurps. All mooks are feps.", "output": "No mooks are glurps."},
            {"input": "All zorbs have trunding. Some quivs are zorbs.", "output": "Some quivs have trunding."},
            {"input": "No blems are vortish. Snurfs are blems.", "output": "Snurfs are not vortish."},
            {"input": "All plonks are mirfy. Some flurbs are plonks.", "output": "Some flurbs are mirfy."},
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

        return results

    def get_ground_truth(self, split: str = "test") -> List[str]:
        rows = self.get_split(split)
        return [str(r.get("output", "")) for r in rows]


def create_syllogism_completion_task(
    category: str = None,
    name: str = "syllogism_completion",
) -> SyllogismCompletionTask:
    """Create a SyllogismCompletionTask, optionally filtered to one category."""
    data = None
    if category and category in SyllogismCompletionTask.CATEGORY_DATA:
        data = [
            {**ex, "category_name": category}
            for ex in SyllogismCompletionTask.CATEGORY_DATA[category]
        ]
        name = f"syllogism_completion:{category}"

    config = TaskConfig(
        name=name,
        description="Syllogism completion (consistent, violate, nonce)",
        data_format="memory",
        in_memory_data=data,
        input_column="input",
        output_column="output",
        evaluation_metrics=["accuracy"],
        metadata={"task_type": "syllogism_completion", "category": category},
    )
    return SyllogismCompletionTask(config)
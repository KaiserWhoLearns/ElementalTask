"""GSM8K sub-skill probe tasks.

Each task isolates one linguistic/conceptual sub-skill required for multi-step
math word problems, while deliberately staying far simpler than any GSM8K item.

Design principle: identical minimal template across all items so that a model
can only succeed by understanding the targeted sub-skill, not by pattern-matching
narrative context.  Numbers are small positive integers (≤35); answers are
always exact integers.

Tasks
-----
word_problem_single_step  — NL-wrapped single arithmetic op (add/subtract/multiply/divide)
quantity_comparison        — relational quantities (more_than/fewer_than/twice_as_many/half_as_many)
rate_unit                  — unit-rate schema  ("If A items cost C, how much do B?")
multi_entity_tracking      — two agents, 1–2 transfers, ask final count of one agent

Format (ICL):
    Input: <sentence>
    Output: <integer>
"""

from typing import Dict, List, Any

from tasks.base_task import BaseTask, TaskConfig


# ---------------------------------------------------------------------------
# Task 1: word_problem_single_step
# ---------------------------------------------------------------------------

class WordProblemSingleStep(BaseTask):
    """Single-operation arithmetic presented as a one-sentence word problem."""

    TASK_NAME = "word_problem_single_step"

    CATEGORY_DATA: Dict[str, List[Dict[str, str]]] = {
        "add": [
            {"input": "Tom has 4 marbles. He gets 5 more. How many marbles does he have?", "output": "9"},
            {"input": "Ana has 6 stickers. She gets 3 more. How many stickers does she have?", "output": "9"},
            {"input": "Ben has 7 coins. He gets 4 more. How many coins does he have?", "output": "11"},
            {"input": "Mia has 3 books. She gets 8 more. How many books does she have?", "output": "11"},
            {"input": "Sam has 9 cards. He gets 6 more. How many cards does he have?", "output": "15"},
        ],
        "subtract": [
            {"input": "Lena has 10 apples. She gives away 3. How many apples does she have?", "output": "7"},
            {"input": "Dan has 12 pencils. He gives away 5. How many pencils does he have?", "output": "7"},
            {"input": "Zoe has 15 beads. She gives away 8. How many beads does she have?", "output": "7"},
            {"input": "Kai has 9 oranges. He gives away 4. How many oranges does he have?", "output": "5"},
            {"input": "Eva has 20 blocks. She gives away 7. How many blocks does she have?", "output": "13"},
        ],
        "multiply": [
            {"input": "Jay has 3 bags. Each bag has 4 apples. How many apples does Jay have?", "output": "12"},
            {"input": "Nia has 5 boxes. Each box has 6 pens. How many pens does Nia have?", "output": "30"},
            {"input": "Leo has 2 jars. Each jar has 9 cookies. How many cookies does Leo have?", "output": "18"},
            {"input": "Fay has 4 trays. Each tray has 7 cups. How many cups does Fay have?", "output": "28"},
            {"input": "Ash has 6 shelves. Each shelf has 5 books. How many books does Ash have?", "output": "30"},
        ],
        "divide": [
            {"input": "Kim has 12 candies. She splits them equally into 4 bags. How many candies are in each bag?", "output": "3"},
            {"input": "Max has 15 stickers. He divides them equally among 5 friends. How many stickers does each friend get?", "output": "3"},
            {"input": "Roy has 18 apples. He puts an equal number in 3 baskets. How many apples are in each basket?", "output": "6"},
            {"input": "Lin has 20 beads. She divides them equally into 4 boxes. How many beads are in each box?", "output": "5"},
            {"input": "Pam has 24 marbles. She splits them equally among 6 people. How many marbles does each person get?", "output": "4"},
        ],
    }

    CATEGORY_DEMOS: Dict[str, List[str]] = {
        "add":      [
            "Input: Tom has 5 apples. He gets 3 more. How many apples does he have?\nOutput: 8",
            "Input: Ned has 6 books. He gets 7 more. How many books does he have?\nOutput: 13",
            "Input: Ava has 2 coins. She gets 9 more. How many coins does she have?\nOutput: 11",
            "Input: Sue has 8 cards. She gets 4 more. How many cards does she have?\nOutput: 12",
            "Input: Ash has 1 marble. He gets 6 more. How many marbles does he have?\nOutput: 7",
        ],
        "subtract": [
            "Input: Lisa has 11 coins. She gives away 4. How many coins does she have?\nOutput: 7",
            "Input: Ned has 13 apples. He gives away 6. How many apples does he have?\nOutput: 7",
            "Input: Pam has 8 books. She gives away 3. How many books does she have?\nOutput: 5",
            "Input: Roy has 16 marbles. He gives away 9. How many marbles does he have?\nOutput: 7",
            "Input: Ava has 14 stickers. She gives away 5. How many stickers does she have?\nOutput: 9",
        ],
        "multiply": [
            "Input: Joe has 2 crates. Each crate has 8 oranges. How many oranges does Joe have?\nOutput: 16",
            "Input: Ned has 3 bags. Each bag has 7 stones. How many stones does Ned have?\nOutput: 21",
            "Input: Ava has 5 cups. Each cup has 4 seeds. How many seeds does Ava have?\nOutput: 20",
            "Input: Roy has 4 rows. Each row has 6 tiles. How many tiles does Roy have?\nOutput: 24",
            "Input: Sue has 7 trays. Each tray has 3 eggs. How many eggs does Sue have?\nOutput: 21",
        ],
        "divide":   [
            "Input: Ava has 14 buttons. She splits them equally into 7 jars. How many buttons are in each jar?\nOutput: 2",
            "Input: Ned has 16 apples. He divides them equally among 4 friends. How many apples does each friend get?\nOutput: 4",
            "Input: Pam has 10 coins. She splits them equally into 5 bags. How many coins are in each bag?\nOutput: 2",
            "Input: Roy has 21 cards. He puts an equal number in 7 boxes. How many cards are in each box?\nOutput: 3",
            "Input: Sue has 18 candies. She divides them equally among 6 people. How many candies does each person get?\nOutput: 3",
        ],
    }

    def __init__(self, config: TaskConfig):
        super().__init__(config)

    def _load_data(self):
        import pandas as pd
        if self.config.in_memory_data:
            self.data = pd.DataFrame(self.config.in_memory_data)
            return
        rows = []
        for category, examples in self.CATEGORY_DATA.items():
            for ex in examples:
                rows.append({"input": ex["input"], "output": ex["output"], "category_name": category})
        self.data = pd.DataFrame(rows)

    def build_prompt(self, instance: Dict[str, Any], num_shots: int = 5) -> str:
        category = instance.get("category_name", "add")
        demos = self.CATEGORY_DEMOS.get(category, [])
        inst_input = instance.get("input", "")
        demos = [d for d in demos if inst_input not in d][:num_shots]
        prompt = ""
        for d in demos:
            prompt += d + "\n\n"
        prompt += f"Input: {inst_input}\nOutput:"
        return prompt

    def _extract_number(self, text: str) -> str:
        import re
        match = re.search(r"-?\d+", text.strip())
        return match.group() if match else text.strip()

    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, Any]:
        ground_truth = self.get_ground_truth(split)
        task_data = self.get_split(split)
        if len(predictions) != len(ground_truth):
            raise ValueError(f"Prediction count ({len(predictions)}) != ground truth ({len(ground_truth)})")
        processed = [self._extract_number(self.preprocess_prediction(p)) for p in predictions]
        correct = sum(1 for p, g in zip(processed, ground_truth) if p.strip() == g.strip())
        results: Dict[str, Any] = {"accuracy": correct / len(ground_truth), "correct": correct, "total": len(ground_truth)}
        cat_stats: Dict[str, Dict[str, int]] = {}
        for p, g, item in zip(processed, ground_truth, task_data):
            cat = item.get("category_name", "unknown")
            cat_stats.setdefault(cat, {"correct": 0, "total": 0})
            cat_stats[cat]["total"] += 1
            if p.strip() == g.strip():
                cat_stats[cat]["correct"] += 1
        for cat, s in cat_stats.items():
            results[f"accuracy_{cat}"] = s["correct"] / s["total"]
        return results

    def get_ground_truth(self, split: str = "test") -> List[str]:
        return [str(r.get("output", "")) for r in self.get_split(split)]


def create_word_problem_single_step_task(
    category: str = None,
    name: str = "word_problem_single_step",
) -> WordProblemSingleStep:
    data = None
    if category and category in WordProblemSingleStep.CATEGORY_DATA:
        data = [{**ex, "category_name": category} for ex in WordProblemSingleStep.CATEGORY_DATA[category]]
        name = f"word_problem_single_step:{category}"
    config = TaskConfig(
        name=name,
        description="Single-operation arithmetic word problem",
        data_format="memory",
        in_memory_data=data,
        input_column="input",
        output_column="output",
        evaluation_metrics=["accuracy"],
        metadata={"task_type": "word_problem_single_step", "category": category},
    )
    return WordProblemSingleStep(config)


# ---------------------------------------------------------------------------
# Task 2: quantity_comparison
# ---------------------------------------------------------------------------

class QuantityComparison(BaseTask):
    """Relational quantity: compute X from 'X is N more/fewer/twice/half than Y'."""

    TASK_NAME = "quantity_comparison"

    CATEGORY_DATA: Dict[str, List[Dict[str, str]]] = {
        "more_than": [
            {"input": "Ava has 3 more marbles than Ben. Ben has 5. How many marbles does Ava have?", "output": "8"},
            {"input": "Mia has 6 more stickers than Tim. Tim has 4. How many stickers does Mia have?", "output": "10"},
            {"input": "Leo has 2 more coins than Sue. Sue has 7. How many coins does Leo have?", "output": "9"},
            {"input": "Dan has 8 more cards than Zoe. Zoe has 3. How many cards does Dan have?", "output": "11"},
            {"input": "Kim has 5 more books than Jay. Jay has 9. How many books does Kim have?", "output": "14"},
        ],
        "fewer_than": [
            {"input": "Sam has 4 fewer apples than Kai. Kai has 11. How many apples does Sam have?", "output": "7"},
            {"input": "Fay has 3 fewer pencils than Roy. Roy has 10. How many pencils does Fay have?", "output": "7"},
            {"input": "Ash has 7 fewer beads than Pam. Pam has 15. How many beads does Ash have?", "output": "8"},
            {"input": "Ned has 5 fewer cookies than Lena. Lena has 12. How many cookies does Ned have?", "output": "7"},
            {"input": "Nia has 2 fewer blocks than Eva. Eva has 9. How many blocks does Nia have?", "output": "7"},
        ],
        "twice_as_many": [
            {"input": "Tom has twice as many marbles as Ana. Ana has 4. How many marbles does Tom have?", "output": "8"},
            {"input": "Jay has twice as many stickers as Kim. Kim has 6. How many stickers does Jay have?", "output": "12"},
            {"input": "Max has twice as many coins as Mia. Mia has 5. How many coins does Max have?", "output": "10"},
            {"input": "Lin has twice as many books as Ben. Ben has 7. How many books does Lin have?", "output": "14"},
            {"input": "Sue has twice as many cards as Dan. Dan has 3. How many cards does Sue have?", "output": "6"},
        ],
        "half_as_many": [
            {"input": "Zoe has half as many apples as Leo. Leo has 12. How many apples does Zoe have?", "output": "6"},
            {"input": "Kai has half as many pencils as Ava. Ava has 18. How many pencils does Kai have?", "output": "9"},
            {"input": "Eva has half as many cookies as Roy. Roy has 10. How many cookies does Eva have?", "output": "5"},
            {"input": "Tim has half as many blocks as Fay. Fay has 16. How many blocks does Tim have?", "output": "8"},
            {"input": "Ned has half as many beads as Ash. Ash has 14. How many beads does Ned have?", "output": "7"},
        ],
    }

    CATEGORY_DEMOS: Dict[str, List[str]] = {
        "more_than": [
            "Input: Pam has 4 more marbles than Sam. Sam has 6. How many marbles does Pam have?\nOutput: 10",
            "Input: Max has 7 more cards than Lena. Lena has 4. How many cards does Max have?\nOutput: 11",
            "Input: Sue has 1 more coin than Roy. Roy has 8. How many coins does Sue have?\nOutput: 9",
            "Input: Ash has 9 more books than Nia. Nia has 2. How many books does Ash have?\nOutput: 11",
            "Input: Fay has 3 more beads than Tim. Tim has 5. How many beads does Fay have?\nOutput: 8",
        ],
        "fewer_than": [
            "Input: Nia has 3 fewer coins than Ned. Ned has 9. How many coins does Nia have?\nOutput: 6",
            "Input: Roy has 6 fewer apples than Ash. Ash has 13. How many apples does Roy have?\nOutput: 7",
            "Input: Pam has 1 fewer sticker than Max. Max has 10. How many stickers does Pam have?\nOutput: 9",
            "Input: Sue has 8 fewer blocks than Mia. Mia has 14. How many blocks does Sue have?\nOutput: 6",
            "Input: Tim has 4 fewer pencils than Fay. Fay has 11. How many pencils does Tim have?\nOutput: 7",
        ],
        "twice_as_many": [
            "Input: Lin has twice as many books as Roy. Roy has 5. How many books does Lin have?\nOutput: 10",
            "Input: Ned has twice as many coins as Pam. Pam has 3. How many coins does Ned have?\nOutput: 6",
            "Input: Ash has twice as many apples as Sue. Sue has 8. How many apples does Ash have?\nOutput: 16",
            "Input: Fay has twice as many cards as Tim. Tim has 4. How many cards does Fay have?\nOutput: 8",
            "Input: Roy has twice as many beads as Nia. Nia has 6. How many beads does Roy have?\nOutput: 12",
        ],
        "half_as_many": [
            "Input: Tim has half as many stickers as Eva. Eva has 14. How many stickers does Tim have?\nOutput: 7",
            "Input: Pam has half as many coins as Ned. Ned has 16. How many coins does Pam have?\nOutput: 8",
            "Input: Roy has half as many books as Ash. Ash has 20. How many books does Roy have?\nOutput: 10",
            "Input: Nia has half as many cards as Sue. Sue has 6. How many cards does Nia have?\nOutput: 3",
            "Input: Fay has half as many marbles as Max. Max has 10. How many marbles does Fay have?\nOutput: 5",
        ],
    }

    def __init__(self, config: TaskConfig):
        super().__init__(config)

    def _load_data(self):
        import pandas as pd
        if self.config.in_memory_data:
            self.data = pd.DataFrame(self.config.in_memory_data)
            return
        rows = []
        for category, examples in self.CATEGORY_DATA.items():
            for ex in examples:
                rows.append({"input": ex["input"], "output": ex["output"], "category_name": category})
        self.data = pd.DataFrame(rows)

    def build_prompt(self, instance: Dict[str, Any], num_shots: int = 5) -> str:
        category = instance.get("category_name", "more_than")
        demos = self.CATEGORY_DEMOS.get(category, [])
        inst_input = instance.get("input", "")
        demos = [d for d in demos if inst_input not in d][:num_shots]
        prompt = ""
        for d in demos:
            prompt += d + "\n\n"
        prompt += f"Input: {inst_input}\nOutput:"
        return prompt

    def _extract_number(self, text: str) -> str:
        import re
        match = re.search(r"-?\d+", text.strip())
        return match.group() if match else text.strip()

    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, Any]:
        ground_truth = self.get_ground_truth(split)
        task_data = self.get_split(split)
        if len(predictions) != len(ground_truth):
            raise ValueError(f"Prediction count ({len(predictions)}) != ground truth ({len(ground_truth)})")
        processed = [self._extract_number(self.preprocess_prediction(p)) for p in predictions]
        correct = sum(1 for p, g in zip(processed, ground_truth) if p.strip() == g.strip())
        results: Dict[str, Any] = {"accuracy": correct / len(ground_truth), "correct": correct, "total": len(ground_truth)}
        cat_stats: Dict[str, Dict[str, int]] = {}
        for p, g, item in zip(processed, ground_truth, task_data):
            cat = item.get("category_name", "unknown")
            cat_stats.setdefault(cat, {"correct": 0, "total": 0})
            cat_stats[cat]["total"] += 1
            if p.strip() == g.strip():
                cat_stats[cat]["correct"] += 1
        for cat, s in cat_stats.items():
            results[f"accuracy_{cat}"] = s["correct"] / s["total"]
        return results

    def get_ground_truth(self, split: str = "test") -> List[str]:
        return [str(r.get("output", "")) for r in self.get_split(split)]


def create_quantity_comparison_task(
    category: str = None,
    name: str = "quantity_comparison",
) -> QuantityComparison:
    data = None
    if category and category in QuantityComparison.CATEGORY_DATA:
        data = [{**ex, "category_name": category} for ex in QuantityComparison.CATEGORY_DATA[category]]
        name = f"quantity_comparison:{category}"
    config = TaskConfig(
        name=name,
        description="Relational quantity comparison word problem",
        data_format="memory",
        in_memory_data=data,
        input_column="input",
        output_column="output",
        evaluation_metrics=["accuracy"],
        metadata={"task_type": "quantity_comparison", "category": category},
    )
    return QuantityComparison(config)


# ---------------------------------------------------------------------------
# Task 3: rate_unit
# ---------------------------------------------------------------------------

class RateUnit(BaseTask):
    """Unit-rate schema: given (A items cost C cents), find cost of B items.

    All rates are whole numbers; answers are exact integers.
    Single schema repeated across all 20 items — only the numbers change.
    """

    TASK_NAME = "rate_unit"

    # All items share one schema; no sub-categories needed.
    ITEM_DATA: List[Dict[str, str]] = [
        {"input": "If 2 apples cost 6 cents, how much do 5 apples cost?", "output": "15"},
        {"input": "If 3 pens cost 9 cents, how much do 7 pens cost?", "output": "21"},
        {"input": "If 4 books cost 12 cents, how much do 6 books cost?", "output": "18"},
        {"input": "If 5 oranges cost 10 cents, how much do 8 oranges cost?", "output": "16"},
        {"input": "If 6 cookies cost 18 cents, how much do 4 cookies cost?", "output": "12"},
        {"input": "If 2 cards cost 8 cents, how much do 9 cards cost?", "output": "36"},
        {"input": "If 3 marbles cost 6 cents, how much do 10 marbles cost?", "output": "20"},
        {"input": "If 4 stickers cost 20 cents, how much do 3 stickers cost?", "output": "15"},
        {"input": "If 5 stamps cost 15 cents, how much do 6 stamps cost?", "output": "18"},
        {"input": "If 6 beads cost 24 cents, how much do 2 beads cost?", "output": "8"},
        {"input": "If 2 blocks cost 10 cents, how much do 7 blocks cost?", "output": "35"},
        {"input": "If 3 pencils cost 12 cents, how much do 5 pencils cost?", "output": "20"},
        {"input": "If 4 candies cost 8 cents, how much do 9 candies cost?", "output": "18"},
        {"input": "If 5 clips cost 25 cents, how much do 4 clips cost?", "output": "20"},
        {"input": "If 6 balls cost 12 cents, how much do 7 balls cost?", "output": "14"},
        {"input": "If 2 cups cost 14 cents, how much do 5 cups cost?", "output": "35"},
        {"input": "If 3 eggs cost 9 cents, how much do 8 eggs cost?", "output": "24"},
        {"input": "If 4 buttons cost 16 cents, how much do 6 buttons cost?", "output": "24"},
        {"input": "If 5 pins cost 20 cents, how much do 3 pins cost?", "output": "12"},
        {"input": "If 7 caps cost 21 cents, how much do 5 caps cost?", "output": "15"},
    ]

    DEMOS: List[str] = [
        "Input: If 2 grapes cost 4 cents, how much do 6 grapes cost?\nOutput: 12",
        "Input: If 3 nuts cost 15 cents, how much do 4 nuts cost?\nOutput: 20",
        "Input: If 4 coins cost 12 cents, how much do 5 coins cost?\nOutput: 15",
        "Input: If 5 chips cost 10 cents, how much do 7 chips cost?\nOutput: 14",
        "Input: If 6 rings cost 18 cents, how much do 3 rings cost?\nOutput: 9",
    ]

    def __init__(self, config: TaskConfig):
        super().__init__(config)

    def _load_data(self):
        import pandas as pd
        if self.config.in_memory_data:
            self.data = pd.DataFrame(self.config.in_memory_data)
            return
        self.data = pd.DataFrame(self.ITEM_DATA)

    def build_prompt(self, instance: Dict[str, Any], num_shots: int = 5) -> str:
        inst_input = instance.get("input", "")
        demos = [d for d in self.DEMOS if inst_input not in d][:num_shots]
        prompt = ""
        for d in demos:
            prompt += d + "\n\n"
        prompt += f"Input: {inst_input}\nOutput:"
        return prompt

    def _extract_number(self, text: str) -> str:
        import re
        match = re.search(r"-?\d+", text.strip())
        return match.group() if match else text.strip()

    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, Any]:
        ground_truth = self.get_ground_truth(split)
        if len(predictions) != len(ground_truth):
            raise ValueError(f"Prediction count ({len(predictions)}) != ground truth ({len(ground_truth)})")
        processed = [self._extract_number(self.preprocess_prediction(p)) for p in predictions]
        correct = sum(1 for p, g in zip(processed, ground_truth) if p.strip() == g.strip())
        return {"accuracy": correct / len(ground_truth), "correct": correct, "total": len(ground_truth)}

    def get_ground_truth(self, split: str = "test") -> List[str]:
        return [str(r.get("output", "")) for r in self.get_split(split)]


def create_rate_unit_task(
    category: str = None,        # accepted for API consistency; ignored (no sub-categories)
    name: str = "rate_unit",
) -> RateUnit:
    config = TaskConfig(
        name=name,
        description="Unit-rate cost schema: given (A items cost C), find cost of B",
        data_format="memory",
        in_memory_data=None,
        input_column="input",
        output_column="output",
        evaluation_metrics=["accuracy"],
        metadata={"task_type": "rate_unit"},
    )
    return RateUnit(config)


# ---------------------------------------------------------------------------
# Task 4: multi_entity_tracking
# ---------------------------------------------------------------------------

class MultiEntityTracking(BaseTask):
    """Two-agent state tracking across 1–2 give operations.

    one_transfer: agent A gives k to agent B; ask for A or B's final count.
    two_transfer: two give operations between A and B; ask for one agent's final count.
    """

    TASK_NAME = "multi_entity_tracking"

    CATEGORY_DATA: Dict[str, List[Dict[str, str]]] = {
        "one_transfer": [
            # ask sender (5)
            {"input": "Alice has 8 apples. Bob has 4. Alice gives 3 to Bob. How many apples does Alice have?", "output": "5"},
            {"input": "Tom has 10 coins. Mia has 2. Tom gives 4 to Mia. How many coins does Tom have?", "output": "6"},
            {"input": "Sam has 9 cards. Zoe has 5. Sam gives 3 to Zoe. How many cards does Sam have?", "output": "6"},
            {"input": "Kai has 12 marbles. Ava has 3. Kai gives 5 to Ava. How many marbles does Kai have?", "output": "7"},
            {"input": "Ben has 7 stickers. Lena has 4. Ben gives 2 to Lena. How many stickers does Ben have?", "output": "5"},
            # ask receiver (5)
            {"input": "Dan has 5 books. Eva has 3. Dan gives 4 to Eva. How many books does Eva have?", "output": "7"},
            {"input": "Fay has 6 pencils. Max has 2. Fay gives 3 to Max. How many pencils does Max have?", "output": "5"},
            {"input": "Leo has 8 beads. Kim has 1. Leo gives 5 to Kim. How many beads does Kim have?", "output": "6"},
            {"input": "Ned has 10 blocks. Roy has 4. Ned gives 6 to Roy. How many blocks does Roy have?", "output": "10"},
            {"input": "Lin has 9 cookies. Jay has 2. Lin gives 3 to Jay. How many cookies does Jay have?", "output": "5"},
        ],
        "two_transfer": [
            {"input": "Alice has 10 apples. Bob has 2. Alice gives 4 to Bob. Bob gives 3 to Alice. How many apples does Alice have?", "output": "9"},
            {"input": "Tom has 8 coins. Mia has 6. Tom gives 3 to Mia. Mia gives 2 to Tom. How many coins does Tom have?", "output": "7"},
            {"input": "Sam has 12 cards. Zoe has 3. Sam gives 5 to Zoe. Zoe gives 2 to Sam. How many cards does Sam have?", "output": "9"},
            {"input": "Kai has 9 marbles. Ava has 7. Kai gives 4 to Ava. Ava gives 3 to Kai. How many marbles does Ava have?", "output": "8"},
            {"input": "Ben has 6 stickers. Lena has 8. Ben gives 2 to Lena. Ben gives 1 more to Lena. How many stickers does Ben have?", "output": "3"},
            {"input": "Dan has 11 books. Eva has 4. Eva gives 3 to Dan. Dan gives 5 to Eva. How many books does Dan have?", "output": "9"},
            {"input": "Fay has 7 pencils. Max has 9. Max gives 4 to Fay. Fay gives 3 to Max. How many pencils does Fay have?", "output": "8"},
            {"input": "Leo has 8 beads. Kim has 6. Leo gives 3 to Kim. Kim gives 5 to Leo. How many beads does Leo have?", "output": "10"},
            {"input": "Ned has 10 blocks. Roy has 5. Ned gives 6 to Roy. Roy gives 2 to Ned. How many blocks does Roy have?", "output": "9"},
            {"input": "Lin has 9 cookies. Jay has 4. Jay gives 3 to Lin. Lin gives 5 to Jay. How many cookies does Lin have?", "output": "7"},
        ],
    }

    CATEGORY_DEMOS: Dict[str, List[str]] = {
        "one_transfer": [
            "Input: Nia has 7 apples. Tim has 3. Nia gives 2 to Tim. How many apples does Nia have?\nOutput: 5",
            "Input: Ash has 8 coins. Pam has 4. Ash gives 3 to Pam. How many coins does Pam have?\nOutput: 7",
            "Input: Roy has 6 marbles. Ava has 5. Roy gives 4 to Ava. How many marbles does Roy have?\nOutput: 2",
            "Input: Mia has 10 cards. Ned has 1. Mia gives 4 to Ned. How many cards does Ned have?\nOutput: 5",
            "Input: Sue has 9 stickers. Kai has 2. Sue gives 6 to Kai. How many stickers does Sue have?\nOutput: 3",
        ],
        "two_transfer": [
            "Input: Sue has 10 cards. Ned has 2. Sue gives 4 to Ned. Ned gives 1 to Sue. How many cards does Sue have?\nOutput: 7",
            "Input: Roy has 6 marbles. Ava has 5. Roy gives 3 to Ava. Ava gives 2 to Roy. How many marbles does Roy have?\nOutput: 5",
            "Input: Kai has 9 stickers. Mia has 3. Kai gives 4 to Mia. Mia gives 2 to Kai. How many stickers does Kai have?\nOutput: 7",
            "Input: Ash has 8 coins. Pam has 4. Ash gives 5 to Pam. Pam gives 3 to Ash. How many coins does Pam have?\nOutput: 6",
            "Input: Nia has 11 apples. Tim has 3. Tim gives 2 to Nia. Nia gives 4 to Tim. How many apples does Nia have?\nOutput: 9",
        ],
    }

    def __init__(self, config: TaskConfig):
        super().__init__(config)

    def _load_data(self):
        import pandas as pd
        if self.config.in_memory_data:
            self.data = pd.DataFrame(self.config.in_memory_data)
            return
        rows = []
        for category, examples in self.CATEGORY_DATA.items():
            for ex in examples:
                rows.append({"input": ex["input"], "output": ex["output"], "category_name": category})
        self.data = pd.DataFrame(rows)

    def build_prompt(self, instance: Dict[str, Any], num_shots: int = 5) -> str:
        category = instance.get("category_name", "one_transfer")
        demos = self.CATEGORY_DEMOS.get(category, [])
        inst_input = instance.get("input", "")
        demos = [d for d in demos if inst_input not in d][:num_shots]
        prompt = ""
        for d in demos:
            prompt += d + "\n\n"
        prompt += f"Input: {inst_input}\nOutput:"
        return prompt

    def _extract_number(self, text: str) -> str:
        import re
        match = re.search(r"-?\d+", text.strip())
        return match.group() if match else text.strip()

    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, Any]:
        ground_truth = self.get_ground_truth(split)
        task_data = self.get_split(split)
        if len(predictions) != len(ground_truth):
            raise ValueError(f"Prediction count ({len(predictions)}) != ground truth ({len(ground_truth)})")
        processed = [self._extract_number(self.preprocess_prediction(p)) for p in predictions]
        correct = sum(1 for p, g in zip(processed, ground_truth) if p.strip() == g.strip())
        results: Dict[str, Any] = {"accuracy": correct / len(ground_truth), "correct": correct, "total": len(ground_truth)}
        cat_stats: Dict[str, Dict[str, int]] = {}
        for p, g, item in zip(processed, ground_truth, task_data):
            cat = item.get("category_name", "unknown")
            cat_stats.setdefault(cat, {"correct": 0, "total": 0})
            cat_stats[cat]["total"] += 1
            if p.strip() == g.strip():
                cat_stats[cat]["correct"] += 1
        for cat, s in cat_stats.items():
            results[f"accuracy_{cat}"] = s["correct"] / s["total"]
        return results

    def get_ground_truth(self, split: str = "test") -> List[str]:
        return [str(r.get("output", "")) for r in self.get_split(split)]


def create_multi_entity_tracking_task(
    category: str = None,
    name: str = "multi_entity_tracking",
) -> MultiEntityTracking:
    data = None
    if category and category in MultiEntityTracking.CATEGORY_DATA:
        data = [{**ex, "category_name": category} for ex in MultiEntityTracking.CATEGORY_DATA[category]]
        name = f"multi_entity_tracking:{category}"
    config = TaskConfig(
        name=name,
        description="Two-agent state tracking across give operations",
        data_format="memory",
        in_memory_data=data,
        input_column="input",
        output_column="output",
        evaluation_metrics=["accuracy"],
        metadata={"task_type": "multi_entity_tracking", "category": category},
    )
    return MultiEntityTracking(config)

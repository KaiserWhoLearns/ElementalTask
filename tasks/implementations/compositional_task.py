"""Compositional task that chains multiple atomic operations."""

import random
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional

from tasks.base_task import BaseTask, TaskConfig


# =============================================================================
# Atomic Operations Registry
# =============================================================================

# Pure string operations (no external data needed)
STRING_OPERATIONS: Dict[str, Callable[[str], str]] = {
    "uppercase": lambda x: x.upper(),
    "lowercase": lambda x: x.lower(),
    "reverse": lambda x: x[::-1],
    "first_letter": lambda x: x[0] if x else "",
    "last_letter": lambda x: x[-1] if x else "",
}


def load_lookup_tables() -> Dict[str, Dict[str, str]]:
    """Load lookup-based operations from simple.csv.
    
    Returns a dictionary mapping operation names to their lookup tables.
    """
    csv_path = Path(__file__).parent.parent.parent / "dataset" / "simple.csv"
    if not csv_path.exists():
        return {}
    
    df = pd.read_csv(csv_path)
    
    # Categories that can be used as lookup operations
    lookup_categories = [
        "translate_eng_fr", "translate_fr_eng",
        "translate_eng_sp", "translate_sp_eng",
        "present_to_gerund", "singular_to_plural",
    ]
    
    lookup_tables = {}
    for category in lookup_categories:
        cat_data = df[df["category_name"] == category]
        if not cat_data.empty:
            lookup_tables[category] = dict(zip(cat_data["question"], cat_data["answer"]))
    
    return lookup_tables


# Global lookup tables (loaded once)
LOOKUP_TABLES: Dict[str, Dict[str, str]] = {}


def get_lookup_operation(op_name: str) -> Optional[Callable[[str], str]]:
    """Get a lookup-based operation function."""
    global LOOKUP_TABLES
    if not LOOKUP_TABLES:
        LOOKUP_TABLES = load_lookup_tables()
    
    if op_name in LOOKUP_TABLES:
        table = LOOKUP_TABLES[op_name]
        return lambda x: table.get(x, x)  # Return original if not found
    return None


def get_operation(op_name: str) -> Callable[[str], str]:
    """Get an operation function by name (string or lookup-based)."""
    if op_name in STRING_OPERATIONS:
        return STRING_OPERATIONS[op_name]
    
    lookup_op = get_lookup_operation(op_name)
    if lookup_op:
        return lookup_op
    
    raise ValueError(f"Unknown operation: {op_name}")


def apply_composition(input_str: str, operations: List[str]) -> str:
    """Apply a sequence of operations to an input string."""
    result = input_str
    for op_name in operations:
        op_func = get_operation(op_name)
        result = op_func(result)
    return result


def parse_operations(operations_str: str) -> List[str]:
    """Parse operations string like 'uppercase+reverse' into list."""
    return operations_str.split("+")


# =============================================================================
# Predefined Compositions
# =============================================================================

# Pure string compositions (any input works)
STRING_COMPOSITIONS = {
    # 2-operation: case + manipulation
    "upper_reverse": ["uppercase", "reverse"],
    "lower_reverse": ["lowercase", "reverse"],
    "reverse_upper": ["reverse", "uppercase"],
    "reverse_lower": ["reverse", "lowercase"],
    "upper_first": ["uppercase", "first_letter"],
    "lower_first": ["lowercase", "first_letter"],
    "upper_last": ["uppercase", "last_letter"],
    "lower_last": ["lowercase", "last_letter"],
    "reverse_first": ["reverse", "first_letter"],
    "reverse_last": ["reverse", "last_letter"],
    "first_upper": ["first_letter", "uppercase"],
    "last_upper": ["last_letter", "uppercase"],
    
    # 3-operation chains
    "upper_reverse_first": ["uppercase", "reverse", "first_letter"],
    "lower_reverse_first": ["lowercase", "reverse", "first_letter"],
    "upper_reverse_last": ["uppercase", "reverse", "last_letter"],
    "lower_reverse_last": ["lowercase", "reverse", "last_letter"],
    "reverse_upper_first": ["reverse", "uppercase", "first_letter"],
    "reverse_lower_first": ["reverse", "lowercase", "first_letter"],
}

# Lookup-based compositions (require specific input domains)
# Format: (composition_name, operations, source_lookup_table)
# The source_lookup_table determines valid inputs
LOOKUP_COMPOSITIONS = {
    # Translation eng->fr + string ops (complete coverage)
    "translate_eng_fr_upper": (["translate_eng_fr", "uppercase"], "translate_eng_fr"),
    "translate_eng_fr_lower": (["translate_eng_fr", "lowercase"], "translate_eng_fr"),
    "translate_eng_fr_reverse": (["translate_eng_fr", "reverse"], "translate_eng_fr"),
    "translate_eng_fr_first": (["translate_eng_fr", "first_letter"], "translate_eng_fr"),
    "translate_eng_fr_last": (["translate_eng_fr", "last_letter"], "translate_eng_fr"),
    
    # Translation eng->sp + string ops (complete coverage)
    "translate_eng_sp_upper": (["translate_eng_sp", "uppercase"], "translate_eng_sp"),
    "translate_eng_sp_lower": (["translate_eng_sp", "lowercase"], "translate_eng_sp"),
    "translate_eng_sp_reverse": (["translate_eng_sp", "reverse"], "translate_eng_sp"),
    "translate_eng_sp_first": (["translate_eng_sp", "first_letter"], "translate_eng_sp"),
    "translate_eng_sp_last": (["translate_eng_sp", "last_letter"], "translate_eng_sp"),
    
    # Translation fr->eng + string ops (complete coverage)
    "translate_fr_eng_upper": (["translate_fr_eng", "uppercase"], "translate_fr_eng"),
    "translate_fr_eng_lower": (["translate_fr_eng", "lowercase"], "translate_fr_eng"),
    "translate_fr_eng_reverse": (["translate_fr_eng", "reverse"], "translate_fr_eng"),
    "translate_fr_eng_first": (["translate_fr_eng", "first_letter"], "translate_fr_eng"),
    "translate_fr_eng_last": (["translate_fr_eng", "last_letter"], "translate_fr_eng"),
    
    # Translation sp->eng + string ops (complete coverage)
    "translate_sp_eng_upper": (["translate_sp_eng", "uppercase"], "translate_sp_eng"),
    "translate_sp_eng_lower": (["translate_sp_eng", "lowercase"], "translate_sp_eng"),
    "translate_sp_eng_reverse": (["translate_sp_eng", "reverse"], "translate_sp_eng"),
    "translate_sp_eng_first": (["translate_sp_eng", "first_letter"], "translate_sp_eng"),
    "translate_sp_eng_last": (["translate_sp_eng", "last_letter"], "translate_sp_eng"),
    
    # Morphological + string ops (complete coverage)
    "gerund_upper": (["present_to_gerund", "uppercase"], "present_to_gerund"),
    "gerund_lower": (["present_to_gerund", "lowercase"], "present_to_gerund"),
    "gerund_reverse": (["present_to_gerund", "reverse"], "present_to_gerund"),
    "gerund_first": (["present_to_gerund", "first_letter"], "present_to_gerund"),
    "gerund_last": (["present_to_gerund", "last_letter"], "present_to_gerund"),
    "plural_upper": (["singular_to_plural", "uppercase"], "singular_to_plural"),
    "plural_lower": (["singular_to_plural", "lowercase"], "singular_to_plural"),
    "plural_reverse": (["singular_to_plural", "reverse"], "singular_to_plural"),
    "plural_first": (["singular_to_plural", "first_letter"], "singular_to_plural"),
    "plural_last": (["singular_to_plural", "last_letter"], "singular_to_plural"),
    
    # 3-operation chains with lookup
    "gerund_upper_reverse": (["present_to_gerund", "uppercase", "reverse"], "present_to_gerund"),
    "gerund_reverse_first": (["present_to_gerund", "reverse", "first_letter"], "present_to_gerund"),
    "plural_upper_reverse": (["singular_to_plural", "uppercase", "reverse"], "singular_to_plural"),
    "plural_reverse_first": (["singular_to_plural", "reverse", "first_letter"], "singular_to_plural"),
    "translate_eng_fr_upper_reverse": (["translate_eng_fr", "uppercase", "reverse"], "translate_eng_fr"),
    "translate_eng_sp_upper_reverse": (["translate_eng_sp", "uppercase", "reverse"], "translate_eng_sp"),
}


# =============================================================================
# Input Pools
# =============================================================================

# Generic strings for pure string compositions
STRING_INPUT_POOL = [
    "hello", "world", "python", "apple", "banana", "cherry",
    "delta", "echo", "foxtrot", "golf", "hotel", "india",
    "juliet", "kilo", "lima", "mike", "november", "oscar",
    "papa", "quebec", "romeo", "sierra", "tango", "uniform",
    "victor", "whiskey", "xray", "yankee", "zulu", "alpha",
    "bravo", "charlie", "example", "testing", "compose", "chain",
]

# Operations that benefit from character spacing
SPACING_BENEFITS = {"reverse", "first_letter", "last_letter"}


def add_spaces(s: str) -> str:
    """Add spaces between each character."""
    return " ".join(list(s))


def remove_spaces(s: str) -> str:
    """Remove spaces from a string."""
    return s.replace(" ", "")


def composition_benefits_from_spacing(operations: List[str]) -> bool:
    """Check if a composition would benefit from character spacing."""
    return any(op in SPACING_BENEFITS for op in operations)


def get_lookup_inputs(lookup_name: str) -> List[str]:
    """Get valid inputs for a lookup-based operation."""
    global LOOKUP_TABLES
    if not LOOKUP_TABLES:
        LOOKUP_TABLES = load_lookup_tables()
    
    if lookup_name in LOOKUP_TABLES:
        return list(LOOKUP_TABLES[lookup_name].keys())
    return []


# =============================================================================
# Compositional Task Implementation
# =============================================================================

class CompositionalTask(BaseTask):
    """A task that chains multiple atomic operations.
    
    This task supports:
    - Loading from CSV file (dataset/compositional.csv)
    - Auto-generating data if CSV doesn't exist
    - Subtask filtering via category_name (e.g., compositional:upper_reverse)
    - Spaced mode for character-level operations (spaced=True)
    
    Examples:
        # Load all compositional tasks
        task = get_task("compositional")
        
        # Load specific composition
        task = get_task("compositional:upper_reverse")
        
        # Load with spacing for character-level operations
        task = get_task("compositional:upper_reverse", spaced=True)
    """
    
    TASK_NAME = "compositional"  # Auto-registration name
    
    def __init__(self, config: TaskConfig, spaced: bool = False):
        """Initialize compositional task.
        
        Args:
            config: Task configuration
            spaced: If True, add spaces between characters in input/output
        """
        self.spaced = spaced
        super().__init__(config)
    
    def _load_data(self):
        """Load compositional task data from CSV or generate it."""
        # Use spaced CSV if in spaced mode
        if self.spaced:
            data_path = Path(__file__).parent.parent.parent / "dataset" / "compositional_spaced.csv"
        else:
            data_path = Path(__file__).parent.parent.parent / "dataset" / "compositional.csv"
        
        if data_path.exists():
            df = pd.read_csv(data_path)
            df = df.fillna("")
            self.data = df.to_dict("records")
        else:
            # Generate data if CSV doesn't exist
            self._generate_data()
            # Optionally save to CSV for future runs
            self._save_data(data_path)
    
    def _generate_data(self):
        """Generate compositional examples programmatically."""
        examples = []
        
        # Generate examples for pure string compositions
        for input_str in STRING_INPUT_POOL:
            for comp_name, ops in STRING_COMPOSITIONS.items():
                try:
                    output = apply_composition(input_str, ops)
                    
                    if self.spaced:
                        examples.append({
                            "input": add_spaces(input_str),
                            "output": add_spaces(output),
                            "original_input": input_str,
                            "original_output": output,
                            "category_name": comp_name,
                            "operations": "+".join(ops),
                            "spaced": True,
                        })
                    else:
                        examples.append({
                            "input": input_str,
                            "output": output,
                            "category_name": comp_name,
                            "operations": "+".join(ops),
                        })
                except Exception as e:
                    print(f"Warning: Failed to generate {comp_name} for '{input_str}': {e}")
                    continue
        
        # Generate examples for lookup-based compositions
        for comp_name, (ops, source_lookup) in LOOKUP_COMPOSITIONS.items():
            valid_inputs = get_lookup_inputs(source_lookup)
            for input_str in valid_inputs:
                try:
                    output = apply_composition(input_str, ops)
                    # Skip if lookup returned original (meaning lookup failed)
                    if ops[0] in LOOKUP_TABLES and output == input_str:
                        continue
                    
                    if self.spaced:
                        examples.append({
                            "input": add_spaces(input_str),
                            "output": add_spaces(output),
                            "original_input": input_str,
                            "original_output": output,
                            "category_name": comp_name,
                            "operations": "+".join(ops),
                            "spaced": True,
                        })
                    else:
                        examples.append({
                            "input": input_str,
                            "output": output,
                            "category_name": comp_name,
                            "operations": "+".join(ops),
                        })
                except Exception as e:
                    print(f"Warning: Failed to generate {comp_name} for '{input_str}': {e}")
                    continue
        
        self.data = examples
    
    def _save_data(self, path: Path):
        """Save generated data to CSV for reproducibility."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.data)
        df.to_csv(path, index=False)
        print(f"Saved compositional data to {path}")
    
    def build_prompt(self, instance: Dict[str, Any], num_shots: int = 5) -> str:
        """Build ICL prompt with demonstrations from same category.
        
        Args:
            instance: Test instance with 'input', 'output', 'category_name'
            num_shots: Number of demonstration examples
            
        Returns:
            Formatted prompt string
        """
        category = instance.get("category_name", "")
        
        # Get demos from same category, excluding test instance
        demos = [
            ex for ex in self.data 
            if ex.get("category_name") == category and ex["input"] != instance["input"]
        ]
        random.shuffle(demos)
        demos = demos[:num_shots]
        
        # Build prompt with simple arrow format (like simple_icl)
        prompt_parts = []
        
        # Add demonstrations
        for demo in demos:
            prompt_parts.append(f"{demo['input']} -> {demo['output']}")
        
        # Add test instance (without answer)
        prompt_parts.append(f"{instance['input']} ->")
        
        return "\n".join(prompt_parts)
    
    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """Evaluate predictions against ground truth.
        
        Args:
            predictions: List of model predictions
            split: Data split to evaluate on
            
        Returns:
            Dictionary with evaluation metrics
        """
        ground_truth = self.get_ground_truth(split)
        task_data = self.get_split(split)
        
        if len(predictions) != len(ground_truth):
            raise ValueError(f"Prediction count ({len(predictions)}) doesn't match ground truth count ({len(ground_truth)})")
        
        # Preprocess predictions
        processed_predictions = []
        for pred in predictions:
            # Clean prediction: take first line, strip whitespace
            pred_clean = pred.strip().split("\n")[0].strip()
            
            # Remove leading arrow if present
            if pred_clean.startswith("->"):
                pred_clean = pred_clean[2:].strip()
            
            # For spaced mode, keep spaces; otherwise take first word
            if self.spaced:
                # Keep the full spaced output
                processed_predictions.append(pred_clean)
            else:
                # Take first word only
                pred_clean = pred_clean.split()[0] if pred_clean.split() else pred_clean
                processed_predictions.append(pred_clean)
        
        def matches(pred: str, gt: str) -> bool:
            """Check if prediction matches ground truth, handling spaced mode."""
            pred_norm = pred.lower().strip()
            gt_norm = gt.lower().strip()
            
            if pred_norm == gt_norm:
                return True
            
            # For spaced mode, also try comparing without spaces
            if self.spaced:
                pred_unspaced = remove_spaces(pred_norm)
                gt_unspaced = remove_spaces(gt_norm)
                return pred_unspaced == gt_unspaced
            
            return False
        
        # Overall accuracy
        correct = sum(1 for pred, gt in zip(processed_predictions, ground_truth) 
                     if matches(pred, gt))
        accuracy = correct / len(ground_truth)
        
        results = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(ground_truth)
        }
        
        # Per-category accuracy
        category_stats = {}
        for pred, gt, item in zip(processed_predictions, ground_truth, task_data):
            category = item.get("category_name", "unknown")
            if category not in category_stats:
                category_stats[category] = {"correct": 0, "total": 0}
            
            category_stats[category]["total"] += 1
            if matches(pred, gt):
                category_stats[category]["correct"] += 1
        
        # Calculate per-category accuracy
        for category, stats in category_stats.items():
            results[f"accuracy_{category}"] = stats["correct"] / stats["total"]
            results[f"correct_{category}"] = stats["correct"]
            results[f"total_{category}"] = stats["total"]
        
        return results
    
    @property
    def task_type(self) -> str:
        return "compositional_spaced" if self.spaced else "compositional"
    
    @property
    def description(self) -> str:
        return "Compositional task that chains multiple atomic string operations"


# =============================================================================
# Utility Functions
# =============================================================================

def generate_compositional_csv(output_path: str = None, spaced: bool = False):
    """Generate the compositional.csv or compositional_spaced.csv file.
    
    Args:
        output_path: Path to save CSV. Defaults to dataset/compositional.csv or compositional_spaced.csv
        spaced: If True, generate spaced version with spaces between characters
    """
    if output_path is None:
        if spaced:
            output_path = Path(__file__).parent.parent.parent / "dataset" / "compositional_spaced.csv"
        else:
            output_path = Path(__file__).parent.parent.parent / "dataset" / "compositional.csv"
    else:
        output_path = Path(output_path)
    
    # Ensure lookup tables are loaded
    global LOOKUP_TABLES
    if not LOOKUP_TABLES:
        LOOKUP_TABLES = load_lookup_tables()
    
    examples = []
    
    # Generate pure string compositions
    for input_str in STRING_INPUT_POOL:
        for comp_name, ops in STRING_COMPOSITIONS.items():
            try:
                output = apply_composition(input_str, ops)
                
                if spaced:
                    examples.append({
                        "input": add_spaces(input_str),
                        "output": add_spaces(output),
                        "original_input": input_str,
                        "original_output": output,
                        "category_name": comp_name,
                        "operations": "+".join(ops),
                        "spaced": True,
                    })
                else:
                    examples.append({
                        "input": input_str,
                        "output": output,
                        "category_name": comp_name,
                        "operations": "+".join(ops),
                    })
            except Exception:
                continue
    
    # Generate lookup-based compositions
    for comp_name, (ops, source_lookup) in LOOKUP_COMPOSITIONS.items():
        valid_inputs = get_lookup_inputs(source_lookup)
        for input_str in valid_inputs:
            try:
                output = apply_composition(input_str, ops)
                
                if spaced:
                    examples.append({
                        "input": add_spaces(input_str),
                        "output": add_spaces(output),
                        "original_input": input_str,
                        "original_output": output,
                        "category_name": comp_name,
                        "operations": "+".join(ops),
                        "spaced": True,
                    })
                else:
                    examples.append({
                        "input": input_str,
                        "output": output,
                        "category_name": comp_name,
                        "operations": "+".join(ops),
                    })
            except Exception:
                continue
    
    df = pd.DataFrame(examples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Summary
    string_comps = len(STRING_COMPOSITIONS)
    lookup_comps = len(LOOKUP_COMPOSITIONS)
    total_comps = string_comps + lookup_comps
    
    mode = "spaced" if spaced else "normal"
    print(f"Generated {len(examples)} {mode} examples across {total_comps} compositions")
    print(f"  - {string_comps} string compositions (pure string ops)")
    print(f"  - {lookup_comps} lookup compositions (translation/morphological + string ops)")
    print(f"Saved to: {output_path}")
    return df


if __name__ == "__main__":
    import sys
    # Generate both normal and spaced CSV files
    if len(sys.argv) > 1 and sys.argv[1] == "--spaced":
        generate_compositional_csv(spaced=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--all":
        generate_compositional_csv(spaced=False)
        generate_compositional_csv(spaced=True)
    else:
        generate_compositional_csv(spaced=False)

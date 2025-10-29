"""String Analogy Task based on Melanie Mitchell's work.
The default ICL examples are taken from: https://melaniemitchell.me/ExplorationsContent/analogy-problems.html

Format: source -> target, query -> ?
Task: Identify the transformation from source to target, then apply it to query.

Example: abc -> abd, ijk -> ?
Answer: ijl (replace last letter with successor)
"""

import random
import string
from typing import Dict, List, Optional, Tuple
from tasks.base_task import BaseTask, TaskConfig


class StringAnalogyTask(BaseTask):
    """
    String analogy task for testing analogical reasoning.
    
    Format: source -> target, query -> ?
    The model must identify the transformation and apply it to the query.
    """
    TASK_NAME = "string_analogy"
    
    def __init__(self, config: TaskConfig, use_generator: bool = False):
        """
        Initialize StringAnalogyTask.
        
        Args:
            config: TaskConfig with task settings
            use_generator: If True, generate synthetic examples for ICL
        """
        self.use_generator = use_generator
        super().__init__(config)
    
    def _successor(self, char: str) -> str:
        """Get the next letter in alphabet (wraps z->a)."""
        if char.isalpha():
            if char == 'z':
                return 'a'
            elif char == 'Z':
                return 'A'
            return chr(ord(char) + 1)
        return char
    
    def _predecessor(self, char: str) -> str:
        """Get the previous letter in alphabet (wraps a->z)."""
        if char.isalpha():
            if char == 'a':
                return 'z'
            elif char == 'A':
                return 'Z'
            return chr(ord(char) - 1)
        return char
    
    def _apply_transformation(self, transformation: str, query: str) -> str:
        """Apply a transformation rule to a query string."""
        
        if transformation == "successor_last":
            # Replace last character with successor
            if not query:
                return query
            return query[:-1] + self._successor(query[-1])
        
        elif transformation == "successor_all":
            # Replace all characters with successors
            return ''.join(self._successor(c) for c in query)
        
        elif transformation == "predecessor_last":
            # Replace last character with predecessor
            if not query:
                return query
            return query[:-1] + self._predecessor(query[-1])
        
        elif transformation == "reverse":
            # Reverse the string
            return query[::-1]
        
        elif transformation == "double_string":
            # Double the entire string
            return query + query
        
        elif transformation == "double_each":
            # Double each character
            return ''.join(c * 2 for c in query)
        
        elif transformation == "delete_middle":
            # Delete middle character(s)
            if len(query) <= 1:
                return query
            mid = len(query) // 2
            if len(query) % 2 == 1:
                return query[:mid] + query[mid+1:]
            else:
                return query[:mid-1] + query[mid+1:]
        
        elif transformation == "delete_first":
            # Delete first character
            return query[1:] if len(query) > 0 else query
        
        elif transformation == "delete_last":
            # Delete last character
            return query[:-1] if len(query) > 0 else query
        
        elif transformation == "append_successor":
            # Append the successor of the last character
            if not query:
                return query
            return query + self._successor(query[-1])
        
        elif transformation == "insert_x_middle":
            # Insert 'x' in the middle
            mid = len(query) // 2
            return query[:mid] + 'x' + query[mid:]
        
        elif transformation == "swap_halves":
            # Swap first and second half
            mid = len(query) // 2
            return query[mid:] + query[:mid]
        
        elif transformation == "delete_duplicates":
            # Remove consecutive duplicates
            if not query:
                return query
            result = [query[0]]
            for c in query[1:]:
                if c != result[-1]:
                    result.append(c)
            return ''.join(result)
        
        else:
            # Unknown transformation, return query unchanged
            return query
    
    def generate_analogy_example(
        self, 
        transformation: str,
        source_length: int = 3,
        seed: Optional[int] = None
    ) -> Dict[str, str]:
        """Generate a synthetic analogy example with a given transformation.
        
        For ICL format, this generates just a single input->output pair showing
        the transformation. Multiple examples of the same transformation type
        should be used together for proper ICL demonstration.
        
        Args:
            transformation: Type of transformation to apply
            source_length: Length of the source string
            seed: Random seed for reproducibility
            
        Returns:
            Dict with input, output, and transformation fields
        """
        if seed is not None:
            random.seed(seed)
        
        # Generate random input string
        input_str = ''.join(random.choices(string.ascii_lowercase, k=source_length))
        
        # Apply transformation to get output
        output_str = self._apply_transformation(transformation, input_str)
        
        return {
            "input": input_str,
            "output": output_str,
            "transformation": transformation
        }
    
    def generate_examples(
        self,
        num_examples: int,
        transformations: Optional[List[str]] = None,
        seed: Optional[int] = None,
        same_transformation: bool = True
    ) -> List[Dict[str, str]]:
        """Generate multiple analogy examples.
        
        Args:
            num_examples: Number of examples to generate
            transformations: List of transformation types to use (if None, use all)
            seed: Random seed for reproducibility
            same_transformation: If True, all examples use the same transformation type
                                (proper ICL format). If False, mix different transformations.
            
        Returns:
            List of example dictionaries
        """
        if seed is not None:
            random.seed(seed)
        
        if transformations is None:
            transformations = [
                "successor_last", "successor_all", "predecessor_last",
                "reverse", "double_string", "double_each",
                "delete_middle", "delete_first", "delete_last",
                "append_successor", "swap_halves"
            ]
        
        examples = []
        
        if same_transformation:
            # Pick one transformation and generate all examples with it
            trans = random.choice(transformations)
            for i in range(num_examples):
                length = random.randint(2, 4)
                example = self.generate_analogy_example(trans, length, seed=seed+i if seed else None)
                examples.append(example)
        else:
            # Mix different transformations
            for i in range(num_examples):
                trans = random.choice(transformations)
                length = random.randint(2, 4)
                example = self.generate_analogy_example(trans, length, seed=seed+i if seed else None)
                examples.append(example)
        
        return examples
    
    def get_icl_examples(
        self,
        num_examples: int = 10,
        shuffle: bool = True,
        seed: Optional[int] = None,
        fresh: bool = True,
        transformations: Optional[List[str]] = None,
        same_transformation: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Return ICL-formatted examples.
        
        If use_generator=True, generates synthetic examples with the same transformation.
        Otherwise, uses the standard BaseTask logic with stored data.
        
        Args:
            num_examples: Number of examples to return
            shuffle: Whether to shuffle (ignored if use_generator=True)
            seed: Random seed for reproducibility
            fresh: Whether to prefer unused examples (ignored if use_generator=True)
            transformations: List of transformation types for generator mode
            same_transformation: If True, all generated examples use the same transformation
                                (recommended for proper ICL format)
            
        Returns:
            List of example dictionaries with 'input' and 'output' keys
        """
        if self.use_generator:
            # Generate fresh examples on-the-fly
            # Use same_transformation=True by default for proper ICL
            examples = self.generate_examples(
                num_examples, 
                transformations, 
                seed,
                same_transformation=same_transformation
            )
            return examples
        else:
            # Use the standard BaseTask logic with stored data
            return super().get_icl_examples(
                num_examples=num_examples,
                shuffle=shuffle,
                seed=seed,
                fresh=fresh
            )
    
    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """Evaluate predictions with exact match accuracy."""
        ground_truth = self.get_ground_truth(split)
        
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Prediction count ({len(predictions)}) doesn't match "
                f"ground truth count ({len(ground_truth)})"
            )
        
        # Preprocess predictions
        processed_predictions = [self.preprocess_prediction(pred) for pred in predictions]
        
        # Calculate exact match accuracy
        correct = sum(
            1 for pred, gt in zip(processed_predictions, ground_truth)
            if pred.strip().lower() == gt.strip().lower()
        )
        accuracy = correct / len(ground_truth) if ground_truth else 0.0
        
        results = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(ground_truth)
        }
        
        # Add per-transformation accuracy if available
        data = self.get_split(split)
        if any("transformation" in item for item in data):
            trans_results = {}
            for pred, gt, item in zip(processed_predictions, ground_truth, data):
                trans = item.get("transformation", "unknown")
                if trans not in trans_results:
                    trans_results[trans] = {"correct": 0, "total": 0}
                
                trans_results[trans]["total"] += 1
                if pred.strip().lower() == gt.strip().lower():
                    trans_results[trans]["correct"] += 1
            
            # Calculate per-transformation accuracy
            for trans, stats in trans_results.items():
                results[f"accuracy_{trans}"] = stats["correct"] / stats["total"]
        
        return results


def make_string_analogy_task(
    config: TaskConfig = None,
    use_generator: bool = False,
    use_fixed_examples: bool = True
) -> StringAnalogyTask:
    """Factory function to create a StringAnalogyTask.
    
    Args:
        config: Optional TaskConfig. If None, creates default config.
        use_generator: If True, generate synthetic examples for ICL.
        use_fixed_examples: If True and config is None, use Mitchell's curated examples.
    
    Returns:
        StringAnalogyTask instance
    """
    if config is None:
        if use_fixed_examples:
            # Melanie Mitchell's curated string analogy examples
            # Format: source -> target, query -> answer
            mitchell_data = [
                {"input": "abc -> abd, ijk -> ?", "output": "ijl", "source": "abc", "target": "abd", "query": "ijk", "answer": "ijl", "transformation": "successor_last"},
                {"input": "abc -> abd, xyz -> ?", "output": "xya", "source": "abc", "target": "abd", "query": "xyz", "answer": "xya", "transformation": "successor_last"},
                {"input": "abc -> abd, mrrjjj -> ?", "output": "mrrjjk", "source": "abc", "target": "abd", "query": "mrrjjj", "answer": "mrrjjk", "transformation": "successor_last"},
                {"input": "abc -> ac, pqr -> ?", "output": "pr", "source": "abc", "target": "ac", "query": "pqr", "answer": "pr", "transformation": "delete_middle"},
                {"input": "abc -> abcd, pqr -> ?", "output": "pqrs", "source": "abc", "target": "abcd", "query": "pqr", "answer": "pqrs", "transformation": "append_successor"},
                {"input": "a -> b, c -> ?", "output": "d", "source": "a", "target": "b", "query": "c", "answer": "d", "transformation": "successor_all"},
                {"input": "a -> aa, b -> ?", "output": "bb", "source": "a", "target": "aa", "query": "b", "answer": "bb", "transformation": "double_string"},
                {"input": "a -> ab, z -> ?", "output": "za", "source": "a", "target": "ab", "query": "z", "answer": "za", "transformation": "append_successor"},
                {"input": "pqr -> rqp, abc -> ?", "output": "cba", "source": "pqr", "target": "rqp", "query": "abc", "answer": "cba", "transformation": "reverse"},
                {"input": "abc -> abd, abcabc -> ?", "output": "abdabd", "source": "abc", "target": "abd", "query": "abcabc", "answer": "abdabd", "transformation": "successor_last"},
            ]
            
            config = TaskConfig(
                name="string_analogy",
                description="String analogy task (Melanie Mitchell): identify transformation and apply to query",
                data_format="memory",
                in_memory_data=mitchell_data,
                input_column="input",
                output_column="output",
                num_demonstrations=5,
            )
        else:
            # Minimal default for generator mode
            config = TaskConfig(
                name="string_analogy",
                description="String analogy task with generated examples",
                data_format="memory",
                in_memory_data=[{"input": "abc -> abd, ijk -> ?", "output": "ijl"}],
                num_demonstrations=5,
            )
    
    return StringAnalogyTask(config, use_generator=use_generator)

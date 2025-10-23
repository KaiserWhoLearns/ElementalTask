"""Copying task for testing induction heads.

This task presents examples where input equals output, testing the model's
ability to perform exact copying - a key capability of induction heads.

Two modes of operation:

1. **Static mode** (use_generator=False):
   - Uses a fixed set of examples (default: 10 simple words/strings)
   - ICL examples are sampled from this static set
   - Good for reproducibility and consistency
   
2. **Generator mode** (use_generator=True):
   - Generates unlimited random character sequences on-the-fly
   - Each call to get_icl_examples() produces fresh random strings
   - Configurable length range and character set
   - Perfect for testing with diverse inputs without data limitations

Example usage:
    # Static mode
    task = make_copying_task(use_generator=False)
    examples = task.get_icl_examples(num_examples=5)
    
    # Generator mode with custom parameters
    task = make_copying_task(use_generator=True, min_length=3, max_length=8)
    examples = task.get_icl_examples(num_examples=10, charset="abc123", seed=42)
"""

import random
import string
from tasks.base_task import BaseTask, TaskConfig
from typing import Dict, List, Iterator


class CopyingTask(BaseTask):
    """
    A task where the output is an exact copy of the input.
    This tests induction head capabilities in language models.
    """
    TASK_NAME = "copying"
    
    def __init__(self, config: TaskConfig, use_generator: bool = False, 
                 min_length: int = 3, max_length: int = 8):
        """
        Initialize the CopyingTask.
        
        Args:
            config: TaskConfig with task settings
            use_generator: If True, generate random examples on-the-fly instead of using static data
            min_length: Minimum length of generated random strings (default: 3)
            max_length: Maximum length of generated random strings (default: 8)
        """
        self.use_generator = use_generator
        self.min_length = min_length
        self.max_length = max_length
        super().__init__(config)
    
    def generate_random_example(self, length: int = None, 
                               charset: str = None, 
                               seed: int = None) -> Dict[str, str]:
        """
        Generate a random copying example.
        
        Args:
            length: Length of the random string. If None, randomly chosen between min_length and max_length
            charset: Character set to use. If None, uses letters and digits
            seed: Random seed for reproducibility
            
        Returns:
            Dict with 'input' and 'output' keys (both identical)
        """
        if seed is not None:
            random.seed(seed)
        
        if charset is None:
            charset = string.ascii_letters + string.digits
        
        if length is None:
            length = random.randint(self.min_length, self.max_length)
        
        random_str = ''.join(random.choices(charset, k=length))
        return {"input": random_str, "output": random_str}
    
    def generate_examples(self, num_examples: int, 
                         charset: str = None,
                         seed: int = None) -> List[Dict[str, str]]:
        """
        Generate multiple random copying examples.
        
        Args:
            num_examples: Number of examples to generate
            charset: Character set to use for generation
            seed: Random seed for reproducibility
            
        Returns:
            List of example dictionaries
        """
        if seed is not None:
            random.seed(seed)
        
        examples = []
        for _ in range(num_examples):
            examples.append(self.generate_random_example(charset=charset))
        return examples
    
    def get_icl_examples(
        self,
        num_examples: int = 10,
        shuffle: bool = True,
        seed: int = None,
        fresh: bool = True,
        charset: str = None,
    ) -> List[Dict[str, str]]:
        """
        Return ICL-formatted examples.
        
        If use_generator=True, generates fresh random examples each time.
        Otherwise, uses the standard BaseTask logic with stored data.
        
        Args:
            num_examples: Number of examples to return
            shuffle: Whether to shuffle (ignored if use_generator=True)
            seed: Random seed for reproducibility
            fresh: Whether to prefer unused examples (ignored if use_generator=True)
            charset: Character set for random generation (only used if use_generator=True)
            
        Returns:
            List of example dictionaries with 'input' and 'output' keys
        """
        if self.use_generator:
            # Generate fresh random examples on-the-fly
            return self.generate_examples(num_examples, charset=charset, seed=seed)
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
            if pred.strip() == gt.strip()
        )
        accuracy = correct / len(ground_truth) if ground_truth else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(ground_truth)
        }


def make_copying_task(config: TaskConfig = None, 
                      use_generator: bool = False,
                      min_length: int = 3,
                      max_length: int = 8) -> CopyingTask:
    """Factory function to create a CopyingTask with default examples.
    
    Args:
        config: Optional TaskConfig. If None, creates a default config with
                example data where input == output.
        use_generator: If True, generate random examples on-the-fly instead of using static data.
                      This allows for limitless unique examples.
        min_length: Minimum length of generated random strings (only used if use_generator=True)
        max_length: Maximum length of generated random strings (only used if use_generator=True)
    
    Returns:
        CopyingTask instance
    """
    if config is None:
        if use_generator:
            # For generator mode, provide minimal static data (won't be used for ICL)
            default_examples = [
                {"input": "test", "output": "test"},
            ]
        else:
            # Default examples - simple strings that should be copied exactly
            default_examples = [
                {"input": "cat", "output": "cat"},
                {"input": "dog", "output": "dog"},
                {"input": "hello", "output": "hello"},
                {"input": "world", "output": "world"},
                {"input": "apple", "output": "apple"},
                {"input": "banana", "output": "banana"},
                {"input": "123", "output": "123"},
                {"input": "xyz", "output": "xyz"},
                {"input": "test", "output": "test"},
                {"input": "copy", "output": "copy"},
            ]
        
        config = TaskConfig(
            name="copying",
            description="Task where output exactly copies the input (induction heads)",
            data_format="memory",
            in_memory_data=default_examples,
            num_demonstrations=5,
        )
    
    return CopyingTask(config, use_generator=use_generator, 
                      min_length=min_length, max_length=max_length)

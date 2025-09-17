"""Simple ICL task implementation with category-based demonstrations."""

from typing import Dict, List, Any
import pandas as pd
from ..base_task import BaseTask, TaskConfig


class SimpleICLTask(BaseTask):
    """Task for simple in-context learning with category-based demonstrations."""
    
    TASK_NAME = "simple_icl"  # Auto-registration name
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        
        # Define hardcoded demonstrations for each category
        self.category_demonstrations = {
            "uppercase": ["a -> A", "c -> C"],
            "lowercase": ["A -> a", "C -> c"],
            "first_letter": ["the cat went up the tree -> t", "elephants are cool -> e"],
            "last_letter": ["the cat went up the tree -> e", "elephants are cool -> l"],
            "translate_eng_fr": ["hello -> bonjour", "goodbye -> au revoir"],
            "translate_fr_eng": ["bonjour -> hello", "au revoir -> goodbye"],
            "translate_eng_sp": ["hello -> hola", "goodbye -> adiós"],
            "translate_sp_eng": ["hola -> hello", "adiós -> goodbye"],
            "present_to_gerund": ["run -> running", "swim -> swimming"],
            "singular_to_plural": ["cat -> cats", "dog -> dogs"],
            "country_to_capital": ["France -> Paris", "Germany -> Berlin"],
            "country_to_currency": ["France -> Euro", "United States -> Dollar"],
        }
    
    def build_prompt(self, instance: Dict[str, Any]) -> str:
        """Build a prompt with category-specific demonstrations."""
        category = instance.get('category_name', '')
        
        # Check if we have demonstrations in the config (from in-memory data)
        if (self.config.in_memory_demonstrations and 
            isinstance(self.config.in_memory_demonstrations, dict)):
            demos = self.config.in_memory_demonstrations.get(category, [])
        else:
            # Fallback to hardcoded demonstrations
            demos = self.category_demonstrations.get(category, [])
        
        prompt = ""
        if demos:
            for demo in demos:
                prompt += f"{demo}\n"
        
        # Add the current question
        prompt += f"{instance[self.config.input_column]} ->"
        
        return prompt
    
    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """Evaluate predictions with exact match accuracy, including per-category metrics."""
        ground_truth = self.get_ground_truth(split)
        task_data = self.get_split(split)
        
        if len(predictions) != len(ground_truth):
            raise ValueError(f"Prediction count ({len(predictions)}) doesn't match ground truth count ({len(ground_truth)})")
        
        # Preprocess predictions
        processed_predictions = [self.preprocess_prediction(pred) for pred in predictions]
        
        # Overall accuracy
        correct = sum(1 for pred, gt in zip(processed_predictions, ground_truth) 
                     if pred.lower().strip() == gt.lower().strip())
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
            if pred.lower().strip() == gt.lower().strip():
                category_stats[category]["correct"] += 1
        
        # Calculate per-category accuracy
        for category, stats in category_stats.items():
            results[f"accuracy_{category}"] = stats["correct"] / stats["total"]
            results[f"correct_{category}"] = stats["correct"]
            results[f"total_{category}"] = stats["total"]
        
        return results

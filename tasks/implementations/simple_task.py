"""Simple task implementation with basic exact match evaluation."""

from typing import Dict, List, Any
from ..base_task import BaseTask, TaskConfig


class SimpleTask(BaseTask):
    """A simple task implementation with basic exact match evaluation."""
    
    TASK_NAME = "simple"  # Auto-registration name
    
    def _load_data(self):
        """Load data from CSV file."""
        import pandas as pd
        
        if self.config.data_path:
            self.data = pd.read_csv(self.config.data_path)
        else:
            # If no data path, create empty DataFrame
            self.data = pd.DataFrame(columns=[self.config.input_column, self.config.output_column])
    
    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """Evaluate predictions with exact match accuracy."""
        ground_truth = self.get_ground_truth(split)
        
        if len(predictions) != len(ground_truth):
            raise ValueError(f"Prediction count ({len(predictions)}) doesn't match ground truth count ({len(ground_truth)})")
        
        # Preprocess predictions
        processed_predictions = [self.preprocess_prediction(pred) for pred in predictions]
        
        # Calculate exact match accuracy
        correct = sum(1 for pred, gt in zip(processed_predictions, ground_truth) 
                     if pred.lower().strip() == gt.lower().strip())
        accuracy = correct / len(ground_truth)
        
        results = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(ground_truth)
        }
        
        # Add per-category accuracy if available
        data = self.get_split(split)
        if any("category" in item for item in data):
            category_results = {}
            for pred, gt, item in zip(processed_predictions, ground_truth, data):
                category = item.get("category", "unknown")
                if category not in category_results:
                    category_results[category] = {"correct": 0, "total": 0}
                
                category_results[category]["total"] += 1
                if pred.lower().strip() == gt.lower().strip():
                    category_results[category]["correct"] += 1
            
            # Calculate per-category accuracy
            for category, stats in category_results.items():
                results[f"accuracy_{category}"] = stats["correct"] / stats["total"]
        
        return results

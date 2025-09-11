"""Example of a custom task with automatic discovery."""

from ..base_task import BaseTask, TaskConfig
import re
from typing import List, Dict


class MathTask(BaseTask):
    """A task for simple math problems."""
    
    TASK_NAME = "math"  # This is all that's needed for automatic registration!
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
    
    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """Evaluate math predictions."""
        correct = 0
        total = len(predictions)
        
        if total == 0:
            return {"accuracy": 0.0}
        
        # Get the targets for this split
        data_split = self.data if split == "test" else self.data  # For now, use same data
        targets = data_split[self.config.output_column].tolist()
        
        if len(predictions) != len(targets):
            raise ValueError(f"Number of predictions ({len(predictions)}) doesn't match targets ({len(targets)})")
        
        detailed_results = []
        for pred, target in zip(predictions, targets):
            result = self.evaluate_response(pred, str(target))
            if result["correct"]:
                correct += 1
            detailed_results.append(result)
        
        accuracy = correct / total
        return {
            "accuracy": accuracy,
            "correct_count": correct,
            "total_count": total,
            "detailed_results": detailed_results
        }
    
    def evaluate_response(self, response: str, target: str) -> dict:
        """Evaluate a math response with number extraction."""
        
        def extract_number(text: str) -> float:
            """Extract the first number from text."""
            # Look for numbers (including decimals)
            match = re.search(r'-?\d+\.?\d*', text.strip())
            if match:
                return float(match.group())
            return None
        
        response_num = extract_number(response)
        target_num = extract_number(target)
        
        if response_num is None or target_num is None:
            return {
                "correct": False,
                "error": "Could not extract numbers",
                "response_extracted": response_num,
                "target_extracted": target_num
            }
        
        # Check if they're equal (with small tolerance for floating point)
        correct = abs(response_num - target_num) < 0.001
        
        return {
            "correct": correct,
            "response_extracted": response_num,
            "target_extracted": target_num,
            "original_response": response,
            "original_target": target
        }
    
    def generate_prompt(self, row: dict, demonstrations: list = None) -> str:
        """Generate a prompt for math problems."""
        prompt = "Solve the following math problem:\n\n"
        
        if demonstrations:
            prompt += "Here are some examples:\n"
            for demo in demonstrations:
                prompt += f"Problem: {demo['input']}\nAnswer: {demo['target']}\n\n"
            prompt += "Now solve this problem:\n"
        
        prompt += f"Problem: {row[self.config.input_column]}\nAnswer:"
        return prompt

"""Basic arithmetic task with simple addition/subtraction/multiplication/division."""

from typing import Dict, List, Any
import re

from tasks.base_task import BaseTask, TaskConfig


class BasicArithmeticTask(BaseTask):
    """Task for basic arithmetic operations."""
    
    TASK_NAME = "basic_arithmetic"  # Auto-discovery key
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
    
    def _load_data(self):
        """Load arithmetic problems - uses in-memory data if provided, otherwise defaults."""
        import pandas as pd
        
        if self.config.in_memory_data:
            self.data = pd.DataFrame(self.config.in_memory_data)
            return
        
        # Default arithmetic problems
        default_problems = [
            {"question": "What is 5 + 3?", "answer": "8"},
            {"question": "What is 12 - 7?", "answer": "5"},
            {"question": "What is 4 × 6?", "answer": "24"},
            {"question": "What is 15 ÷ 3?", "answer": "5"},
            {"question": "What is 2 + 2?", "answer": "4"},
            {"question": "What is 10 - 4?", "answer": "6"},
            {"question": "What is 3 × 7?", "answer": "21"},
            {"question": "What is 20 ÷ 4?", "answer": "5"},
            {"question": "What is 8 + 9?", "answer": "17"},
            {"question": "What is 25 - 13?", "answer": "12"}
        ]
        
        self.data = pd.DataFrame(default_problems)
    
    def get_split(self, split: str = "test") -> List[Dict[str, Any]]:
        # Convert DataFrame to list of dictionaries for iteration
        if hasattr(self.data, 'to_dict'):
            return self.data.to_dict('records')
        return self.data
    
    def build_prompt(self, instance: Dict[str, Any], num_shots: int = 5) -> str:
        """Build prompt for arithmetic problems with optional ICL examples.
        
        Args:
            instance: The instance to build a prompt for
            num_shots: Number of in-context learning examples (default: 5)
        
        Returns:
            Formatted prompt string
        """
        prompt = ""
        
        # Add ICL examples if requested and we have demonstrations
        if num_shots > 0 and self.demonstrations:
            # Use the first num_shots demonstrations
            prompt += ""
            for demo in self.demonstrations[:num_shots]:
                prompt += f"{demo}\n"
            prompt += "\n"
        
        # Add instruction and current problem
        question = instance[self.config.input_column]
        prompt += f"Input: {question}\nOutput:"
        
        return prompt
    
    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """Evaluate arithmetic predictions by extracting numbers."""
        data = self.get_split(split)
        
        if len(predictions) != len(data):
            return {
                'accuracy': 0.0,
                'error': f'Prediction count ({len(predictions)}) does not match data count ({len(data)})'
            }
        
        correct = 0
        total = len(predictions)
        detailed_results = []
        
        for pred, example in zip(predictions, data):
            expected = example[self.config.output_column]
            
            # Extract number from prediction (handle various formats)
            pred_number = self._extract_number(pred)
            expected_number = self._extract_number(expected)
            
            is_correct = pred_number is not None and pred_number == expected_number
            
            if is_correct:
                correct += 1
            
            detailed_results.append({
                'question': example[self.config.input_column],
                'prediction': pred.strip(),
                'expected': expected,
                'pred_number': pred_number,
                'expected_number': expected_number,
                'correct': is_correct
            })
        
        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'correct': correct,
            'total': total,
            'detailed_results': detailed_results
        }
    
    def _extract_number(self, text: str) -> float:
        """Extract the first number from text."""
        if not text:
            return None
        
        # Try to find integers or floats in the text
        numbers = re.findall(r'-?\d+\.?\d*', str(text).strip())
        
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                return None
        
        return None


def create_basic_arithmetic_task(
    problems: List[Dict[str, str]] = None,
    name: str = "basic_arithmetic"
) -> BasicArithmeticTask:
    """Create a BasicArithmeticTask instance."""
    # Define static demonstrations (separate from test set)
    demonstrations = [
        "Input: What is 7 + 5?\nOutput: 12",
        "Input: What is 18 - 9?\nOutput: 9",
        "Input: What is 6 - 4?\nOutput: 2",
        "Input: What is 16 + 2?\nOutput: 18",
        "Input: What is 11 + 3?\nOutput: 14",
    ]
    
    config = TaskConfig(
        name=name,
        description="Basic arithmetic evaluation task",
        data_format="memory",
        in_memory_data=problems,  # Use provided problems or None for defaults
        in_memory_demonstrations=demonstrations,  # Static ICL examples
        input_column="question",
        output_column="answer",
        evaluation_metrics=["accuracy"],
        metadata={"task_type": "arithmetic"}
    )
    
    return BasicArithmeticTask(config)
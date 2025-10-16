"""Indirect object identification (pronoun resolution) task for identifying correct referents in context using mib-bench/ioi dataset."""

from typing import Dict, List, Any
import re

from tasks.base_task import BaseTask, TaskConfig
from datasets import load_dataset


class IOITask(BaseTask):
    """Task for indirect object identification using IOI dataset from mib-bench/ioi."""

    TASK_NAME = "ioi_task"  # Auto-discovery key
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
    pass
    
    def _load_data(self):
        """Load the IOI (Indirect Object Identification) dataset from mib-bench."""
        self.data = load_dataset("mib-bench/ioi")

    def get_icl_examples(self, num_examples: int = 10, shuffle: bool = True, seed: int = None, fresh: bool = True) -> List[Dict[str, str]]:
        """Return ICL examples for IOI-style data using the test split."""
        data = self.get_split("test")
        if not data:
            return []

        indices = list(range(len(data)))
        if shuffle:
            import random
            if seed is not None:
                random.seed(seed)
            random.shuffle(indices)

        selected = indices[:min(num_examples, len(indices))]
        examples = []
        for i in selected:
            example = data[i]
            prompt = example.get('prompt', '')
            choices = example.get('choices', [])
            answer_index = example.get('answer_index', 0)
            output = choices[answer_index] if choices and 0 <= answer_index < len(choices) else example.get('answer', '')
            examples.append({"input": prompt, "output": str(output)})
        return examples

    def get_split(self, split: str = "test") -> List[Dict[str, Any]]:
        """Get data split for evaluation."""
        if split in self.data:
            split_data = self.data[split]
            # Convert to list of dictionaries
            return [dict(example) for example in split_data]
        elif hasattr(self.data, 'to_dict'):
            return self.data.to_dict('records')
        return self.data if isinstance(self.data, list) else []
    
    def build_prompt(self, instance: Dict[str, Any]) -> str:
        """Build prompt for pronoun resolution problems."""
        # The mib-bench/ioi dataset typically has these fields:
        # - 'prompt': the incomplete sentence
        # - 'choices': list of possible completions 
        # - 'answer_index': index of correct choice
        
        prompt_text = instance.get('prompt', '')
        
        # For IOI, we want the model to complete the sentence with the correct name
        # The prompt should be the incomplete sentence ending with the preposition
        prompt = f"""Complete the following sentence by identifying who should be referenced:

Sentence: {prompt_text}

Answer with only the name of the person who should be referenced."""
        
        return prompt
    
    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """Evaluate indirect object identification predictions."""
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
            # Get the correct answer from the choices and answer_index
            choices = example.get('choices', [])
            answer_index = example.get('answer_index', 0)
            
            if choices and 0 <= answer_index < len(choices):
                expected = choices[answer_index]
            else:
                # Fallback to other possible answer fields
                expected = example.get('answer', example.get('correct_answer', ''))
            
            # Extract name from prediction and expected answer
            pred_name = self._extract_name(pred)
            expected_name = self._extract_name(expected)
            
            # Check if prediction matches expected answer
            is_correct = (pred_name is not None and 
                         expected_name is not None and 
                         pred_name.lower() == expected_name.lower())
            
            if is_correct:
                correct += 1
            
            detailed_results.append({
                'prompt': example.get('prompt', ''),
                'prediction': pred.strip(),
                'expected': expected,
                'pred_name': pred_name,
                'expected_name': expected_name,
                'correct': is_correct,
                'choices': choices,
                'answer_index': answer_index
            })
        
        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'correct': correct,
            'total': total,
            'detailed_results': detailed_results
        }
    
    def _extract_name(self, text: str) -> str:
        """Extract the name from text, handling various response formats."""
        if not text:
            return None
        
        text = text.strip()
        
        # Common patterns in responses
        patterns = [
            r'^([A-Z][a-z]+)(?:\s|$)',  # Name at start
            r'(?:is|answer|name)?\s*:?\s*([A-Z][a-z]+)',  # After common prefixes
            r'([A-Z][a-z]+)(?:\s*\.?\s*$)',  # Name at end
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # Fallback: look for any capitalized word that could be a name
        words = text.split()
        for word in words:
            # Remove punctuation and check if it's a likely name
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word[0].isupper() and clean_word.isalpha():
                # Additional check: common English names are typically 3+ characters
                if len(clean_word) >= 3:
                    return clean_word
        
        return None


def create_ioi_task(
    name: str = "ioi_task"
) -> IOITask:
    """Create a IOITask instance using mib-bench/ioi dataset."""
    config = TaskConfig(
        name=name,
        description="Indirect Object Identification (IOI) evaluation task using mib-bench/ioi dataset",
        data_format="huggingface",
        dataset_name="mib-bench/ioi",
        input_column="prompt",  # The incomplete sentence
        output_column="choices",  # List of choices, use with answer_index
        evaluation_metrics=["accuracy"],
        metadata={
            "task_type": "pronoun_resolution", 
            "requires_reasoning": True,
            "dataset_source": "mib-bench/ioi",
            "task_description": "Indirect Object Identification (IOI) - complete sentences by identifying the correct indirect object"
        }
    )

    return IOITask(config)
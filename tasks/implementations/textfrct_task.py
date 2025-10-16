"""TextFRCT task implementation that integrates with the existing dataset utilities."""

from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd

from ..base_task import BaseTask, TaskConfig


class TextFRCTTask(BaseTask):
        """Task wrapper for the TextFRCT dataset."""
        
        TASK_NAME = "textfrct"  # Auto-registration name
        
        def __init__(self, config: TaskConfig, skip_subjective: bool = False, categories: Optional[List[str]] = None):
            self.skip_subjective = skip_subjective
            self.categories = categories
            super().__init__(config)
        
        def _load_data(self):
            """Load TextFRCT data and optionally filter by categories and subjective tasks."""
            data = pd.read_csv(self.config.data_path)
            
            # Filter by categories if specified
            if self.categories:
                data = data[data['category_id'].isin(self.categories)]
                print(f"Filtered to categories {self.categories}: {len(data)} examples")
            
            # Filter out subjective tasks if requested
            if self.skip_subjective:
                subjective_mask = data['answer'].astype(str).str.contains('<LLMEval>', na=False)
                data = data[~subjective_mask]
                print(f"Filtered out {subjective_mask.sum()} subjective tasks, {len(data)} objective tasks remaining")
            
            # Convert to list of dictionaries and assign to self.data
            self.data = data.to_dict('records')
        
        def get_split(self, split: str = "test") -> List[Dict[str, Any]]:
            """Get data split for TextFRCT."""
            return self.data

    # use BaseTask.get_icl_examples for standard TextFRCT records (input_column/output_column configured)
        
        def build_prompt(self, instance: Dict[str, Any]) -> str:
            """Build prompt based on category type."""
            category = instance['category_id']
            question = instance['question']
            category_name = instance.get('category_name', category)
            
            if category.startswith('CV'):  # Convergent Visual
                if category == 'CV1':  # Scrambled Words
                    return f"Unscramble each group of letters to form a common English word. Use all the letters in each group. Respond with only the word.\n\nInput: {question}\nOutput:"
                elif category == 'CV2':  # Hidden Words
                    return f"Find all the hidden words in the following string of letters. Words are spelled forwards and are at least 4 letters long. List them separated by semicolons.\n\nInput: {question}\nOutput:"
                elif category == 'CV3':  # Incomplete Words
                    return f"Complete the word by filling in the missing letters.\n\nInput: {question}\nOutput:"
            
            elif category.startswith('FA'):  # Fluent Associational
                if category == 'FA1':  # Controlled Association
                    return f"List words that are related to or associated with '{question}'. Separate multiple answers with semicolons."
                elif category == 'FA2':  # Opposites
                    return f"List words that have the opposite meaning of '{question}'. Separate multiple answers with semicolons."
            
            elif category.startswith('V'):  # Vocabulary
                choices = instance.get('choice', '').split(';;') if instance.get('choice') else []
                if choices:
                    choice_text = '\n'.join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
                    return f"Choose the best definition for '{question}':\n\n{choice_text}\n\nAnswer (number):"
                else:
                    return f"What does '{question}' mean?"
            
            elif category.startswith('RG'):  # Reasoning
                return f"Solve this problem: {question}\nAnswer:"
            
            # Default format for other categories
            return f"Task: {category_name}\nQuestion: {question}\nAnswer:"
        
        def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
            """Evaluate predictions based on category type."""
            data = self.get_split(split)
            if len(predictions) != len(data):
                return {
                    'accuracy': 0.0,
                    'error': f'Prediction count ({len(predictions)}) does not match data count ({len(data)})'
                }
            
            correct = 0
            total = len(predictions)
            category_stats = {}
            
            for i, (pred, example) in enumerate(zip(predictions, data)):
                expected = example['answer']
                category = example['category_id']
                
                # Initialize category stats
                if category not in category_stats:
                    category_stats[category] = {'correct': 0, 'total': 0}
                
                is_correct = self._is_correct(pred, expected, category)
                if is_correct:
                    correct += 1
                    category_stats[category]['correct'] += 1
                category_stats[category]['total'] += 1
            
            # Build results
            results = {
                'accuracy': correct / total if total > 0 else 0.0,
                'correct': correct,
                'total': total
            }
            
            # Add per-category results
            for category, stats in category_stats.items():
                cat_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
                results[f'accuracy_{category}'] = cat_accuracy
                results[f'correct_{category}'] = stats['correct'] 
                results[f'total_{category}'] = stats['total']
            
            return results
        
        def _is_correct(self, prediction: str, expected: str, category: str) -> bool:
            """Check if prediction is correct based on category."""
            pred_clean = prediction.strip().lower()
            expected_clean = str(expected).strip().lower()
            
            # Skip subjective tasks marked with <LLMEval>
            if '<llmeval>' in expected_clean:
                return False
            
            # Handle multiple correct answers separated by ;;
            if ';;' in expected_clean:
                correct_answers = [ans.strip().lower() for ans in expected_clean.split(';;')]
                return pred_clean in correct_answers
            
            # For vocabulary tests, check if first character matches (multiple choice)
            if category.startswith('V') and len(pred_clean) == 1:
                try:
                    answer_num = int(expected_clean)
                    return pred_clean == str(answer_num)
                except ValueError:
                    pass
            
            return pred_clean == expected_clean
        
        def get_ground_truth(self, split: str = "test") -> List[str]:
            """Get ground truth for TextFRCT."""
            data = self.get_split(split)
            return [str(example['answer']) for example in data]


def create_textfrct_task(
    data_path: str = "dataset/TextFRCT.csv",
    skip_subjective: bool = False,
    categories: Optional[List[str]] = None,
    name: str = "textfrct"
) -> 'TextFRCTTask':
    """Create a TextFRCT task instance."""
    # Update name to reflect filtering
    if categories:
        name = f"textfrct_{'_'.join(categories)}"
    
    config = TaskConfig(
        name=name,
        description=f"TextFRCT evaluation dataset{' (filtered categories)' if categories else ''}",
        data_path=data_path,
        data_format="csv",
        input_column="question",
        output_column="answer",
        evaluation_metrics=["accuracy"],
        metadata={
            "skip_subjective": skip_subjective,
            "categories": categories
        }
    )
    
    return TextFRCTTask(config, skip_subjective=skip_subjective, categories=categories)

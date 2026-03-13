"""TextFRCT task implementation that integrates with the existing dataset utilities."""

from typing import Dict, List, Any, Optional
from pathlib import Path
import random
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
            
            # Fill NaN values to avoid Arrow conversion issues
            # Replace NaN with empty string for string columns
            data = data.fillna('')
            
            # Convert to list of dictionaries and assign to self.data
            self.data = data.to_dict('records')
        
        def get_split(self, split: str = "test") -> List[Dict[str, Any]]:
            """Get data split for TextFRCT."""
            return self.data

    # use BaseTask.get_icl_examples for standard TextFRCT records (input_column/output_column configured)
        
        def _format_category_prompt(self, instance: Dict[str, Any]) -> str:
            """Format a single TextFRCT instance as a query-style prompt block.

            Prompt formats (BEFORE → AFTER for key fixes):

            RG1/RG2/RG3 — multiple-choice arithmetic/reasoning
              BEFORE: "Solve this problem: {q}\\nAnswer:"
              AFTER:  "Solve this problem: {q}\\n\\nA. 4:30\\nB. 5:00\\n...\\nAnswer (letter):"

            MA2/MA3 — object/name ↔ number lookup (table in `additional`)
              BEFORE: "Task: ...\\nQuestion: coat\\nAnswer:"   (table never shown)
              AFTER:  "tree: 58\\nfloor: 29\\n...\\n\\nQuestion: What number corresponds to 'coat'?\\nAnswer:"

            RL1 — nonsense syllogisms (answer G=Good/P=Poor logic)
              BEFORE: "Task: ...\\nQuestion: {syllogism}\\nAnswer:"
              AFTER:  "Does the following syllogism follow logically (regardless of whether the premises are true)?\\n{syllogism}\\nAnswer G if the logic is valid, P if it is not. Answer (G or P):"

            RL3/RL4, I1/I2 — inference / multiple-choice with numbered choices
              BEFORE: "Task: ...\\nQuestion: {q}\\nAnswer:"   (choices never shown)
              AFTER:  "Statement: {q}\\n1. ...\\n2. ...\\n\\nAnswer (number):"

            V1/V2/V3 — vocabulary multiple-choice (was mostly correct already)
              No change to prompt; scoring now extracts first line.
            """
            category = instance['category_id']
            question = instance['question']
            category_name = instance.get('category_name', category)
            choices_raw = instance.get('choice', '') or ''
            additional = instance.get('additional', '') or ''
            question_text = str(question).replace('<br>', '\n').strip()

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

            elif category.startswith('RG'):  # Arithmetic/Reasoning — multiple choice, answer is a letter A-E
                choices = [c.strip() for c in choices_raw.split(';;')] if choices_raw else []
                if choices:
                    letters = 'ABCDE'
                    choice_text = '\n'.join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
                    if category == 'RG3':
                        return (
                            f"Identify which arithmetic operation(s) are needed to solve this problem. "
                            f"Do not compute the final numeric result.\n\n"
                            f"Problem: {question_text}\n\n{choice_text}\n\nAnswer (letter):"
                        )
                    return f"Solve this problem: {question}\n\n{choice_text}\n\nAnswer (letter):"
                return f"Solve this problem: {question}\nAnswer:"

            elif category.startswith('MA'):  # Memory/lookup — table in `additional`, answer is a number
                if additional:
                    table = additional.replace('<br>', '\n').strip()
                    if category == 'MA3':
                        return (
                            "FIRST AND LAST NAMES TEST\n"
                            "Learn and use first/last name pairings from the list below. "
                            "Given a last name, return the matching first name only.\n\n"
                            f"{table}\n\nLast name: {question_text}\nFirst name:"
                        )
                    return f"{table}\n\nQuestion: What number corresponds to '{question_text}'?\nAnswer:"
                if category == 'MA3':
                    return f"Last name: {question_text}\nFirst name:"
                return f"Question: What number corresponds to '{question_text}'?\nAnswer:"

            elif category == 'RL1':  # Nonsense Syllogisms — answer is G (good/valid) or P (poor/invalid)
                return (f"Does the following syllogism follow logically, regardless of whether "
                        f"the premises are true?\n\n{question}\n\n"
                        f"Answer G if the logic is valid, P if it is not.\nAnswer (G or P):")

            elif category.startswith('RL') or category.startswith('I'):  # Inference — numbered choices, answer is 1-5
                choices = [c.strip() for c in choices_raw.split(';;')] if choices_raw else []
                if category == 'I1' and choices:
                    choice_text = '\n'.join(f"{i+1}. {c}" for i, c in enumerate(choices))
                    return (
                        "Four options follow one letter-pattern rule and one does not. "
                        "Pick the option that does NOT fit the same pattern.\n\n"
                        f"{choice_text}\n\nAnswer (number):"
                    )

                if category == 'I2':
                    return (
                        "Each row marks one location with an 'x'. Use the pattern across rows to determine "
                        "which numbered position (1-5) is correct.\n\n"
                        f"{question_text}\n\nAnswer (number):"
                    )

                if category == 'RL4' and choices:
                    reference = additional.replace('<br>', '\n').strip() if additional else ''
                    choice_text = '\n'.join(f"{i+1}. {c}" for i, c in enumerate(choices))
                    if reference:
                        return (
                            "DECIPHERING LANGUAGES\n"
                            "Reason across the language fragments below to infer how the ancient language maps "
                            "to the target language, then choose the best translation for the query.\n\n"
                            f"Known pairs:\n{reference}\n\n"
                            f"Query: {question_text}\n\n{choice_text}\n\nAnswer (number):"
                        )
                    return (
                        "Choose the best translation for the query from the numbered options.\n\n"
                        f"Query: {question_text}\n\n{choice_text}\n\nAnswer (number):"
                    )

                if choices:
                    choice_text = '\n'.join(f"{i+1}. {c}" for i, c in enumerate(choices))
                    return f"Statement: {question_text}\n\nWhich conclusion follows?\n{choice_text}\n\nAnswer (number):"
                return f"Task: {category_name}\nStatement: {question}\nAnswer:"

            elif category.startswith('V'):  # Vocabulary — numbered choices, answer is 1-5
                choices = [c.strip() for c in choices_raw.split(';;')] if choices_raw else []
                if choices:
                    choice_text = '\n'.join(f"{i+1}. {c}" for i, c in enumerate(choices))
                    level_note = {
                        'V1': 'V1 (easier)',
                        'V2': 'V2',
                        'V3': 'V3',
                        'V4': 'V4',
                        'V5': 'V5 (harder)',
                    }.get(category, category)
                    return (
                        f"VOCABULARY TEST - {level_note}\n"
                        "Test your knowledge of word meanings.\n"
                        f"Choose the best definition for '{question_text}'. Respond with only the option number.\n\n"
                        f"{choice_text}\n\nAnswer (number):"
                    )
                return f"What does '{question}' mean?"

            # Default format for other categories
            return f"Task: {category_name}\nQuestion: {question}\nAnswer:"

        def _with_answer(self, prompt_block: str, answer: str) -> str:
            """Attach a gold answer to a query-style prompt block for demonstrations."""
            block = prompt_block.rstrip()
            answer_text = str(answer).strip()

            if block.endswith(":"):
                return f"{block} {answer_text}"
            return f"{block}\nAnswer: {answer_text}"

        def build_prompt(self, instance: Dict[str, Any], num_shots: int = 5) -> str:
            """Build prompt with same-category few-shot examples before the query."""
            sections: List[str] = []

            if num_shots > 0:
                category = instance.get('category_id')
                question = instance.get('question')
                candidates = [
                    ex for ex in self.data
                    if ex.get('category_id') == category and ex.get('question') != question
                ]
                rng = random.Random(42)
                rng.shuffle(candidates)

                for demo in candidates[:num_shots]:
                    demo_block = self._format_category_prompt(demo)
                    sections.append(self._with_answer(demo_block, demo.get('answer', '')))

            sections.append(self._format_category_prompt(instance))
            return "\n\n".join(sections)
        
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
                choices_raw = example.get('choice', '') or ''
                
                # Initialize category stats
                if category not in category_stats:
                    category_stats[category] = {'correct': 0, 'total': 0}
                
                is_correct = self._is_correct(pred, expected, category, choices_raw=choices_raw)
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
        
        def _is_correct(self, prediction: str, expected: str, category: str,
                        choices_raw: str = '') -> bool:
            """Check if prediction is correct based on category.

            Always extracts only the first non-empty line of the prediction so
            that model continuations ("1\\nQuestion: ...\\n...") don't break matches.
            """
            # Extract first meaningful line
            first_line = ''
            for line in prediction.split('\n'):
                s = line.strip()
                if s:
                    first_line = s
                    break
            pred_clean = first_line.lower()
            expected_clean = str(expected).strip().lower()

            # Skip subjective tasks marked with <LLMEval>
            if '<llmeval>' in expected_clean:
                return False

            # RG (A-E letter answer): accept the letter directly, OR check if the
            # correct choice text appears in the prediction (for old free-gen preds)
            if category.startswith('RG'):
                letters = 'abcde'
                # Direct letter match (first char of prediction is the answer letter)
                if pred_clean and pred_clean[0] == expected_clean:
                    return True
                # Soft match: prediction contains the text of the correct choice
                if choices_raw:
                    choices = [c.strip().lower() for c in choices_raw.split(';;')]
                    try:
                        idx = letters.index(expected_clean)
                        correct_text = choices[idx]
                        # prediction contains correct choice value
                        if correct_text and correct_text in pred_clean:
                            return True
                    except (ValueError, IndexError):
                        pass
                return False

            # RL1 (G/P answer)
            if category == 'RL1':
                return pred_clean.startswith(expected_clean)

            # RL3/RL4, I1/I2 — numbered choice, answer is digit string
            if category.startswith('RL') or category.startswith('I'):
                # Accept first digit character matching
                import re
                m = re.match(r'(\d+)', pred_clean)
                return bool(m and m.group(1) == expected_clean)

            # Vocabulary: numbered choice, answer is digit string
            if category.startswith('V'):
                import re
                m = re.match(r'(\d+)', pred_clean)
                if m:
                    return m.group(1) == expected_clean
                # fallback: first char
                if pred_clean and pred_clean[0] == expected_clean:
                    return True
                return False

            # For tasks with multiple correct answers separated by ;;
            if ';;' in expected_clean:
                correct_answers = [ans.strip() for ans in expected_clean.split(';;')]
                return pred_clean in correct_answers

            # Default: exact match of first line
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

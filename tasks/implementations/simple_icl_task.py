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
        import string

        # Build full alphabet demonstrations for completeness
        uppercase_demos = [f"{ch.lower()} -> {ch}" for ch in string.ascii_uppercase]
        lowercase_demos = [f"{ch} -> {ch.lower()}" for ch in string.ascii_uppercase]

        self.category_demonstrations = {
            "uppercase": uppercase_demos,
            "lowercase": lowercase_demos,
            "first_letter": [
                "the cat went up the tree -> t",
                "elephants are cool -> e",
                "beautiful morning sunshine -> b",
                "running through the forest -> r",
                "amazing journey ahead -> a",
            ],
            "last_letter": [
                "the cat went up the tree -> e",
                "elephants are cool -> l",
                "beautiful morning sunshine -> e",
                "running through the forest -> t",
                "amazing journey ahead -> d",
            ],
            "translate_eng_fr": [
                "hello -> bonjour",
                "goodbye -> au revoir",
                "thank you -> merci",
                "please -> s'il vous plaît",
                "yes -> oui",
                "cat -> chat",
                "dog -> chien",
                "house -> maison",
                "car -> voiture",
                "book -> livre",
                "flower -> fleur",
                "pen -> stylo",
                "pencil -> crayon",
            ],
            "translate_fr_eng": [
                "bonjour -> hello",
                "au revoir -> goodbye",
                "merci -> thank you",
                "s'il vous plaît -> please",
                "oui -> yes",
                "chat -> cat",
                "chien -> dog",
                "maison -> house",
                "voiture -> car",
                "livre -> book",
                "fleur -> flower",
                "stylo -> pen",
                "crayon -> pencil",
            ],
            "translate_eng_sp": [
                "hello -> hola",
                "goodbye -> adiós",
                "thank you -> gracias",
                "please -> por favor",
                "yes -> sí",
                "cat -> gato",
                "dog -> perro",
                "house -> casa",
                "car -> coche",
                "book -> libro",
                "flower -> flor",
                "pen -> bolígrafo",
                "pencil -> lápiz",
                "flower -> flor",
            ],
            "translate_sp_eng": [
                "hola -> hello",
                "adiós -> goodbye",
                "gracias -> thank you",
                "por favor -> please",
                "sí -> yes",
                "gato -> cat",
                "perro -> dog",
                "casa -> house",
                "coche -> car",
                "libro -> book",
                "flor -> flower",
                "bolígrafo -> pen",
                "lápiz -> pencil",
                "flor -> flower"
            ],
            "present_to_gerund": [
                "run -> running", "swim -> swimming", "walk -> walking",
                "jump -> jumping", "dance -> dancing", "sing -> singing",
                "read -> reading", "write -> writing", "sleep -> sleeping",
            ],
            "singular_to_plural": [
                "cat -> cats", "dog -> dogs", "book -> books", "car -> cars",
                "house -> houses", "child -> children", "mouse -> mice",
            ],
            "country_to_capital": [
                "France -> Paris", "Germany -> Berlin", "Italy -> Rome",
                "Spain -> Madrid", "United Kingdom -> London", "Japan -> Tokyo",
                "Canada -> Ottawa", "Australia -> Canberra",
            ],
            "country_to_currency": [
                "France -> Euro", "United States -> Dollar", "United Kingdom -> Pound",
                "Japan -> Yen", "Canada -> Canadian Dollar", "Australia -> Australian Dollar",
                "India -> Rupee", "China -> Yuan",
            ],
        }
    
    def _load_data(self):
        """Load data from CSV file."""
        import pandas as pd
        
        if self.config.data_path:
            self.data = pd.read_csv(self.config.data_path)
        else:
            # If no data path, create empty DataFrame
            self.data = pd.DataFrame(columns=[self.config.input_column, self.config.output_column])
    
    def build_prompt(self, instance: Dict[str, Any], num_shots: int = 5) -> str:
        """Build a prompt with category-specific demonstrations.
        
        Args:
            instance: The instance to build a prompt for
            num_shots: Number of in-context learning examples (default: 5)
        
        Returns:
            Formatted prompt string
        """
        category = instance.get('category_name', '')
        
        # Check if we have demonstrations in the config (from in-memory data)
        if (self.config.in_memory_demonstrations and 
            isinstance(self.config.in_memory_demonstrations, dict)):
            demos = self.config.in_memory_demonstrations.get(category, [])
        else:
            # Fallback to hardcoded demonstrations
            demos = self.category_demonstrations.get(category, [])
        
        # Exclude demonstrations that match the current instance input to avoid leaking the test item
        instance_input = instance.get(self.config.input_column, '')
        filtered_demos = []
        for d in demos:
            # Each demo may be formatted like 'input -> output'
            demo_input = d.split('->', 1)[0].strip()
            if demo_input != instance_input:
                filtered_demos.append(d)

        # Limit to num_shots if provided
        if filtered_demos and num_shots > 0:
            demos = filtered_demos[:num_shots]
        else:
            demos = filtered_demos
        
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

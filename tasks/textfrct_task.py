"""TextFRCT task implementation that integrates with the existing dataset utilities."""

import sys
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_task import BaseTask, TaskConfig
from dataset.utils import TextFRCT as TextFRCTDataset
from dataset.demonstrations import VANILLA_DEMONSTRATIONS
from dataset.questions import VANILLA_QUESTIONS


class TextFRCTTask(BaseTask):
    """Task wrapper for the TextFRCT dataset."""
    
    def __init__(self, config: TaskConfig, skip_subjective: bool = False):
        # Initialize the TextFRCT dataset
        self.textfrct_dataset = TextFRCTDataset(
            data_path=config.data_path,
            skip_subjective=skip_subjective
        )
        
        # Call parent init with modified config
        super().__init__(config)
        
        # Override data with TextFRCT prompts
        self.prompts = list(self.textfrct_dataset.build_prompt(VANILLA_QUESTIONS, VANILLA_DEMONSTRATIONS))
        
        # Store original data for evaluation
        self.original_data = self.textfrct_dataset.data
    
    def _load_data(self):
        """Override data loading since we use TextFRCT dataset directly."""
        # Data is loaded in __init__ via TextFRCTDataset
        pass
    
    def get_split(self, split: str = "test") -> List[Dict[str, Any]]:
        """Get data split for TextFRCT."""
        if split == "test" or split == "all":
            # Convert prompts to the expected format
            result = []
            for i, prompt in enumerate(self.prompts):
                result.append({
                    "input": prompt,
                    "index": i
                })
            return result
        else:
            return []
    
    def build_prompt(self, instance: Dict[str, Any]) -> str:
        """Build prompt for TextFRCT - prompts are already built."""
        return instance["input"]
    
    def evaluate(self, predictions: List[str], split: str = "test", **kwargs) -> Dict[str, float]:
        """Evaluate predictions using TextFRCT's evaluation method."""
        if len(predictions) != len(self.prompts):
            raise ValueError(f"Prediction count ({len(predictions)}) doesn't match prompt count ({len(self.prompts)})")
        
        # Use TextFRCT's evaluation method
        # Create a temporary file for the evaluation
        temp_file = Path(self.config.metadata.get("temp_file", "temp_textfrct_eval.csv"))
        
        try:
            results = self.textfrct_dataset.evaluate(predictions, temp_file)
            return results
        except Exception as e:
            print(f"Error in TextFRCT evaluation: {e}")
            # Fallback to simple accuracy if TextFRCT evaluation fails
            return {"accuracy": 0.0, "error": str(e)}
    
    def get_ground_truth(self, split: str = "test") -> List[str]:
        """Get ground truth for TextFRCT - not directly available."""
        # TextFRCT doesn't expose ground truth directly
        # Return empty list as ground truth is handled in evaluate()
        return [""] * len(self.prompts)


def create_textfrct_task(
    data_path: str = "dataset/TextFRCT.csv",
    skip_subjective: bool = False,
    name: str = "textfrct"
) -> TextFRCTTask:
    """Create a TextFRCT task instance."""
    config = TaskConfig(
        name=name,
        description="TextFRCT evaluation dataset",
        data_path=data_path,
        data_format="csv",
        input_column="input",
        output_column="output",
        evaluation_metrics=["accuracy"],
        metadata={"skip_subjective": skip_subjective}
    )
    
    return TextFRCTTask(config, skip_subjective=skip_subjective)

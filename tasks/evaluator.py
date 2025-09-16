"""Task evaluator for running models on tasks with different backends."""

import json
import torch
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Model backends
try:
    import vllm
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False

from .base_task import BaseTask


@dataclass
class ModelConfig:
    """Configuration for model loading and generation."""
    model_id: str
    backend: str  # 'vllm', 'transformers', 'openai', 'together'
    checkpoint: Optional[str] = None
    local_path: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 100
    top_p: float = 1.0
    tensor_parallel_size: Optional[int] = None
    trust_remote_code: bool = True
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""
    output_dir: str = "results"
    save_predictions: bool = True
    save_detailed_results: bool = True
    batch_size: int = 1
    retry_attempts: int = 3
    retry_delay: float = 1.0


class TaskEvaluator:
    """Main evaluator class for running models on tasks."""
    
    def __init__(self, model_config: ModelConfig, eval_config: EvaluationConfig):
        self.model_config = model_config
        self.eval_config = eval_config
        self.model = None
        self.tokenizer = None
        self.client = None
        
        # Create output directory
        Path(self.eval_config.output_dir).mkdir(parents=True, exist_ok=True)
        
        self._load_model()
    
    def _load_model(self):
        """Load the model based on the backend configuration."""
        backend = self.model_config.backend.lower()
        
        if backend == 'vllm':
            self._load_vllm_model()
        elif backend == 'transformers':
            self._load_transformers_model()
        elif backend == 'openai':
            self._load_openai_client()
        elif backend == 'together':
            self._load_together_client()
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def _load_vllm_model(self):
        """Load model using vLLM backend."""
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not available. Please install it with: pip install vllm")
        
        model_path = self.model_config.local_path or self.model_config.model_id
        tensor_parallel_size = self.model_config.tensor_parallel_size or torch.cuda.device_count()
        
        self.model = vllm.LLM(
            model=model_path,
            tokenizer=model_path,
            revision=self.model_config.checkpoint,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=self.model_config.trust_remote_code,
        )
    
    def _load_transformers_model(self):
        """Load model using Transformers backend."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers is not available. Please install it with: pip install transformers")
        
        model_path = self.model_config.local_path or self.model_config.model_id
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            revision=self.model_config.checkpoint,
            trust_remote_code=self.model_config.trust_remote_code
        )
        
        # Handle missing pad token (common in GPT-2 based models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"⚠️  No pad token found, using EOS token as pad token: '{self.tokenizer.pad_token}'")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=self.model_config.checkpoint,
            trust_remote_code=self.model_config.trust_remote_code,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def _load_openai_client(self):
        """Load OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI is not available. Please install it with: pip install openai")
        
        api_key = self.model_config.api_key
        if not api_key:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=api_key)
    
    def _load_together_client(self):
        """Load Together client."""
        if not TOGETHER_AVAILABLE:
            raise ImportError("Together is not available. Please install it with: pip install together")
        
        api_key = self.model_config.api_key
        if not api_key:
            import os
            api_key = os.getenv("TOGETHER_API_KEY")
        
        if not api_key:
            raise ValueError("Together API key not provided")
        
        import os
        os.environ["TOGETHER_API_KEY"] = api_key
        self.client = Together()
    
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for a list of prompts."""
        backend = self.model_config.backend.lower()
        
        if backend == 'vllm':
            return self._generate_vllm(prompts)
        elif backend == 'transformers':
            return self._generate_transformers(prompts)
        elif backend in ['openai', 'together']:
            return self._generate_api(prompts)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def _generate_vllm(self, prompts: List[str]) -> List[str]:
        """Generate using vLLM."""
        sampling_params = vllm.SamplingParams(
            temperature=self.model_config.temperature,
            max_tokens=self.model_config.max_tokens,
            top_p=self.model_config.top_p,
            **self.model_config.generation_kwargs
        )
        
        print(f"Generating {len(prompts)} predictions with vLLM...")
        outputs = self.model.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def _generate_transformers(self, prompts: List[str]) -> List[str]:
        """Generate using Transformers."""
        generated_texts = []
        
        # Use tqdm for progress tracking
        for prompt in tqdm(prompts, desc="Generating predictions", unit="prompt"):
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Prepare generation arguments
            generation_kwargs = {
                "max_new_tokens": self.model_config.max_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                **self.model_config.generation_kwargs
            }
            
            # Handle temperature and sampling
            if self.model_config.temperature > 0:
                generation_kwargs.update({
                    "do_sample": True,
                    "temperature": self.model_config.temperature,
                    "top_p": self.model_config.top_p,
                })
            else:
                generation_kwargs["do_sample"] = False
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # Decode only the generated part (exclude input)
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def _generate_api(self, prompts: List[str]) -> List[str]:
        """Generate using API backends (OpenAI/Together)."""
        generated_texts = []
        
        for prompt in tqdm(prompts, desc="Generating predictions", unit="prompt"):
            for attempt in range(self.eval_config.retry_attempts):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_config.model_id,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.model_config.temperature,
                        max_tokens=self.model_config.max_tokens,
                        top_p=self.model_config.top_p,
                        **self.model_config.generation_kwargs
                    )
                    generated_texts.append(response.choices[0].message.content)
                    break
                except Exception as e:
                    if attempt == self.eval_config.retry_attempts - 1:
                        print(f"Failed to generate for prompt after {self.eval_config.retry_attempts} attempts: {e}")
                        generated_texts.append("")
                    else:
                        print(f"Attempt {attempt + 1} failed, retrying: {e}")
                        time.sleep(self.eval_config.retry_delay)
        
        return generated_texts
    
    def evaluate_task(self, task: BaseTask, split: str = "test") -> Dict[str, Any]:
        """Evaluate a model on a specific task."""
        print(f"Evaluating task: {task.config.name}")
        
        # Get task data
        task_data = task.get_split(split)
        print(f"Loaded {len(task_data)} examples")
        
        # Build prompts
        prompts = [task.build_prompt(instance) for instance in task_data]
        
        # Generate predictions
        print("Generating predictions...")
        predictions = self.generate(prompts)
        
        # Evaluate
        print("Evaluating predictions...")
        metrics = task.evaluate(predictions, split=split)
        
        # Prepare results
        results = {
            "task_name": task.config.name,
            "model_id": self.model_config.model_id,
            "backend": self.model_config.backend,
            "checkpoint": self.model_config.checkpoint,
            "split": split,
            "num_examples": len(task_data),
            "metrics": metrics,
            "config": {
                "model_config": self.model_config.__dict__,
                "eval_config": self.eval_config.__dict__,
                "task_config": task.config.__dict__
            }
        }
        
        # Save results
        if self.eval_config.save_predictions or self.eval_config.save_detailed_results:
            self._save_results(results, task_data, prompts, predictions)
        
        return results
    
    def _save_results(self, results: Dict[str, Any], task_data: List[Dict], 
                     prompts: List[str], predictions: List[str]):
        """Save evaluation results to files."""
        model_name = self.model_config.model_id.replace('/', '_')
        task_name = results["task_name"]
        checkpoint = self.model_config.checkpoint or "main"
        
        base_filename = f"{model_name}_{checkpoint}_{task_name}"
        
        # Save summary metrics
        summary_path = Path(self.eval_config.output_dir) / f"{base_filename}_metrics.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save detailed predictions if requested
        if self.eval_config.save_detailed_results:
            detailed_data = []
            for i, (data, prompt, prediction) in enumerate(zip(task_data, prompts, predictions)):
                detailed_data.append({
                    "index": i,
                    "input": data.get("input", ""),
                    "ground_truth": data.get("output", ""),
                    "prompt": prompt,
                    "prediction": prediction,
                    "metadata": {k: v for k, v in data.items() if k not in ["input", "output"]}
                })
            
            detailed_path = Path(self.eval_config.output_dir) / f"{base_filename}_detailed.jsonl"
            with open(detailed_path, 'w', encoding='utf-8') as f:
                for item in detailed_data:
                    f.write(json.dumps(item, default=str) + '\n')
        
        print(f"Results saved to {self.eval_config.output_dir}")


# Convenience function for quick evaluation
def evaluate_model_on_task(
    model_config: ModelConfig,
    task: BaseTask,
    eval_config: Optional[EvaluationConfig] = None,
    split: str = "test"
) -> Dict[str, Any]:
    """Convenience function to evaluate a model on a task."""
    if eval_config is None:
        eval_config = EvaluationConfig()
    
    evaluator = TaskEvaluator(model_config, eval_config)
    return evaluator.evaluate_task(task, split)

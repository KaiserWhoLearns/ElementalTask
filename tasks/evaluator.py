"""Task evaluator for running models on tasks with different backends."""

import json
import torch
import torch.nn.functional as F
import time
import math
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
    
    # Evaluation mode: "exact_match", "continuous", "all"
    eval_mode: str = "exact_match"
    
    # Continuous metrics settings
    compute_loss: bool = True
    compute_perplexity: bool = True


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
    
    # ============================================================
    # CONTINUOUS METRICS: Loss/Perplexity/Probability Computation
    # ============================================================
    
    def compute_target_metrics(self, prompts: List[str], targets: List[str]) -> List[Dict[str, float]]:
        """
        Compute continuous metrics (loss, perplexity, probability) for each (prompt, target) pair.
        
        This provides a much smoother and more informative metric than binary exact match,
        capturing "partial" learning and model confidence.
        
        Args:
            prompts: List of input prompts
            targets: List of expected target outputs
            
        Returns:
            List of dicts with loss, perplexity, probability for each example
        """
        backend = self.model_config.backend.lower()
        
        if backend == "transformers":
            return self._compute_target_metrics_transformers(prompts, targets)
        elif backend == "vllm":
            return self._compute_target_metrics_vllm(prompts, targets)
        elif backend in ["openai", "together"]:
            print("⚠️  Continuous metrics have limited support for API backends")
            return self._compute_target_metrics_api(prompts, targets)
        else:
            raise ValueError(f"Continuous metrics not supported for backend: {backend}")
    
    def _compute_target_metrics_transformers(self, prompts: List[str], targets: List[str]) -> List[Dict[str, float]]:
        """Compute loss/perplexity for each (prompt, target) pair using Transformers."""
        device = next(self.model.parameters()).device
        results = []
        
        for prompt, target in tqdm(zip(prompts, targets), total=len(prompts), desc="Computing target metrics"):
            # Tokenize prompt and full sequence separately
            prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True)["input_ids"]
            full_text = prompt + target
            full_ids = self.tokenizer(full_text, return_tensors="pt", truncation=True)["input_ids"]
            
            prompt_len = prompt_ids.shape[1]
            full_len = full_ids.shape[1]
            
            # If target adds no new tokens, skip
            if full_len <= prompt_len:
                results.append({
                    "loss": float('inf'),
                    "perplexity": float('inf'),
                    "probability": 0.0,
                    "normalized_probability": 0.0,
                    "num_target_tokens": 0
                })
                continue
            
            full_ids = full_ids.to(device)
            
            with torch.no_grad():
                outputs = self.model(full_ids)
                logits = outputs.logits  # [1, seq_len, vocab_size]
                
                # Compute loss only on target tokens
                # Shift: predict token[i+1] from position[i]
                target_logits = logits[0, prompt_len-1:full_len-1, :]  # [target_len, vocab_size]
                target_labels = full_ids[0, prompt_len:full_len]  # [target_len]
                
                num_target_tokens = target_labels.shape[0]
                
                # Compute per-token log probabilities
                log_probs = F.log_softmax(target_logits, dim=-1)
                token_log_probs = log_probs[range(num_target_tokens), target_labels]
                
                # Average negative log likelihood (loss)
                avg_loss = -token_log_probs.mean().item()
                
                # Perplexity = exp(loss)
                perplexity = math.exp(avg_loss)
                
                # Probability = exp(sum of log probs)
                total_log_prob = token_log_probs.sum().item()
                probability = math.exp(total_log_prob)
                
                # Normalized probability (geometric mean)
                normalized_prob = math.exp(total_log_prob / num_target_tokens)
            
            results.append({
                "loss": avg_loss,
                "perplexity": perplexity,
                "probability": probability,
                "normalized_probability": normalized_prob,
                "total_log_prob": total_log_prob,
                "num_target_tokens": num_target_tokens
            })
        
        return results
    
    def _compute_target_metrics_vllm(self, prompts: List[str], targets: List[str]) -> List[Dict[str, float]]:
        """Compute loss/perplexity for each (prompt, target) pair using vLLM."""
        results = []
        
        # vLLM can return logprobs - we'll use prompt_logprobs
        sampling_params = vllm.SamplingParams(
            max_tokens=1,
            prompt_logprobs=0,  # Return logprobs for prompt tokens
            temperature=0.0
        )
        
        print("Computing target metrics with vLLM...")
        
        for prompt, target in tqdm(zip(prompts, targets), total=len(prompts), desc="Computing target metrics"):
            full_text = prompt + target
            
            try:
                outputs = self.model.generate([full_text], sampling_params)
                output = outputs[0]
                
                if output.prompt_logprobs is not None:
                    # Find where target starts
                    prompt_tokens = self.model.get_tokenizer().encode(prompt)
                    prompt_len = len(prompt_tokens)
                    
                    # Get logprobs for target tokens
                    target_logprobs = []
                    for i, lp in enumerate(output.prompt_logprobs):
                        if i >= prompt_len and lp is not None:
                            token_id = output.prompt_token_ids[i]
                            if token_id in lp:
                                target_logprobs.append(lp[token_id].logprob)
                    
                    if target_logprobs:
                        total_log_prob = sum(target_logprobs)
                        avg_loss = -sum(target_logprobs) / len(target_logprobs)
                        perplexity = math.exp(avg_loss)
                        probability = math.exp(total_log_prob)
                        normalized_prob = math.exp(total_log_prob / len(target_logprobs))
                        
                        results.append({
                            "loss": avg_loss,
                            "perplexity": perplexity,
                            "probability": probability,
                            "normalized_probability": normalized_prob,
                            "total_log_prob": total_log_prob,
                            "num_target_tokens": len(target_logprobs)
                        })
                        continue
                
                # Fallback if logprobs not available
                results.append({
                    "loss": float('inf'),
                    "perplexity": float('inf'),
                    "probability": 0.0,
                    "normalized_probability": 0.0,
                    "num_target_tokens": 0
                })
                
            except Exception as e:
                print(f"Warning: Failed to compute metrics: {e}")
                results.append({
                    "loss": float('inf'),
                    "perplexity": float('inf'),
                    "probability": 0.0,
                    "normalized_probability": 0.0,
                    "num_target_tokens": 0,
                    "error": str(e)
                })
        
        return results
    
    def _compute_target_metrics_api(self, prompts: List[str], targets: List[str]) -> List[Dict[str, float]]:
        """Compute metrics using API (limited support)."""
        # Most chat models don't support logprobs directly
        # Return placeholder results
        return [
            {
                "loss": None,
                "perplexity": None,
                "probability": None,
                "note": "Continuous metrics not fully supported for API backends"
            }
            for _ in prompts
        ]
    
    def evaluate_task(self, task: BaseTask, split: str = "test") -> Dict[str, Any]:
        """
        Evaluate a model on a specific task.
        
        Supports multiple evaluation modes:
        - "exact_match": Traditional binary accuracy (default)
        - "continuous": Loss/perplexity-based evaluation (smoother metrics)
        - "all": Both exact_match and continuous metrics
        """
        eval_mode = self.eval_config.eval_mode
        print(f"Evaluating task: {task.config.name}")
        print(f"Evaluation mode: {eval_mode}")
        
        # Get task data
        task_data = task.get_split(split)
        print(f"Loaded {len(task_data)} examples")
        
        # Build prompts
        prompts = [task.build_prompt(instance) for instance in task_data]
        
        # Get targets for continuous metrics
        output_col = task.config.output_column or "answer"
        targets = [str(instance.get(output_col, instance.get("output", ""))) for instance in task_data]
        
        # Initialize results
        results = {
            "task_name": task.config.name,
            "model_id": self.model_config.model_id,
            "backend": self.model_config.backend,
            "checkpoint": self.model_config.checkpoint,
            "split": split,
            "eval_mode": eval_mode,
            "num_examples": len(task_data),
            "metrics": {},
            "config": {
                "model_config": self.model_config.__dict__,
                "eval_config": self.eval_config.__dict__,
                "task_config": task.config.__dict__
            }
        }
        
        predictions = None
        target_metrics = None
        
        # === EXACT MATCH EVALUATION ===
        if eval_mode in ["exact_match", "all"]:
            print("\nGenerating predictions for exact match...")
            predictions = self.generate(prompts)
            
            print("Evaluating predictions...")
            exact_match_metrics = task.evaluate(predictions, split=split)
            results["metrics"]["exact_match"] = exact_match_metrics
            results["predictions"] = predictions
        
        # === CONTINUOUS METRICS EVALUATION ===
        if eval_mode in ["continuous", "all"]:
            print("\nComputing continuous metrics (loss/perplexity)...")
            target_metrics = self.compute_target_metrics(prompts, targets)
            
            # Aggregate metrics
            valid_metrics = [m for m in target_metrics if m.get("loss") is not None and m.get("loss") != float('inf')]
            
            if valid_metrics:
                results["metrics"]["continuous"] = {
                    "mean_loss": sum(m["loss"] for m in valid_metrics) / len(valid_metrics),
                    "mean_perplexity": sum(m["perplexity"] for m in valid_metrics) / len(valid_metrics),
                    "mean_probability": sum(m.get("probability", 0) for m in valid_metrics) / len(valid_metrics),
                    "mean_normalized_probability": sum(m.get("normalized_probability", 0) for m in valid_metrics) / len(valid_metrics),
                    "num_valid_examples": len(valid_metrics),
                    "num_total_examples": len(target_metrics)
                }
            else:
                results["metrics"]["continuous"] = {
                    "error": "No valid metrics computed",
                    "num_valid_examples": 0,
                    "num_total_examples": len(target_metrics)
                }
            
            results["target_metrics"] = target_metrics
        
        # Save results
        if self.eval_config.save_predictions or self.eval_config.save_detailed_results:
            self._save_results(results, task_data, prompts, predictions, target_metrics, targets)
        
        return results
    
    def _save_results(self, results: Dict[str, Any], task_data: List[Dict], 
                     prompts: List[str], predictions: Optional[List[str]] = None,
                     target_metrics: Optional[List[Dict]] = None, 
                     targets: Optional[List[str]] = None):
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
            for i, data in enumerate(task_data):
                item = {
                    "index": i,
                    "input": data.get("input", ""),
                    "ground_truth": data.get("output", ""),
                    "prompt": prompts[i],
                    "metadata": {k: v for k, v in data.items() if k not in ["input", "output"]}
                }
                
                # Add prediction if available
                if predictions is not None:
                    item["prediction"] = predictions[i]
                
                # Add target and continuous metrics if available
                if targets is not None:
                    item["target"] = targets[i]
                
                if target_metrics is not None:
                    item["continuous_metrics"] = target_metrics[i]
                
                detailed_data.append(item)
            
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

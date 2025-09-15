#!/usr/bin/env python3
"""Command-line interface for the unified task evaluation system."""

import argparse
import json
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tasks.base_task import create_task_from_config, SimpleTask
from tasks.simple_icl_task import SimpleICLTask
from tasks.textfrct_task import create_textfrct_task
from tasks.evaluator import TaskEvaluator, ModelConfig, EvaluationConfig


def create_model_config_from_args(args) -> ModelConfig:
    """Create ModelConfig from command line arguments."""
    return ModelConfig(
        model_id=args.model_id,
        backend=args.backend,
        checkpoint=args.checkpoint,
        local_path=args.local_path,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code
    )


def create_eval_config_from_args(args) -> EvaluationConfig:
    """Create EvaluationConfig from command line arguments."""
    return EvaluationConfig(
        output_dir=args.output_dir,
        save_predictions=args.save_predictions,
        save_detailed_results=args.save_detailed_results,
        batch_size=args.batch_size,
        retry_attempts=args.retry_attempts,
        retry_delay=args.retry_delay
    )


def main():
    parser = argparse.ArgumentParser(description="Unified Task Evaluation System")
    
    # Task arguments
    task_group = parser.add_argument_group("Task Configuration")
    task_group.add_argument("--task_type", type=str, required=True,
                           choices=["simple_icl", "textfrct", "config"],
                           help="Type of task to run")
    task_group.add_argument("--task_config", type=str,
                           help="Path to task configuration file (for config task type)")
    task_group.add_argument("--skip_subjective", action="store_true",
                           help="Skip subjective categories (for TextFRCT)")
    
    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_id", type=str, required=True,
                            help="Model identifier")
    model_group.add_argument("--backend", type=str, required=True,
                            choices=["vllm", "transformers", "openai", "together"],
                            help="Model backend to use")
    model_group.add_argument("--checkpoint", type=str,
                            help="Model checkpoint/revision")
    model_group.add_argument("--local_path", type=str,
                            help="Local path to model (overrides model_id)")
    model_group.add_argument("--api_key", type=str,
                            help="API key for OpenAI/Together")
    
    # Generation arguments
    gen_group = parser.add_argument_group("Generation Configuration")
    gen_group.add_argument("--temperature", type=float, default=0.0,
                          help="Generation temperature")
    gen_group.add_argument("--max_tokens", type=int, default=100,
                          help="Maximum tokens to generate")
    gen_group.add_argument("--top_p", type=float, default=1.0,
                          help="Top-p sampling parameter")
    gen_group.add_argument("--tensor_parallel_size", type=int,
                          help="Tensor parallel size for vLLM")
    gen_group.add_argument("--trust_remote_code", action="store_true", default=True,
                          help="Trust remote code for model loading")
    
    # Evaluation arguments
    eval_group = parser.add_argument_group("Evaluation Configuration")
    eval_group.add_argument("--output_dir", type=str, default="results",
                           help="Output directory for results")
    eval_group.add_argument("--save_predictions", action="store_true", default=True,
                           help="Save prediction results")
    eval_group.add_argument("--save_detailed_results", action="store_true", default=True,
                           help="Save detailed results with prompts and predictions")
    eval_group.add_argument("--batch_size", type=int, default=1,
                           help="Batch size for evaluation")
    eval_group.add_argument("--retry_attempts", type=int, default=3,
                           help="Number of retry attempts for API calls")
    eval_group.add_argument("--retry_delay", type=float, default=1.0,
                           help="Delay between retry attempts")
    
    args = parser.parse_args()
    
    # Create configurations
    model_config = create_model_config_from_args(args)
    eval_config = create_eval_config_from_args(args)
    
    # Create task based on type
    if args.task_type == "simple_icl":
        print("Creating Simple ICL task...")
        task_config_path = "tasks/configs/simple_icl_tasks.json"
        if not Path(task_config_path).exists():
            print(f"Error: Task config file not found: {task_config_path}")
            return 1
        task = create_task_from_config(task_config_path, SimpleICLTask)
        
    elif args.task_type == "textfrct":
        print("Creating TextFRCT task...")
        task = create_textfrct_task(skip_subjective=args.skip_subjective)
        
    elif args.task_type == "config":
        if not args.task_config:
            print("Error: --task_config required for config task type")
            return 1
        print(f"Creating task from config: {args.task_config}")
        task = create_task_from_config(args.task_config, SimpleTask)
        
    else:
        print(f"Error: Unknown task type: {args.task_type}")
        return 1
    
    # Create evaluator and run evaluation
    print("Creating evaluator...")
    evaluator = TaskEvaluator(model_config, eval_config)
    
    print("Running evaluation...")
    results = evaluator.evaluate_task(task)
    
    # Print summary results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Task: {results['task_name']}")
    print(f"Model: {results['model_id']}")
    print(f"Backend: {results['backend']}")
    print(f"Examples: {results['num_examples']}")
    print("\nMetrics:")
    for metric, value in results['metrics'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    print(f"\nDetailed results saved to: {eval_config.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())

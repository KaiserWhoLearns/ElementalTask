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
from tasks.implementations.simple_icl_task import SimpleICLTask
from tasks.implementations.textfrct_task import create_textfrct_task
from tasks.evaluator import TaskEvaluator, ModelConfig, EvaluationConfig
from tasks.registry import TaskRegistry


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
    # Check for list categories flag first (before full argument parsing)
    if "--list_textfrct_categories" in sys.argv:
        print_textfrct_categories()
        return 0
    
    parser = argparse.ArgumentParser(description="Unified Task Evaluation System")
    
    # Task arguments
    task_group = parser.add_argument_group("Task Configuration")
    
    # Discover available tasks dynamically
    registry = TaskRegistry()
    available_tasks = list(registry.discover_tasks().keys())
    
    task_group.add_argument("--task_type", type=str, required=True,
                           choices=available_tasks + ["config"],
                           help=f"Type of task to run. Available: {', '.join(available_tasks)}")
    task_group.add_argument("--task_config", type=str,
                           help="Path to task configuration file (for config task type)")
    task_group.add_argument("--skip_subjective", action="store_true",
                           help="Skip subjective categories (for TextFRCT)")
    
    # TextFRCT specific arguments
    textfrct_group = parser.add_argument_group("TextFRCT Configuration")
    textfrct_group.add_argument("--textfrct_categories", type=str, nargs="*",
                               help="Specific TextFRCT categories to evaluate (e.g., CV1 CV2 FA1). "
                                    "If not specified, all categories are used. "
                                    "Common categories: CV1 (scrambled words), CV2 (hidden words), "
                                    "CV3 (incomplete words), FA1 (associations), FA2 (opposites), "
                                    "V1-V5 (vocabulary), RG1-RG3 (reasoning)")
    textfrct_group.add_argument("--list_textfrct_categories", action="store_true",
                               help="List all available TextFRCT categories and exit")
    
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
    
    # Create task dynamically using registry
    if args.task_type == "config":
        if not args.task_config:
            print("Error: --task_config required for config task type")
            return 1
        print(f"Creating task from config: {args.task_config}")
        
        # For config type, we need to determine the task type from the config file
        # For now, default to SimpleTask but this could be improved
        task = create_task_from_config(args.task_config, SimpleTask)
        
    elif args.task_type == "textfrct":
        # TextFRCT has special handling due to its factory function
        print("Creating TextFRCT task...")
        categories = args.textfrct_categories
        if categories:
            print(f"Filtering to categories: {categories}")
            task = create_textfrct_task(
                skip_subjective=args.skip_subjective,
                categories=categories
            )
        else:
            print("Using all categories (default)")
            task = create_textfrct_task(skip_subjective=args.skip_subjective)
            
    else:
        # Use registry for dynamic task creation
        print(f"Creating {args.task_type} task...")
        
        # Get the task class from registry
        task_class = registry.get_task_class(args.task_type)
        if not task_class:
            print(f"Error: Task '{args.task_type}' not found in registry")
            return 1
        
        # Create basic config for the task
        # This could be made more sophisticated with task-specific config files
        from tasks.base_task import TaskConfig
        config = TaskConfig(
            name=f"{args.task_type}_cli",
            description=f"{args.task_type} evaluation",
            data_path=f"dataset/{args.task_type}.csv",  # Default data path
            data_format="csv",
            input_column="question",
            output_column="answer",
            evaluation_metrics=["accuracy"]
        )
        
        # Create task instance
        task = task_class(config)
    
    # Create evaluator and run evaluation
    print("Creating evaluator...")
    evaluator = TaskEvaluator(model_config, eval_config)
    breakpoint()
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


def print_textfrct_categories():
    """Print all available TextFRCT categories with descriptions."""
    print("Available TextFRCT Categories:")
    print("=" * 50)
    
    categories = {
        "CV1": "Scrambled Words - Unscramble letters to form words",
        "CV2": "Hidden Words - Find words hidden in letter strings", 
        "CV3": "Incomplete Words - Complete words with missing letters",
        "FA1": "Controlled Association - Generate related words",
        "FA2": "Opposites - Generate antonyms",
        "FA3": "Figures of Speech - Complete metaphors/similes",
        "FE1": "Making Sentences - Create sentences from patterns",
        "FE2": "Arranging Words - Form sentences from word lists",
        "FE3": "Rewriting - Rewrite sentences with same meaning",
        "FI1": "Topics Test - Generate ideas about topics",
        "FI2": "Theme Test - Write paragraphs about themes", 
        "FI3": "Thing Categories - List items in categories",
        "FW1": "Word Fluency - Generate words with constraints",
        "FW2": "First and Last Letters - Words starting/ending with letters",
        "FW3": "Pattern Words - Words matching letter patterns",
        "I1": "Letter Sets - Pattern recognition with letters",
        "I2": "Figure Classification - Spatial reasoning",
        "MA2": "Object-Number - Associate objects with numbers",
        "MA3": "First Names - Recall associated names",
        "RG1": "Arithmetic Aptitude - Basic math operations",
        "RG2": "Number Series - Continue number sequences",
        "RG3": "Mathematical Reasoning - Solve math word problems",
        "RL1": "Syllogistic Reasoning - Logical syllogisms",
        "RL3": "Necessary Inference - Required logical conclusions",
        "RL4": "Language Deciphering - Decode artificial languages",
        "V1": "Vocabulary Level 1 - Basic vocabulary",
        "V2": "Vocabulary Level 2 - Intermediate vocabulary",
        "V3": "Vocabulary Level 3 - Advanced vocabulary", 
        "V4": "Vocabulary Level 4 - Expert vocabulary",
        "V5": "Vocabulary Level 5 - Scholar vocabulary",
        "XU1": "Object Combination - Combine objects creatively",
        "XU2": "Substitute Uses - Alternative uses for objects",
        "XU3": "Object Grouping - Group objects by function",
        "XU4": "Different Uses - Creative uses for objects"
    }
    
    # Group by main category
    groups = {
        "CV (Convergent Visual)": ["CV1", "CV2", "CV3"],
        "FA (Fluent Associational)": ["FA1", "FA2", "FA3"],
        "FE (Flexible Expression)": ["FE1", "FE2", "FE3"],
        "FI (Fluent Ideational)": ["FI1", "FI2", "FI3"],
        "FW (Fluent Word)": ["FW1", "FW2", "FW3"],
        "I (Inductive)": ["I1", "I2"],
        "MA (Memory/Association)": ["MA2", "MA3"],
        "RG (Reasoning)": ["RG1", "RG2", "RG3"],
        "RL (Reasoning/Logic)": ["RL1", "RL3", "RL4"],
        "V (Vocabulary)": ["V1", "V2", "V3", "V4", "V5"],
        "XU (Creative)": ["XU1", "XU2", "XU3", "XU4"]
    }
    
    for group_name, group_categories in groups.items():
        print(f"\n{group_name}:")
        for cat in group_categories:
            if cat in categories:
                print(f"  {cat}: {categories[cat]}")
    
    print(f"\nExample usage:")
    print(f"  # Test only scrambled words and vocabulary:")
    print(f"  python run_unified_eval.py --task_type textfrct --textfrct_categories CV1 V1 V2")
    print(f"  ")
    print(f"  # Test only reasoning tasks:")
    print(f"  python run_unified_eval.py --task_type textfrct --textfrct_categories RG1 RG2 RG3 RL1")
    print(f"  ")
    print(f"  # Test all categories (default):")
    print(f"  python run_unified_eval.py --task_type textfrct")


if __name__ == "__main__":
    exit(main())

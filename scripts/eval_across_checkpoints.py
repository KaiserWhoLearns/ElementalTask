#!/usr/bin/env python3
"""Evaluate a model across multiple checkpoints on all discovered tasks.

This script runs evaluation across different model checkpoints, useful for
tracking performance improvements during training.

Usage:
    # Evaluate OLMo-2-1124-7B across multiple checkpoints
    python scripts/eval_across_checkpoints.py \
        --model_id allenai/OLMo-2-1124-7B \
        --checkpoints step10000-tokens42B step50000-tokens210B main \
        --output_path results/olmo2_7b_progression \
        --tasks basic_arithmetic copying simple_icl

    # Or from a json config file
    python scripts/eval_across_checkpoints.py \
        --model_configs configs/model_checkpoints.json \
        --output_path results/multi_model_eval
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd


from tasks.registry import TaskRegistry


def check_existing_results(output_dir: Path, model_id: str, checkpoint: str, task_name: str):
    """Check if results already exist and load them if available."""
    model_name = model_id.replace('/', '_')
    chkpt_name = checkpoint.replace('/', '_')
    
    # Check for predictions file
    predictions_file = output_dir / f"{model_name}_{chkpt_name}_{task_name}.jsonl"
    
    if not predictions_file.exists() or predictions_file.stat().st_size == 0:
        return None
    
    try:
        # Load existing predictions
        import json
        predictions = []
        with open(predictions_file, 'r') as f:
            for line in f:
                predictions.append(json.loads(line))
        
        if len(predictions) == 0:
            return None
        
        print(f"  ✓ Found cached results with {len(predictions)} predictions")
        
        # Reconstruct metrics from predictions
        # This is a simplified version - actual metrics depend on task type
        metrics = {
            "cached": True,
            "predictions_file": str(predictions_file),
            "num_predictions": len(predictions)
        }
        
        return metrics
        
    except Exception as e:
        print(f"  ⚠️  Error loading cached results: {e}")
        return None


def run_single_evaluation(
    model_id: str,
    checkpoint: str,
    tasks: List[str],
    output_base: Path,
    max_new_tokens: int,
    load_vllm: bool,
    num_shots: int,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """Run evaluation for a single model checkpoint."""
    from models.evaluate_models import evaluate_model
    
    # Create checkpoint-specific output directory
    chkpt_name = checkpoint.replace('/', '_')
    output_dir = output_base / f"{model_id.split('/')[-1]}_{chkpt_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Evaluating {model_id} @ {checkpoint}")
    print(f"{'='*70}")
    
    results = []
    for i, task_name in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] Task: {task_name}")
        
        # Check for existing results
        if skip_existing:
            existing_metrics = check_existing_results(output_dir, model_id, checkpoint, task_name)
            if existing_metrics is not None:
                results.append({
                    "model": model_id,
                    "checkpoint": checkpoint,
                    "task": task_name,
                    "success": True,
                    "metrics": existing_metrics,
                    "error": "",
                    "cached": True,
                })
                continue
        
        try:
            metrics = evaluate_model(
                model_id=model_id,
                chkpt=checkpoint,
                task_name=task_name,
                output_path=str(output_dir),
                use_vllm=load_vllm,
                max_new_tokens=max_new_tokens,
                preprocess_fn=None,
                num_shots=num_shots,
            )
            
            results.append({
                "model": model_id,
                "checkpoint": checkpoint,
                "task": task_name,
                "success": True,
                "metrics": metrics,
                "error": "",
                "cached": False,
            })
            
        except Exception as e:
            print(f"❌ Failed to evaluate {task_name}: {e}")
            results.append({
                "model": model_id,
                "checkpoint": checkpoint,
                "task": task_name,
                "success": False,
                "metrics": {},
                "error": str(e),
                "cached": False,
            })
    
    return {
        "model_id": model_id,
        "checkpoint": checkpoint,
        "output_dir": str(output_dir),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models across multiple checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model specification (either individual or config file)
    parser.add_argument("--model_id", type=str,
                        help="Single model identifier (HuggingFace model ID)")
    parser.add_argument("--checkpoints", nargs="+",
                        help="List of checkpoints/revisions to evaluate")
    parser.add_argument("--model_configs", type=str,
                        help="JSON file with model->checkpoints mapping")
    
    # Evaluation settings
    parser.add_argument("--output_path", type=str, default="results/checkpoint_eval",
                        help="Base directory to save results")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Specific tasks to evaluate (default: all)")
    parser.add_argument("--skip_tasks", nargs="+", default=None,
                        help="Tasks to skip")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--load_vllm", action="store_true",
                        help="Use vLLM for faster inference")
    parser.add_argument("--num_shots", type=int, default=5,
                        help="Number of in-context learning examples")
    parser.add_argument("--force_reeval", action="store_true",
                        help="Force re-evaluation even if results exist")
    
    args = parser.parse_args()
    
    # Validate input
    if args.model_configs:
        with open(args.model_configs, 'r') as f:
            model_configs = json.load(f)
    elif args.model_id and args.checkpoints:
        model_configs = {args.model_id: args.checkpoints}
    else:
        parser.error("Must provide either --model_configs or both --model_id and --checkpoints")
    
    # Create output directory
    output_base = Path(args.output_path)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Save run configuration
    run_config = {
        "model_configs": model_configs,
        "tasks": args.tasks,
        "skip_tasks": args.skip_tasks,
        "max_new_tokens": args.max_new_tokens,
        "load_vllm": args.load_vllm,
        "num_shots": args.num_shots,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_base / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    
    print("\n" + "="*70)
    print("CHECKPOINT EVALUATION CONFIGURATION")
    print("="*70)
    print(f"  Models: {list(model_configs.keys())}")
    print(f"  Total checkpoints: {sum(len(chkpts) for chkpts in model_configs.values())}")
    print(f"  num_shots: {args.num_shots}")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  load_vllm: {args.load_vllm}")
    
    # Discover tasks
    print("\n" + "="*70)
    print("DISCOVERING TASKS")
    print("="*70)
    
    registry = TaskRegistry()
    all_tasks = registry.discover_tasks()
    
    # Filter tasks. Support subtask syntax 'task:sub1,sub2' by matching base task name.
    def _base_name(spec: str) -> str:
        return spec.split(':', 1)[0] if isinstance(spec, str) and ':' in spec else spec

    if args.tasks:
        task_names = []
        missing = []
        for spec in args.tasks:
            base = _base_name(spec)
            if base in all_tasks:
                task_names.append(spec)
            else:
                missing.append(spec)
        if missing:
            print(f"\n⚠️  Tasks not found: {set(missing)}")
    else:
        task_names = list(all_tasks.keys())
    
    if args.skip_tasks:
        # Allow skip entries with subtask syntax as well
        skip_bases = {_base_name(s) for s in args.skip_tasks}
        task_names = [t for t in task_names if _base_name(t) not in skip_bases]
        print(f"\nSkipping tasks: {args.skip_tasks}")
    
    print(f"\nTasks to evaluate ({len(task_names)}): {task_names}")
    
    # Run evaluations across all model-checkpoint combinations
    print("\n" + "="*70)
    print("RUNNING EVALUATIONS")
    print("="*70)
    
    all_results = []
    total_evals = sum(len(chkpts) for chkpts in model_configs.values())
    eval_count = 0
    
    for model_id, checkpoints in model_configs.items():
        for checkpoint in checkpoints:
            eval_count += 1
            print(f"\n{'='*70}")
            print(f"[{eval_count}/{total_evals}] {model_id} @ {checkpoint}")
            print(f"{'='*70}")
            
            result = run_single_evaluation(
                model_id=model_id,
                checkpoint=checkpoint,
                tasks=task_names,
                output_base=output_base,
                max_new_tokens=args.max_new_tokens,
                load_vllm=args.load_vllm,
                num_shots=args.num_shots,
                skip_existing=not args.force_reeval,
            )
            all_results.append(result)
    
    # Create summary DataFrames
    print("\n" + "="*70)
    print("CREATING SUMMARY")
    print("="*70)
    
    # Count cached vs new evaluations
    total_evals = sum(len(r['results']) for r in all_results)
    cached_evals = sum(1 for r in all_results for t in r['results'] if t.get('cached', False))
    new_evals = total_evals - cached_evals
    
    print(f"\n📊 Evaluation Statistics:")
    print(f"  Total evaluations: {total_evals}")
    print(f"  ✓ Cached (skipped): {cached_evals}")
    print(f"  🔄 Newly evaluated: {new_evals}")
    print(f"  Time saved: ~{cached_evals * 3} minutes (est.)")
    
    # Flatten results for detailed CSV
    detailed_rows = []
    for eval_result in all_results:
        for task_result in eval_result['results']:
            row = {
                'model': eval_result['model_id'],
                'checkpoint': eval_result['checkpoint'],
                'task': task_result['task'],
                'success': task_result['success'],
                'error': task_result['error'],
                'cached': task_result.get('cached', False),
            }
            # Add all metrics
            if task_result['success'] and task_result['metrics']:
                row.update(task_result['metrics'])
            detailed_rows.append(row)
    
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_path = output_base / "detailed_results.csv"
    detailed_df.to_csv(detailed_path, index=False)
    print(f"✓ Detailed results saved to: {detailed_path}")
    
    # Create pivot table for easy comparison
    if 'accuracy' in detailed_df.columns:
        pivot_df = detailed_df.pivot_table(
            index=['model', 'checkpoint'],
            columns='task',
            values='accuracy',
            aggfunc='first'
        )
        pivot_path = output_base / "accuracy_pivot.csv"
        pivot_df.to_csv(pivot_path)
        print(f"✓ Accuracy pivot table saved to: {pivot_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("ACCURACY SUMMARY")
        print("="*70)
        print(pivot_df.to_string())
    
    # Success summary
    success_df = detailed_df.groupby(['model', 'checkpoint'])['success'].agg(['sum', 'count'])
    success_df['success_rate'] = success_df['sum'] / success_df['count']
    
    print("\n" + "="*70)
    print("EVALUATION SUCCESS RATE")
    print("="*70)
    print(success_df.to_string())
    
    print(f"\n💾 All results saved to: {output_base}")
    
    # Generate plots if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        from scripts.plot_checkpoint_results import prepare_plot_data, plot_all_tasks_overview, plot_task_progression
        
        print("\n" + "="*70)
        print("GENERATING PLOTS")
        print("="*70)
        
        plot_dir = output_base / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        # Prepare data
        plot_df = detailed_df[detailed_df['success'] == True].copy()
        if len(plot_df) > 0 and 'accuracy' in plot_df.columns:
            plot_df = prepare_plot_data(plot_df)
            
            # Overview plot
            print("\n📊 Creating overview plot...")
            overview_path = plot_dir / "overview_all_tasks.png"
            plot_all_tasks_overview(
                df=plot_df,
                metric='accuracy',
                output_path=overview_path,
                figsize=(16, 12),
                style='seaborn',
            )
            
            # Individual task plots
            tasks = plot_df['task'].unique()
            print(f"\n📊 Creating {len(tasks)} individual task plots...")
            for task in tasks:
                task_path = plot_dir / f"{task}_accuracy.png"
                plot_task_progression(
                    df=plot_df,
                    task=task,
                    metric='accuracy',
                    output_path=task_path,
                    figsize=(12, 8),
                    style='seaborn',
                )
                matplotlib.pyplot.close('all')  # Free memory
            
            print(f"\n✅ Plots saved to: {plot_dir}")
        else:
            print("\n⚠️  No successful evaluations with accuracy metric found, skipping plots")
    
    except ImportError:
        print("\n⚠️  matplotlib not available, skipping plot generation")
        print("   Install with: pip install matplotlib seaborn")
    except Exception as e:
        print(f"\n⚠️  Error generating plots: {e}")
        print("   Results are still saved in CSV format")


if __name__ == "__main__":
    main()

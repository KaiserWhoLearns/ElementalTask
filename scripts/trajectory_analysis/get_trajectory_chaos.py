#!/usr/bin/env python3
"""Analyze the "noisiness" or "chaoticness" of accuracy trajectories.

This script computes a normalized total variation metric to measure how
chaotic/noisy a learning curve is, vs how smooth/monotonic it is.

Metrics computed:
- Total Variation (TV): Sum of absolute differences between consecutive points
- Net Change (Δ_net): Overall improvement from start to end
- Smoothness (S): |Δ_net| / TV (1 = perfectly monotonic, 0 = pure noise)
- Chaoticness (C_tv): 1 - S (0 = perfectly monotonic, 1 = pure noise)

A perfectly monotonic increasing curve has S=1, C_tv=0.
A curve that oscillates wildly but ends up at the same place has S≈0, C_tv≈1.

Usage:
    python scripts/trajectory_analysis/get_trajectory_chaos.py \
        --results_dir results/olmo2_continuous_1b_early_revised \
        --output chaos_metrics.csv
    
    # Compare multiple models
    python scripts/trajectory_analysis/get_trajectory_chaos.py \
        --results_dirs results/olmo2_continuous_1b_early_revised results/olmo2_continuous_7b_early_revised \
        --model_names "1B" "7B" \
        --output chaos_comparison.csv
"""

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd


def extract_tokens_from_checkpoint(checkpoint: str) -> Optional[int]:
    """Extract token count (in billions) from checkpoint name."""
    if checkpoint == "main":
        return None
    match = re.search(r'tokens(\d+)B', checkpoint)
    if match:
        return int(match.group(1))
    return None


def load_accuracy_data(pivot_file: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load and sort accuracy data from a pivot file.
    
    Returns:
        tokens: Array of token counts (in billions)
        accuracy: Array of accuracy values
        task_name: Name of the task
    """
    df = pd.read_csv(pivot_file)
    
    # Get task name from column (third column after model, checkpoint)
    task_name = [c for c in df.columns if c not in ['model', 'checkpoint']][0]
    
    # Extract tokens and filter out None/main
    df['tokens'] = df['checkpoint'].apply(extract_tokens_from_checkpoint)
    df = df[df['tokens'].notna() & (df['tokens'] > 0)]
    df = df.sort_values('tokens')
    
    tokens = df['tokens'].values.astype(float)
    accuracy = df[task_name].values.astype(float)
    
    return tokens, accuracy, task_name


def compute_chaos_metrics(accuracy: np.ndarray) -> Dict[str, float]:
    """Compute chaoticness/smoothness metrics for a trajectory.
    
    Args:
        accuracy: Array of accuracy values (unsmoothed, sorted by tokens)
    
    Returns:
        Dict with:
        - total_variation: Sum of |Δ_i|
        - net_change: y_n - y_1
        - smoothness: |Δ_net| / TV (1 = monotonic, 0 = chaotic)
        - chaoticness: 1 - smoothness
        - n_checkpoints: Number of checkpoints
        - n_increases: Number of times accuracy increased
        - n_decreases: Number of times accuracy decreased
    """
    if len(accuracy) < 2:
        return {
            "total_variation": 0.0,
            "net_change": 0.0,
            "smoothness": 1.0,
            "chaoticness": 0.0,
            "n_checkpoints": len(accuracy),
            "n_increases": 0,
            "n_decreases": 0,
        }
    
    # First differences
    deltas = np.diff(accuracy)
    
    # Total variation
    total_variation = np.sum(np.abs(deltas))
    
    # Net change
    net_change = accuracy[-1] - accuracy[0]
    
    # Smoothness and chaoticness
    if total_variation == 0:
        # Perfectly flat curve
        smoothness = 1.0
        chaoticness = 0.0
    else:
        smoothness = abs(net_change) / total_variation
        chaoticness = 1.0 - smoothness
    
    # Count increases and decreases
    n_increases = np.sum(deltas > 0)
    n_decreases = np.sum(deltas < 0)
    
    return {
        "total_variation": float(total_variation),
        "net_change": float(net_change),
        "smoothness": float(smoothness),
        "chaoticness": float(chaoticness),
        "n_checkpoints": len(accuracy),
        "n_increases": int(n_increases),
        "n_decreases": int(n_decreases),
    }


def analyze_single_dir(
    results_dir: Path,
    min_max_accuracy: float = 0.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """Analyze chaos metrics for all tasks in a results directory.
    
    Args:
        results_dir: Directory containing accuracy_pivot_*.csv files
        min_max_accuracy: Skip tasks with max accuracy below this threshold
    
    Returns:
        Tuple of:
        - DataFrame with chaos metrics for all tasks
        - List of skipped task names
    """
    results = []
    skipped_tasks = []
    
    for pivot_file in sorted(results_dir.glob("accuracy_pivot_*.csv")):
        try:
            tokens, accuracy, task_name = load_accuracy_data(pivot_file)
            
            if len(tokens) == 0:
                continue
            
            max_acc = float(accuracy.max())
            
            # Skip tasks with trivial performance
            if max_acc <= min_max_accuracy:
                skipped_tasks.append(task_name)
                continue
            
            metrics = compute_chaos_metrics(accuracy)
            
            results.append({
                "task": task_name,
                "max_accuracy": max_acc,
                "final_accuracy": float(accuracy[-1]),
                "start_accuracy": float(accuracy[0]),
                **metrics,
            })
            
        except Exception as e:
            print(f"Warning: Failed to process {pivot_file.name}: {e}")
    
    return pd.DataFrame(results), skipped_tasks


def analyze_multiple_dirs(
    results_dirs: List[str],
    model_names: List[str],
    min_max_accuracy: float = 0.0,
    tasks: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Analyze chaos metrics for all tasks across multiple models.
    
    Args:
        results_dirs: List of results directories (one per model)
        model_names: List of model names (same order as results_dirs)
        min_max_accuracy: Skip tasks below this threshold
        tasks: Optional list of specific tasks to analyze
    
    Returns:
        Tuple of (results DataFrame, skipped tasks list)
    """
    # Discover all tasks across all directories
    all_task_files = {}  # sanitized_task -> {model_name -> pivot_file}
    
    for results_dir, model_name in zip(results_dirs, model_names):
        results_path = Path(results_dir)
        if not results_path.exists():
            print(f"Warning: {results_path} does not exist")
            continue
        
        for pivot_file in results_path.glob("accuracy_pivot_*.csv"):
            prefix = "accuracy_pivot_"
            sanitized_task = pivot_file.stem.replace(prefix, "")
            
            if sanitized_task not in all_task_files:
                all_task_files[sanitized_task] = {}
            all_task_files[sanitized_task][model_name] = pivot_file
    
    # Filter to requested tasks if specified
    if tasks:
        sanitized_requested = {t.replace(":", "_").replace("/", "_") for t in tasks}
        all_task_files = {k: v for k, v in all_task_files.items() if k in sanitized_requested}
    
    results = []
    skipped_tasks = []
    
    for sanitized_task, model_files in sorted(all_task_files.items()):
        task_name = None
        max_acc_any = 0.0
        
        # First pass: check max accuracy across all models
        for model_name, pivot_file in model_files.items():
            try:
                _, accuracy, task_name = load_accuracy_data(pivot_file)
                if len(accuracy) > 0:
                    max_acc_any = max(max_acc_any, accuracy.max())
            except:
                pass
        
        # Skip if max accuracy is too low
        if max_acc_any <= min_max_accuracy:
            skipped_tasks.append(task_name or sanitized_task)
            continue
        
        display_name = task_name or sanitized_task.replace("_", ":", 1)
        
        # Analyze each model
        for model_name, pivot_file in model_files.items():
            try:
                tokens, accuracy, _ = load_accuracy_data(pivot_file)
                if len(tokens) == 0:
                    continue
                
                metrics = compute_chaos_metrics(accuracy)
                
                results.append({
                    "task": display_name,
                    "model": model_name,
                    "max_accuracy": float(accuracy.max()),
                    "final_accuracy": float(accuracy[-1]),
                    "start_accuracy": float(accuracy[0]),
                    **metrics,
                })
            except Exception as e:
                print(f"Warning: Failed to process {pivot_file}: {e}")
    
    return pd.DataFrame(results), skipped_tasks


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trajectory chaoticness/smoothness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Metrics explained:
  total_variation  - Sum of |Δ_i| (total up-and-down movement)
  net_change       - Final - Initial accuracy
  smoothness       - |net_change| / total_variation (1 = monotonic)
  chaoticness      - 1 - smoothness (0 = smooth, 1 = chaotic)
  n_increases      - Number of steps where accuracy increased
  n_decreases      - Number of steps where accuracy decreased

Examples:
  # Analyze single model
  python get_trajectory_chaos.py -d results/1b -o chaos_1b.csv

  # Compare multiple models
  python get_trajectory_chaos.py \\
      --results_dirs results/1b results/7b \\
      --model_names "1B" "7B" \\
      -o chaos_comparison.csv

  # Only analyze tasks with >50% max accuracy
  python get_trajectory_chaos.py -d results/1b --min-accuracy 0.5
        """
    )
    
    # Single model mode
    parser.add_argument("-d", "--results_dir", type=str, default=None,
                        help="Directory containing accuracy_pivot_*.csv files")
    
    # Multi-model mode
    parser.add_argument("--results_dirs", nargs="+", default=None,
                        help="Multiple results directories (one per model)")
    parser.add_argument("--model_names", nargs="+", default=None,
                        help="Names for each model (same order as results_dirs)")
    
    # Filtering
    parser.add_argument("--min-accuracy", type=float, default=0.0,
                        help="Skip tasks with max accuracy at or below this value")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Specific tasks to analyze")
    
    # Output
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output CSV file (default: print to stdout)")
    parser.add_argument("--sort-by", type=str, default="chaoticness",
                        choices=["chaoticness", "smoothness", "total_variation", "task"],
                        help="Column to sort results by (default: chaoticness)")
    parser.add_argument("--ascending", action="store_true",
                        help="Sort in ascending order (default: descending for metrics)")
    
    args = parser.parse_args()
    
    # Validate input modes
    if args.results_dirs and args.results_dir:
        parser.error("Cannot use both --results_dir and --results_dirs")
    
    if args.results_dirs:
        if not args.model_names or len(args.model_names) != len(args.results_dirs):
            parser.error("--model_names must match --results_dirs in length")
        multi_model = True
    elif args.results_dir:
        multi_model = False
    else:
        parser.error("Must provide either --results_dir or --results_dirs")
    
    print(f"Min accuracy filter: {args.min_accuracy}")
    print()
    
    if multi_model:
        print(f"Models: {args.model_names}")
        print(f"Results dirs: {args.results_dirs}")
        print()
        
        df, skipped_tasks = analyze_multiple_dirs(
            results_dirs=args.results_dirs,
            model_names=args.model_names,
            min_max_accuracy=args.min_accuracy,
            tasks=args.tasks,
        )
    else:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"Error: Directory {results_dir} does not exist")
            return 1
        
        print(f"Results dir: {results_dir}")
        print()
        
        df, skipped_tasks = analyze_single_dir(
            results_dir,
            min_max_accuracy=args.min_accuracy,
        )
    
    # Report skipped tasks
    if skipped_tasks:
        print(f"⚠️  Skipped {len(skipped_tasks)} tasks with max accuracy <= {args.min_accuracy}:")
        for task in skipped_tasks[:10]:  # Show first 10
            print(f"    - {task}")
        if len(skipped_tasks) > 10:
            print(f"    ... and {len(skipped_tasks) - 10} more")
        print()
    
    if df.empty:
        print("No tasks to analyze!")
        return 1
    
    # Sort results
    ascending = args.ascending if args.sort_by == "task" else args.ascending
    if args.sort_by != "task" and not args.ascending:
        ascending = False  # Default descending for metrics
    df = df.sort_values(args.sort_by, ascending=ascending, na_position="last")
    
    # Select columns for display
    if multi_model:
        display_cols = ["task", "model", "chaoticness", "smoothness", "total_variation", 
                        "net_change", "max_accuracy", "n_increases", "n_decreases"]
    else:
        display_cols = ["task", "chaoticness", "smoothness", "total_variation",
                        "net_change", "max_accuracy", "n_increases", "n_decreases"]
    
    df_display = df[[c for c in display_cols if c in df.columns]]
    
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")
    
    print(df_display.to_string(index=False))
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Total tasks analyzed: {len(df)}")
    print(f"Mean chaoticness: {df['chaoticness'].mean():.4f}")
    print(f"Median chaoticness: {df['chaoticness'].median():.4f}")
    print(f"Most chaotic: {df.loc[df['chaoticness'].idxmax(), 'task']} ({df['chaoticness'].max():.4f})")
    print(f"Most smooth: {df.loc[df['chaoticness'].idxmin(), 'task']} ({df['chaoticness'].min():.4f})")
    
    # Per-model statistics (if multi-model)
    if multi_model and 'model' in df.columns:
        print(f"\n{'='*60}")
        print("PER-MODEL CHAOTICNESS")
        print(f"{'='*60}")
        model_stats = df.groupby('model')['chaoticness'].agg(['mean', 'median', 'std', 'min', 'max'])
        model_stats.columns = ['mean', 'median', 'std', 'min', 'max']
        print(model_stats.to_string())
    
    return 0


if __name__ == "__main__":
    exit(main())

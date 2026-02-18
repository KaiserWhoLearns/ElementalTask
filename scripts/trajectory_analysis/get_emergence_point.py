#!/usr/bin/env python3
"""Find emergence points for tasks based on accuracy trajectories.

This script analyzes accuracy over training tokens to determine when
a task "emerges" - i.e., when the model starts to successfully perform it.

Supports multiple detection methods:
- fixed: First time accuracy exceeds a fixed threshold (e.g., 0.95)
- relative: First time accuracy exceeds X% of maximum performance
- elbow: Elbow/knee point using Kneedle algorithm (recommended for emergence)
- stable: First time accuracy stays above threshold for N consecutive checkpoints

Usage:
    python scripts/trajectory_analysis/get_emergence_point.py \
        --results_dir results/olmo2_continuous_1b_early_revised \
        --method relative \
        --threshold 0.5 \
        --output emergence_points.csv
    
    # Plot with multiple models
    python scripts/trajectory_analysis/get_emergence_point.py \
        --results_dirs results/olmo2_continuous_1b_early_revised results/olmo2_continuous_7b_early_revised \
        --model_names "1B" "7B" \
        --method elbow \
        --plot --plot_dir plots/emergence
"""

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from kneed import KneeLocator


# Plotting colors and styles
MODEL_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
MODEL_MARKERS = ['o', 's', '^', 'D', 'v']


def extract_tokens_from_checkpoint(checkpoint: str) -> Optional[float]:
    """Extract token count (in billions) from checkpoint name.

    Supports multiple formats:
    - OLMo: 'stage1-step100000-tokens210B' -> 210
    - K2-V2: 'base_1245000' -> 12450B (12.45T) - checkpoint number in units of 10M tokens
    - Generic: 'tokens100B' -> 100
    """
    if checkpoint in ("main", "base_final", "final"):
        return None

    # Try explicit token count first (e.g., 'tokens210B')
    match = re.search(r'tokens(\d+)B', checkpoint)
    if match:
        return int(match.group(1))

    # Try K2-V2 format: 'base_XXXXXXX' (checkpoint number in units of ~10M tokens)
    # e.g., base_1245000 = 1,245,000 * 10M = 12.45T tokens = 12450B
    match = re.search(r'base_(\d+)', checkpoint)
    if match:
        checkpoint_num = int(match.group(1))
        # Each unit = 10M tokens = 0.01B tokens
        tokens_b = checkpoint_num * 0.01
        return tokens_b

    # Try generic step format: 'step100000'
    match = re.search(r'step(\d+)', checkpoint)
    if match:
        # If no token info, return step number as proxy (will be sorted correctly)
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


def load_combined_pivot(pivot_file: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load data from a combined accuracy_pivot.csv with multiple tasks as columns.

    Args:
        pivot_file: Path to combined pivot file with columns: model, checkpoint, task1, task2, ...

    Returns:
        Dictionary mapping task_name -> (tokens_array, accuracy_array)
    """
    df = pd.read_csv(pivot_file)

    # Extract tokens from checkpoint names
    df['tokens'] = df['checkpoint'].apply(extract_tokens_from_checkpoint)
    df = df[df['tokens'].notna() & (df['tokens'] > 0)]
    df = df.sort_values('tokens')

    if len(df) == 0:
        return {}

    # Get task columns (everything except model, checkpoint, tokens)
    task_columns = [c for c in df.columns if c not in ['model', 'checkpoint', 'tokens']]

    result = {}
    tokens = df['tokens'].values.astype(float)

    for task_name in task_columns:
        accuracy = df[task_name].values.astype(float)
        result[task_name] = (tokens, accuracy)

    return result


def smooth_trajectory(accuracy: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian smoothing to accuracy trajectory."""
    if sigma <= 0:
        return accuracy
    return gaussian_filter1d(accuracy, sigma=sigma)


def find_emergence_fixed(
    tokens: np.ndarray,
    accuracy: np.ndarray,
    threshold: float = 0.95,
) -> Optional[float]:
    """Find first token count where accuracy exceeds fixed threshold.
    
    Args:
        tokens: Token counts
        accuracy: Accuracy values (possibly smoothed)
        threshold: Fixed accuracy threshold (e.g., 0.95)
    
    Returns:
        Token count at emergence, or None if never reached
    """
    mask = accuracy >= threshold
    if not mask.any():
        return None
    return float(tokens[mask.argmax()])


def find_emergence_relative(
    tokens: np.ndarray,
    accuracy: np.ndarray,
    fraction: float = 0.5,
    use_max: bool = True,
) -> Optional[float]:
    """Find first token count where accuracy exceeds fraction of final/max.
    
    Args:
        tokens: Token counts
        accuracy: Accuracy values (possibly smoothed)
        fraction: Fraction of reference performance (e.g., 0.5 = 50%)
        use_max: If True, use max accuracy as reference; else use final
    
    Returns:
        Token count at emergence, or None if never reached
    """
    reference = accuracy.max() if use_max else accuracy[-1]
    target = reference * fraction
    
    mask = accuracy >= target
    if not mask.any():
        return None
    return float(tokens[mask.argmax()])


def find_emergence_inflection(
    tokens: np.ndarray,
    accuracy: np.ndarray,
    smoothing_factor: float = 0.1,
) -> Optional[float]:
    """Find elbow/knee point of learning curve using Kneedle algorithm.
    
    This finds the point where the curve transitions from rapid improvement
    to slower improvement (or vice versa) - the "elbow" of the curve.
    
    For accuracy curves (increasing), we look for a "knee" (concave curve).
    
    Args:
        tokens: Token counts
        accuracy: Accuracy values (possibly smoothed)
        smoothing_factor: Sensitivity parameter (S) for Kneedle. 
                          Higher = less sensitive, finds more prominent knees.
    
    Returns:
        Token count at elbow/knee point, or None if cannot be determined
    """
    if len(tokens) < 4:
        return None
    
    try:
        # KneeLocator finds the elbow/knee point
        # curve="concave" for accuracy (starts slow, then fast, then slow)
        # direction="increasing" for accuracy curves
        # S is the sensitivity parameter
        kneedle = KneeLocator(
            tokens, 
            accuracy, 
            curve="concave",
            direction="increasing",
            S=smoothing_factor,
            interp_method="polynomial",
        )
        
        if kneedle.knee is not None:
            return float(kneedle.knee)
        
        # If no knee found with concave, try convex (for S-shaped curves)
        kneedle_convex = KneeLocator(
            tokens,
            accuracy,
            curve="convex", 
            direction="increasing",
            S=smoothing_factor,
            interp_method="polynomial",
        )
        
        if kneedle_convex.knee is not None:
            return float(kneedle_convex.knee)
        
        return None
    except Exception:
        return None


def find_emergence_stable(
    tokens: np.ndarray,
    accuracy: np.ndarray,
    threshold: float = 0.5,
    min_consecutive: int = 3,
) -> Optional[float]:
    """Find first token count where accuracy stays above threshold for N checkpoints.
    
    This is more robust to noise than single-crossing methods.
    
    Args:
        tokens: Token counts
        accuracy: Accuracy values
        threshold: Accuracy threshold to maintain
        min_consecutive: Minimum consecutive checkpoints above threshold
    
    Returns:
        Token count at stable emergence, or None if never reached
    """
    above = accuracy >= threshold
    
    # Find runs of consecutive True values
    for i in range(len(above) - min_consecutive + 1):
        if all(above[i:i + min_consecutive]):
            return float(tokens[i])
    
    return None


def find_emergence(
    tokens: np.ndarray,
    accuracy: np.ndarray,
    method: str = "relative",
    smooth_sigma: float = 1.0,
    **kwargs,
) -> Dict[str, Any]:
    """Find emergence point using specified method.
    
    Args:
        tokens: Token counts (in billions)
        accuracy: Raw accuracy values
        method: One of 'fixed', 'relative', 'inflection', 'stable'
        smooth_sigma: Gaussian smoothing sigma (0 = no smoothing)
        **kwargs: Method-specific parameters
    
    Returns:
        Dict with:
        - emergence_tokens: Token count at emergence (None if not found)
        - method: Method used
        - parameters: Parameters used
        - max_accuracy: Maximum accuracy achieved
        - final_accuracy: Final accuracy
    """
    # Apply smoothing
    accuracy_smooth = smooth_trajectory(accuracy, sigma=smooth_sigma)
    
    # Find emergence based on method
    if method == "fixed":
        threshold = kwargs.get("threshold", 0.95)
        emergence = find_emergence_fixed(tokens, accuracy_smooth, threshold=threshold)
        params = {"threshold": threshold}
        
    elif method == "relative":
        fraction = kwargs.get("threshold", 0.5)  # Use threshold param for fraction
        use_max = kwargs.get("use_max", True)
        emergence = find_emergence_relative(tokens, accuracy_smooth, fraction=fraction, use_max=use_max)
        params = {"fraction": fraction, "use_max": use_max}
        
    elif method == "inflection" or method == "elbow":
        smoothing_factor = kwargs.get("smoothing_factor", 1.0)  # Kneedle S parameter
        emergence = find_emergence_inflection(tokens, accuracy_smooth, smoothing_factor=smoothing_factor)
        params = {"sensitivity": smoothing_factor}
        
    elif method == "stable":
        threshold = kwargs.get("threshold", 0.5)
        min_consecutive = kwargs.get("min_consecutive", 3)
        emergence = find_emergence_stable(tokens, accuracy_smooth, threshold=threshold, min_consecutive=min_consecutive)
        params = {"threshold": threshold, "min_consecutive": min_consecutive}
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'fixed', 'relative', 'inflection', or 'stable'.")
    
    return {
        "emergence_tokens": emergence,
        "method": method,
        "parameters": params,
        "smooth_sigma": smooth_sigma,
        "max_accuracy": float(accuracy.max()),
        "final_accuracy": float(accuracy[-1]),
        "n_checkpoints": len(tokens),
    }


def analyze_all_tasks(
    results_dir: Path,
    method: str = "relative",
    smooth_sigma: float = 1.0,
    min_max_accuracy: float = 0.0,
    **kwargs,
) -> Tuple[pd.DataFrame, List[str]]:
    """Analyze emergence points for all tasks in a results directory.

    Args:
        results_dir: Directory containing accuracy_pivot_*.csv or accuracy_pivot.csv
        method: Emergence detection method
        smooth_sigma: Smoothing parameter
        min_max_accuracy: Skip tasks with max accuracy below this threshold
        **kwargs: Method-specific parameters

    Returns:
        Tuple of:
        - DataFrame with emergence data for all tasks
        - List of skipped task names (due to trivial performance)
    """
    results = []
    skipped_tasks = []

    # First try per-task pivot files
    pivot_files = sorted(results_dir.glob("accuracy_pivot_*.csv"))

    if pivot_files:
        # Use per-task files
        for pivot_file in pivot_files:
            try:
                tokens, accuracy, task_name = load_accuracy_data(pivot_file)

                if len(tokens) == 0:
                    continue

                max_acc = float(accuracy.max())

                # Skip tasks with trivial performance
                if max_acc <= min_max_accuracy:
                    skipped_tasks.append(task_name)
                    continue

                emergence_data = find_emergence(
                    tokens, accuracy,
                    method=method,
                    smooth_sigma=smooth_sigma,
                    **kwargs
                )

                results.append({
                    "task": task_name,
                    "emergence_tokens_B": emergence_data["emergence_tokens"],
                    "max_accuracy": emergence_data["max_accuracy"],
                    "final_accuracy": emergence_data["final_accuracy"],
                    "n_checkpoints": emergence_data["n_checkpoints"],
                })

            except Exception as e:
                print(f"Warning: Failed to process {pivot_file.name}: {e}")
    else:
        # Try combined pivot file
        combined_file = results_dir / "accuracy_pivot.csv"
        if combined_file.exists():
            print(f"Using combined pivot file: {combined_file}")
            task_data = load_combined_pivot(combined_file)

            for task_name, (tokens, accuracy) in task_data.items():
                if len(tokens) == 0:
                    continue

                max_acc = float(accuracy.max())

                if max_acc <= min_max_accuracy:
                    skipped_tasks.append(task_name)
                    continue

                emergence_data = find_emergence(
                    tokens, accuracy,
                    method=method,
                    smooth_sigma=smooth_sigma,
                    **kwargs
                )

                results.append({
                    "task": task_name,
                    "emergence_tokens_B": emergence_data["emergence_tokens"],
                    "max_accuracy": emergence_data["max_accuracy"],
                    "final_accuracy": emergence_data["final_accuracy"],
                    "n_checkpoints": emergence_data["n_checkpoints"],
                })

    return pd.DataFrame(results), skipped_tasks


def plot_task_emergence(
    task_name: str,
    model_data: Dict[str, Tuple[np.ndarray, np.ndarray]],  # model_name -> (tokens, accuracy)
    method: str,
    smooth_sigma: float,
    output_path: Path,
    figsize: Tuple[int, int] = (12, 8),
    **kwargs,
):
    """Plot emergence analysis for a single task with multiple models.
    
    Shows:
    - Raw data points (faint)
    - Smoothed curve (solid line)
    - Emergence point (vertical dashed line)
    - Inflection point (vertical red line, if method is inflection)
    
    Args:
        task_name: Name of the task for title
        model_data: Dict mapping model_name -> (tokens, accuracy)
        method: Emergence detection method
        smooth_sigma: Smoothing parameter used
        output_path: Where to save the plot
        figsize: Figure size
        **kwargs: Method-specific parameters
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for idx, (model_name, (tokens, accuracy)) in enumerate(sorted(model_data.items())):
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        marker = MODEL_MARKERS[idx % len(MODEL_MARKERS)]
        
        # Plot raw data (faint)
        ax.scatter(tokens, accuracy, 
                   color=color, marker=marker, alpha=0.3, s=40,
                   label=f"{model_name} (raw)")
        
        # Plot smoothed curve
        accuracy_smooth = smooth_trajectory(accuracy, sigma=smooth_sigma)
        ax.plot(tokens, accuracy_smooth,
                color=color, linewidth=2, alpha=0.9,
                label=f"{model_name} (smoothed)")
        
        # Find and plot emergence point
        emergence_result = find_emergence(
            tokens, accuracy,
            method=method,
            smooth_sigma=smooth_sigma,
            **kwargs
        )
        emergence_tokens = emergence_result["emergence_tokens"]
        
        if emergence_tokens is not None:
            ax.axvline(x=emergence_tokens, color=color, linestyle='--', alpha=0.7,
                       label=f"{model_name} emergence: {emergence_tokens:.0f}B")
        
        # If using inflection method, also show the inflection point specifically
        # Or always show inflection for reference
        inflection_tokens = find_emergence_inflection(tokens, accuracy_smooth)
        if inflection_tokens is not None and method != "inflection":
            # Show inflection as a subtle marker if not the primary method
            ax.axvline(x=inflection_tokens, color=color, linestyle=':', alpha=0.4)
    
    ax.set_xlabel("Training Tokens (B)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{task_name}: Emergence Analysis ({method} method)", fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_tasks_emergence(
    results_dirs: List[str],
    model_names: List[str],
    method: str,
    smooth_sigma: float,
    output_dir: Path,
    min_max_accuracy: float = 0.0,
    tasks: Optional[List[str]] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, List[str]]:
    """Plot emergence for all tasks across multiple models.

    Args:
        results_dirs: List of results directories (one per model)
        model_names: List of model names (same order as results_dirs)
        method: Emergence detection method
        smooth_sigma: Smoothing parameter
        output_dir: Directory to save plots
        min_max_accuracy: Skip tasks below this threshold
        tasks: Optional list of specific tasks to plot
        **kwargs: Method-specific parameters

    Returns:
        Tuple of (results DataFrame, skipped tasks list)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all tasks across all directories
    # all_task_data: task_name -> {model_name -> (tokens, accuracy)}
    all_task_data = {}

    for results_dir, model_name in zip(results_dirs, model_names):
        results_path = Path(results_dir)
        if not results_path.exists():
            print(f"Warning: {results_path} does not exist")
            continue

        # First try per-task pivot files
        pivot_files = list(results_path.glob("accuracy_pivot_*.csv"))

        if pivot_files:
            for pivot_file in pivot_files:
                # Extract task name from filename
                prefix = "accuracy_pivot_"
                sanitized_task = pivot_file.stem.replace(prefix, "")

                try:
                    tokens, accuracy, task_name = load_accuracy_data(pivot_file)
                    if len(tokens) > 0:
                        if sanitized_task not in all_task_data:
                            all_task_data[sanitized_task] = {"_task_name": task_name}
                        all_task_data[sanitized_task][model_name] = (tokens, accuracy)
                except Exception as e:
                    print(f"Warning: Failed to load {pivot_file}: {e}")
        else:
            # Try combined pivot file
            combined_file = results_path / "accuracy_pivot.csv"
            if combined_file.exists():
                print(f"Using combined pivot file for {model_name}: {combined_file}")
                task_data = load_combined_pivot(combined_file)

                for task_name, (tokens, accuracy) in task_data.items():
                    sanitized_task = task_name.replace(":", "_").replace("/", "_")
                    if sanitized_task not in all_task_data:
                        all_task_data[sanitized_task] = {"_task_name": task_name}
                    all_task_data[sanitized_task][model_name] = (tokens, accuracy)

    # Filter to requested tasks if specified
    if tasks:
        sanitized_requested = {t.replace(":", "_").replace("/", "_") for t in tasks}
        all_task_data = {k: v for k, v in all_task_data.items() if k in sanitized_requested}

    results = []
    skipped_tasks = []

    print(f"Processing {len(all_task_data)} tasks...")

    for sanitized_task, task_info in sorted(all_task_data.items()):
        # Extract task name and model data
        task_name = task_info.pop("_task_name", None)
        model_data = {k: v for k, v in task_info.items() if k != "_task_name"}

        if not model_data:
            continue

        max_acc_any = max(acc.max() for _, acc in model_data.values())

        # Skip if max accuracy is too low
        if max_acc_any <= min_max_accuracy:
            skipped_tasks.append(task_name or sanitized_task)
            continue

        display_name = task_name or sanitized_task.replace("_", ":", 1)
        print(f"  {display_name}")

        # Create plot
        output_path = output_dir / f"{sanitized_task}_emergence.png"
        plot_task_emergence(
            display_name,
            model_data,
            method=method,
            smooth_sigma=smooth_sigma,
            output_path=output_path,
            **kwargs
        )

        # Collect results for each model
        for model_name, (tokens, accuracy) in model_data.items():
            emergence_result = find_emergence(
                tokens, accuracy,
                method=method,
                smooth_sigma=smooth_sigma,
                **kwargs
            )
            results.append({
                "task": display_name,
                "model": model_name,
                "emergence_tokens_B": emergence_result["emergence_tokens"],
                "max_accuracy": emergence_result["max_accuracy"],
                "final_accuracy": emergence_result["final_accuracy"],
            })

    return pd.DataFrame(results), skipped_tasks


def main():
    parser = argparse.ArgumentParser(
        description="Find emergence points for tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods:
  fixed      - First time accuracy >= threshold (default: 0.95)
  relative   - First time accuracy >= fraction of max (default: 0.5)
  elbow      - Elbow/knee point using Kneedle algorithm (recommended for emergence)
  stable     - First time accuracy stays >= threshold for N checkpoints

Examples:
  # Find when tasks reach 50% of their max performance
  python get_emergence_point.py -d results/1b -m relative -t 0.5

  # Find when tasks reach 95% accuracy
  python get_emergence_point.py -d results/1b -m fixed -t 0.95

  # Find elbow points (emergence)
  python get_emergence_point.py -d results/1b -m elbow

  # Find stable emergence (above 0.5 for 3+ checkpoints)
  python get_emergence_point.py -d results/1b -m stable -t 0.5 --min-consecutive 3

  # Plot multiple models with emergence points
  python get_emergence_point.py \\
      --results_dirs results/1b results/7b \\
      --model_names "1B" "7B" \\
      -m relative -t 0.5 \\
      --plot --plot_dir plots/emergence
        """
    )
    
    # Single model mode
    parser.add_argument("-d", "--results_dir", type=str, default=None,
                        help="Directory containing accuracy_pivot_*.csv files (single model mode)")
    
    # Multi-model mode
    parser.add_argument("--results_dirs", nargs="+", default=None,
                        help="Multiple results directories (one per model)")
    parser.add_argument("--model_names", nargs="+", default=None,
                        help="Names for each model (same order as results_dirs)")
    
    # Method settings
    parser.add_argument("-m", "--method", type=str, default="relative",
                        choices=["fixed", "relative", "elbow", "inflection", "stable"],
                        help="Emergence detection method (default: relative)")
    parser.add_argument("-t", "--threshold", type=float, default=0.5,
                        help="Threshold value (meaning depends on method)")
    parser.add_argument("-s", "--smooth-sigma", type=float, default=1.0,
                        help="Gaussian smoothing sigma (0 = no smoothing)")
    parser.add_argument("--sensitivity", type=float, default=1.0,
                        help="Sensitivity (S) for elbow method. Higher = less sensitive (default: 1.0)")
    parser.add_argument("--min-consecutive", type=int, default=3,
                        help="Min consecutive checkpoints for 'stable' method")
    parser.add_argument("--use-final", action="store_true",
                        help="For 'relative' method, use final acc instead of max")
    
    # Output settings
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output CSV file (default: print to stdout)")
    parser.add_argument("--task", type=str, default=None,
                        help="Analyze single task only")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Specific tasks to analyze/plot")
    parser.add_argument("--min-accuracy", type=float, default=0.0,
                        help="Skip tasks with max accuracy at or below this value (default: 0)")
    
    # Plotting
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots for each task")
    parser.add_argument("--plot_dir", type=str, default="plots/emergence",
                        help="Directory to save plots (default: plots/emergence)")
    
    args = parser.parse_args()
    
    # Validate input modes
    if args.results_dirs and args.results_dir:
        parser.error("Cannot use both --results_dir and --results_dirs")
    
    if args.results_dirs:
        # Multi-model mode
        if not args.model_names or len(args.model_names) != len(args.results_dirs):
            parser.error("--model_names must match --results_dirs in length")
        multi_model = True
    elif args.results_dir:
        # Single model mode
        multi_model = False
    else:
        parser.error("Must provide either --results_dir or --results_dirs")
    
    # Build kwargs for method
    kwargs = {
        "threshold": args.threshold,
        "use_max": not args.use_final,
        "min_consecutive": args.min_consecutive,
        "smoothing_factor": args.sensitivity,
    }
    
    print(f"Method: {args.method}")
    print(f"Threshold: {args.threshold}")
    print(f"Smoothing sigma: {args.smooth_sigma}")
    print(f"Min accuracy filter: {args.min_accuracy}")
    if args.plot:
        print(f"Plot directory: {args.plot_dir}")
    print()
    
    if multi_model:
        # Multi-model mode with plotting
        print(f"Models: {args.model_names}")
        print(f"Results dirs: {args.results_dirs}")
        print()
        
        df, skipped_tasks = plot_all_tasks_emergence(
            results_dirs=args.results_dirs,
            model_names=args.model_names,
            method=args.method,
            smooth_sigma=args.smooth_sigma,
            output_dir=Path(args.plot_dir),
            min_max_accuracy=args.min_accuracy,
            tasks=args.tasks,
            **kwargs
        )
        
        # Report skipped tasks
        if skipped_tasks:
            print(f"\n⚠️  Skipped {len(skipped_tasks)} tasks with max accuracy <= {args.min_accuracy}:")
            for task in skipped_tasks:
                print(f"    - {task}")
        
        # Sort by emergence point (earliest first)
        df = df.sort_values(["emergence_tokens_B", "task", "model"], na_position="last")
        
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nSaved results to {args.output}")
        else:
            print(f"\n{df.to_string(index=False)}")
        
        print(f"\n✅ Plots saved to: {args.plot_dir}")
        
    elif args.task:
        # Single task analysis
        results_dir = Path(args.results_dir)
        pivot_file = results_dir / f"accuracy_pivot_{args.task.replace(':', '_')}.csv"
        if not pivot_file.exists():
            print(f"Error: {pivot_file} not found")
            return 1
        
        tokens, accuracy, task_name = load_accuracy_data(pivot_file)
        
        # Check for trivial performance
        if accuracy.max() <= args.min_accuracy:
            print(f"Task: {task_name}")
            print(f"⚠️  Skipped: max accuracy ({accuracy.max():.4f}) <= {args.min_accuracy}")
            return 0
        
        result = find_emergence(tokens, accuracy, method=args.method, 
                               smooth_sigma=args.smooth_sigma, **kwargs)
        
        print(f"Task: {task_name}")
        print(f"Emergence at: {result['emergence_tokens']} B tokens")
        print(f"Max accuracy: {result['max_accuracy']:.4f}")
        print(f"Final accuracy: {result['final_accuracy']:.4f}")
        print(f"Parameters: {result['parameters']}")
        
        # Optionally plot single task
        if args.plot:
            plot_dir = Path(args.plot_dir)
            plot_dir.mkdir(parents=True, exist_ok=True)
            output_path = plot_dir / f"{args.task.replace(':', '_')}_emergence.png"
            plot_task_emergence(
                task_name,
                {"Model": (tokens, accuracy)},
                method=args.method,
                smooth_sigma=args.smooth_sigma,
                output_path=output_path,
                **kwargs
            )
            print(f"\n✅ Plot saved to: {output_path}")
        
    else:
        # Analyze all tasks (single model, no plotting or with plotting)
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"Error: Directory {results_dir} does not exist")
            return 1
        
        print(f"Results dir: {results_dir}")
        print()
        
        if args.plot:
            # Use multi-model plotting with single model
            df, skipped_tasks = plot_all_tasks_emergence(
                results_dirs=[str(results_dir)],
                model_names=["Model"],
                method=args.method,
                smooth_sigma=args.smooth_sigma,
                output_dir=Path(args.plot_dir),
                min_max_accuracy=args.min_accuracy,
                tasks=args.tasks,
                **kwargs
            )
            print(f"\n✅ Plots saved to: {args.plot_dir}")
        else:
            df, skipped_tasks = analyze_all_tasks(
                results_dir,
                method=args.method,
                smooth_sigma=args.smooth_sigma,
                min_max_accuracy=args.min_accuracy,
                **kwargs
            )
        
        # Report skipped tasks
        if skipped_tasks:
            print(f"\n⚠️  Skipped {len(skipped_tasks)} tasks with max accuracy <= {args.min_accuracy}:")
            for task in skipped_tasks:
                print(f"    - {task}")
            print()

        # Handle empty results
        if df.empty:
            print("\n⚠️  No tasks found or all tasks were skipped.")
            print("Check that your results directory contains accuracy_pivot_*.csv or accuracy_pivot.csv files.")
            return 1

        # Sort by emergence point
        df = df.sort_values("emergence_tokens_B", na_position="last")

        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Saved to {args.output}")
        else:
            print(df.to_string(index=False))

        # Summary stats
        emerged = df["emergence_tokens_B"].notna().sum()
        print(f"\n{emerged}/{len(df)} tasks emerged")
        if emerged > 0:
            print(f"Median emergence: {df['emergence_tokens_B'].median():.2f} B tokens")
    
    return 0


if __name__ == "__main__":
    exit(main())

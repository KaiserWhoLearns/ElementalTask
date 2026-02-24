#!/usr/bin/env python3
"""Analyze if compositional task learning can be predicted from component tasks.

This script:
1. Identifies compositional tasks and their atomic components
2. Loads accuracy trajectories for both
3. Tests different prediction models:
   - Linear combination: acc(A∘B) = α·acc(A) + β·acc(B) + γ
   - Multiplicative: acc(A∘B) = acc(A) × acc(B)
   - Min (bottleneck): acc(A∘B) = min(acc(A), acc(B))
   - Max emergence: emergence(A∘B) = max(emergence(A), emergence(B))
4. Visualizes predictions vs actual trajectories

Usage:
    python scripts/trajectory_analysis/predict_compositional_from_components.py \
        --results_dir results/olmo2_continuous_1b_early_revised \
        --output_dir plots/compositional_prediction \
        --method all
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d

from get_emergence_point import (
    load_accuracy_data,
    load_combined_pivot,
    extract_tokens_from_checkpoint,
    smooth_trajectory,
)


@dataclass
class TaskTrajectory:
    """Accuracy trajectory for a task."""
    task_name: str
    tokens: np.ndarray  # in billions
    accuracy: np.ndarray
    
    def interpolate(self, target_tokens: np.ndarray) -> np.ndarray:
        """Interpolate accuracy at target token counts."""
        return np.interp(target_tokens, self.tokens, self.accuracy)


# ============================================================================
# Component Mapping
# ============================================================================

# Map compositional task suffixes to their atomic component tasks
# NOTE: Most simple_icl tasks use underscores (simple_icl_uppercase)
# but present_to_gerund uses a colon (simple_icl:present_to_gerund)
COMPONENT_MAP = {
    # String operations
    'upper': 'simple_icl:uppercase',
    'lower': 'simple_icl:lowercase',
    'reverse': 'token_reversal',
    
    # Grammatical
    'plural': 'simple_icl:singular_to_plural',
    'gerund': 'simple_icl:present_to_gerund',
    
    # Translation
    'translate_eng_fr': 'simple_icl:translate_eng_fr',
    'translate_fr_eng': 'simple_icl:translate_fr_eng',
    'translate_sp_eng': 'simple_icl:translate_sp_eng',
    'translate_eng_sp': 'simple_icl:translate_eng_sp',
    
    # Extraction
    'first': 'simple_icl:first_letter',
    'last': 'simple_icl:last_letter',
}


def parse_compositional_task(task_name: str) -> Optional[List[str]]:
    """Parse compositional task name into component operations.
    
    Args:
        task_name: e.g., "compositional:plural_upper" or "compositional:translate_eng_fr_reverse"
    
    Returns:
        List of operations in order, e.g., ["plural", "upper"]
        Returns None if not a compositional task
    """
    if not task_name.startswith("compositional:"):
        return None
    
    suffix = task_name.replace("compositional:", "")
    
    # Handle translation specially (it has underscores in the operation name)
    if "translate" in suffix:
        # e.g., "translate_eng_fr_upper" -> ["translate_eng_fr", "upper"]
        parts = []
        remaining = suffix
        
        # Extract translation operations first (in order of longest match)
        for trans in ['translate_eng_fr', 'translate_fr_eng', 'translate_sp_eng', 'translate_eng_sp']:
            if remaining.startswith(trans):
                parts.append(trans)
                remaining = remaining[len(trans):]
                if remaining.startswith('_'):
                    remaining = remaining[1:]
                break
        
        # Split remaining by underscore
        if remaining:
            parts.extend(remaining.split('_'))
        
        return parts if parts else None
    
    # Simple case: split by underscore
    parts = suffix.split('_')
    return parts if len(parts) >= 2 else None


def get_component_tasks(compositional_task: str) -> Optional[List[str]]:
    """Get the atomic component task names for a compositional task.
    
    Args:
        compositional_task: e.g., "compositional:plural_upper"
    
    Returns:
        List of component task names, e.g., ["simple_icl:singular_to_plural", "simple_icl:uppercase"]
        Returns None if components cannot be identified
    """
    operations = parse_compositional_task(compositional_task)
    if operations is None:
        return None
    
    components = []
    for op in operations:
        if op in COMPONENT_MAP:
            components.append(COMPONENT_MAP[op])
        else:
            # Unknown operation - might need to add to COMPONENT_MAP
            print(f"  Warning: Unknown operation '{op}' in {compositional_task}")
            return None
    
    return components


# ============================================================================
# Loading Data
# ============================================================================

def normalize_task_name(task_name: str) -> str:
    """Normalize task names so both 'compositional_X' and 'compositional:X' use colon format.
    Same for 'simple_icl_X' -> 'simple_icl:X'."""
    # Known multi-word subtask suffixes (after the task prefix)
    SIMPLE_ICL_SUBTASKS = [
        'country_to_capital', 'country_to_currency',
        'first_letter', 'last_letter',
        'present_to_gerund', 'singular_to_plural',
        'translate_eng_fr', 'translate_fr_eng',
        'translate_eng_sp', 'translate_sp_eng',
        'uppercase', 'lowercase',
    ]
    
    # Normalize simple_icl tasks: simple_icl_uppercase -> simple_icl:uppercase
    if task_name.startswith("simple_icl_") or task_name.startswith("simple_icl:"):
        for subtask in SIMPLE_ICL_SUBTASKS:
            if task_name == f"simple_icl_{subtask}" or task_name == f"simple_icl:{subtask}":
                return f"simple_icl:{subtask}"
    
    # Normalize compositional tasks: compositional_gerund_lower -> compositional:gerund_lower
    if task_name.startswith("compositional_"):
        suffix = task_name[len("compositional_"):]
        return f"compositional:{suffix}"
    
    return task_name


def load_all_trajectories(results_dir: Path) -> Dict[str, TaskTrajectory]:
    """Load accuracy trajectories for all tasks."""
    trajectories = {}

    # First try per-task pivot files
    pivot_files = list(results_dir.glob("accuracy_pivot_*.csv"))

    if pivot_files:
        for pivot_file in pivot_files:
            try:
                tokens, accuracy, task_name = load_accuracy_data(pivot_file)
                task_name = normalize_task_name(task_name)

                if len(tokens) == 0:
                    continue

                trajectories[task_name] = TaskTrajectory(
                    task_name=task_name,
                    tokens=tokens,
                    accuracy=accuracy
                )
            except Exception as e:
                print(f"Warning: Failed to load {pivot_file.name}: {e}")
    else:
        # Try combined pivot file
        combined_file = results_dir / "accuracy_pivot.csv"
        if combined_file.exists():
            print(f"  Using combined pivot file: {combined_file}")
            task_data = load_combined_pivot(combined_file)

            for task_name, (tokens, accuracy) in task_data.items():
                task_name = normalize_task_name(task_name)

                if len(tokens) == 0:
                    continue

                trajectories[task_name] = TaskTrajectory(
                    task_name=task_name,
                    tokens=tokens,
                    accuracy=accuracy
                )

    return trajectories


def discover_compositional_tasks(
    trajectories: Dict[str, TaskTrajectory]
) -> Dict[str, List[str]]:
    """Find all compositional tasks and their components.
    
    Returns:
        Dict mapping compositional_task -> [component1, component2, ...]
    """
    compositional = {}
    
    for task_name in trajectories.keys():
        if not task_name.startswith("compositional"):
            continue
        
        components = get_component_tasks(task_name)
        if components is None:
            continue
        
        # Check if all components exist in our data
        if all(comp in trajectories for comp in components):
            compositional[task_name] = components
        else:
            missing = [c for c in components if c not in trajectories]
            print(f"  ⚠️  Skipping {task_name}: missing components {missing}")
    
    return compositional


# ============================================================================
# Prediction Models
# ============================================================================

def predict_multiplicative(
    component_accs: List[np.ndarray]
) -> np.ndarray:
    """Predict compositional accuracy as product of components.
    
    acc(A∘B) = acc(A) × acc(B)
    
    Assumes independence: success requires succeeding at both steps.
    """
    result = np.ones_like(component_accs[0])
    for acc in component_accs:
        result *= acc
    return result


def predict_min(
    component_accs: List[np.ndarray]
) -> np.ndarray:
    """Predict compositional accuracy as minimum of components.
    
    acc(A∘B) = min(acc(A), acc(B))
    
    Bottleneck model: limited by hardest component.
    """
    return np.minimum.reduce(component_accs)


def predict_mean(
    component_accs: List[np.ndarray]
) -> np.ndarray:
    """Predict compositional accuracy as mean of components.
    
    acc(A∘B) = (acc(A) + acc(B)) / 2
    """
    return np.mean(component_accs, axis=0)


def predict_harmonic_mean(
    component_accs: List[np.ndarray]
) -> np.ndarray:
    """Predict compositional accuracy as harmonic mean of components.
    
    Harmonic mean gives more weight to lower values, similar to bottleneck.
    """
    # Avoid division by zero
    safe_accs = [np.maximum(acc, 1e-6) for acc in component_accs]
    return len(safe_accs) / np.sum([1.0 / acc for acc in safe_accs], axis=0)


def predict_geometric_mean(
    component_accs: List[np.ndarray]
) -> np.ndarray:
    """Predict compositional accuracy as geometric mean of components.
    
    Similar to multiplicative but with a root: (acc_A × acc_B)^(1/n)
    """
    result = np.ones_like(component_accs[0])
    for acc in component_accs:
        result *= acc
    return np.power(result, 1.0 / len(component_accs))


def fit_linear_combination(
    component_accs: List[np.ndarray],
    target_acc: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Fit linear combination: acc(A∘B) = α·acc(A) + β·acc(B) + γ.
    
    Returns:
        coefficients: [α, β, ..., γ]
        r2_score: R² of fit
        prediction: Predicted accuracy
    """
    # Build design matrix
    X = np.column_stack(component_accs + [np.ones_like(target_acc)])
    
    # Solve least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(X, target_acc, rcond=None)
    
    # Compute R²
    prediction = X @ coeffs
    ss_res = np.sum((target_acc - prediction) ** 2)
    ss_tot = np.sum((target_acc - target_acc.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return coeffs, r2, prediction


def evaluate_prediction(
    predicted: np.ndarray,
    actual: np.ndarray
) -> Dict[str, float]:
    """Evaluate prediction quality."""
    # Compute metrics
    mse = np.mean((predicted - actual) ** 2)
    mae = np.mean(np.abs(predicted - actual))
    rmse = np.sqrt(mse)
    
    # R² (correlation coefficient squared)
    if len(actual) > 1 and actual.std() > 0:
        r, _ = pearsonr(predicted, actual)
        r2 = r ** 2
    else:
        r2 = 0.0
    
    # Max error
    max_error = np.max(np.abs(predicted - actual))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error,
        'r2': r2,
    }


# ============================================================================
# Emergence Analysis
# ============================================================================

def predict_emergence_max(
    component_trajectories: List[TaskTrajectory],
    threshold: float = 0.5
) -> Optional[float]:
    """Predict emergence as max of component emergences.
    
    Hypothesis: compositional task emerges when last component has emerged.
    """
    emergences = []
    for traj in component_trajectories:
        mask = traj.accuracy >= threshold
        if mask.any():
            emergences.append(float(traj.tokens[mask.argmax()]))
        else:
            return None  # Component never emerges
    
    return max(emergences) if emergences else None


def predict_emergence_sum(
    component_trajectories: List[TaskTrajectory],
    threshold: float = 0.5
) -> Optional[float]:
    """Predict emergence as sum of component emergences.
    
    Hypothesis: takes time for both to emerge plus integration time.
    """
    emergences = []
    for traj in component_trajectories:
        mask = traj.accuracy >= threshold
        if mask.any():
            emergences.append(float(traj.tokens[mask.argmax()]))
        else:
            return None
    
    return sum(emergences) if emergences else None


# ============================================================================
# Visualization
# ============================================================================

def plot_prediction_comparison(
    comp_task_name: str,
    actual_traj: TaskTrajectory,
    component_trajs: List[TaskTrajectory],
    predictions: Dict[str, np.ndarray],
    metrics: Dict[str, Dict[str, float]],
    output_path: Path,
    smooth_sigma: float = 1.0,
):
    """Plot actual vs predicted trajectories with component accuracy panels.
    
    Layout:
        Left (large): Compositional task actual vs predicted trajectories
        Right (stacked): Individual component task accuracy trajectories
    
    Args:
        comp_task_name: Name of the compositional task
        actual_traj: Actual trajectory for the compositional task
        component_trajs: Trajectories for component tasks
        predictions: Dict mapping method -> predicted accuracy
        metrics: Dict mapping method -> evaluation metrics
        output_path: Where to save the plot
        smooth_sigma: Smoothing sigma used (for display in title)
    """
    n_components = len(component_trajs)
    
    # Create layout: large left panel + stacked right mini-panels
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(n_components, 2, width_ratios=[2, 1], hspace=0.4, wspace=0.3)
    
    # Left panel spans all rows
    ax_main = fig.add_subplot(gs[:, 0])
    
    # Right mini-panels: one per component
    ax_components = [fig.add_subplot(gs[i, 1]) for i in range(n_components)]
    
    tokens = actual_traj.tokens
    
    # Component colors (consistent between main and mini-panels)
    comp_colors = plt.cm.Set2(np.linspace(0, 0.6, n_components))
    
    # ===== Left plot: Compositional trajectories =====
    # Plot components (faint) on main panel for reference
    for i, comp_traj in enumerate(component_trajs):
        comp_interp = comp_traj.interpolate(tokens)
        comp_smooth = smooth_trajectory(comp_interp, sigma=smooth_sigma)
        comp_label = comp_traj.task_name.replace('simple_icl_', '').replace('simple_icl:', '')
        
        ax_main.plot(tokens, comp_smooth, '--', alpha=0.35, linewidth=1.5,
                     color=comp_colors[i], label=f'{comp_label}')
    
    # Plot actual compositional (both raw and smoothed)
    ax_main.scatter(tokens, actual_traj.accuracy, s=30, alpha=0.3, color='black', marker='o', zorder=9)
    actual_smooth = smooth_trajectory(actual_traj.accuracy, sigma=smooth_sigma)
    ax_main.plot(tokens, actual_smooth, 'k-', linewidth=3, alpha=0.8,
            label='Actual (smoothed)', marker='o', markersize=6, markevery=max(1, len(tokens)//10), zorder=10)
    
    # Plot predictions
    pred_colors = {
        'multiplicative': 'blue',
        'min': 'red',
        'mean': 'green',
        'linear': 'purple',
        'harmonic_mean': 'orange',
        'geometric_mean': 'brown'
    }
    
    # Sort by R² to show best prediction most prominently
    sorted_methods = sorted(predictions.keys(), 
                           key=lambda m: metrics[m]['r2'], 
                           reverse=True)
    
    for i, method in enumerate(sorted_methods):
        pred_acc = predictions[method]
        color = pred_colors.get(method, 'gray')
        r2 = metrics[method]['r2']
        
        ax_main.plot(tokens, pred_acc, '--', linewidth=2, color=color,
                label=f'{method} (R²={r2:.3f})', alpha=0.7, zorder=9-i)
    
    ax_main.set_xlabel('Training Tokens (B)', fontsize=12)
    ax_main.set_ylabel('Accuracy', fontsize=12)
    title = f'{comp_task_name}'
    if smooth_sigma > 0:
        title += f'  (σ={smooth_sigma:.1f})'
    ax_main.set_title(title, fontsize=14)
    ax_main.legend(fontsize=8, loc='best', ncol=2)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylim(-0.05, 1.05)
    
    # ===== Right mini-panels: Component accuracies =====
    for i, (comp_traj, ax_comp) in enumerate(zip(component_trajs, ax_components)):
        comp_interp = comp_traj.interpolate(tokens)
        comp_smooth = smooth_trajectory(comp_interp, sigma=smooth_sigma)
        comp_label = comp_traj.task_name.replace('simple_icl_', '').replace('simple_icl:', '')
        
        # Raw dots
        ax_comp.scatter(tokens, comp_interp, s=15, alpha=0.25, color=comp_colors[i])
        # Smoothed line
        ax_comp.plot(tokens, comp_smooth, '-', linewidth=2, color=comp_colors[i], alpha=0.9)
        
        # Final accuracy annotation
        final_acc = comp_smooth[-1] if len(comp_smooth) > 0 else 0
        ax_comp.annotate(f'{final_acc:.2f}', xy=(tokens[-1], final_acc),
                        fontsize=9, fontweight='bold', color=comp_colors[i],
                        ha='left', va='center', xytext=(5, 0), textcoords='offset points')
        
        ax_comp.set_title(comp_label, fontsize=11, color=comp_colors[i], fontweight='bold')
        ax_comp.set_ylim(-0.05, 1.05)
        ax_comp.grid(True, alpha=0.3)
        ax_comp.tick_params(labelsize=9)
        
        # Only show x-label on bottom mini-panel
        if i == n_components - 1:
            ax_comp.set_xlabel('Tokens (B)', fontsize=10)
        else:
            ax_comp.set_xticklabels([])
        
        ax_comp.set_ylabel('Acc', fontsize=10)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_emergence_comparison(
    results_df: pd.DataFrame,
    output_path: Path,
):
    """Plot predicted vs actual emergence points."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Filter rows with valid actual emergence
    df = results_df[results_df['actual_emergence'].notna()].copy()
    
    if len(df) == 0:
        print("  No valid emergence data to plot")
        return
    
    # ===== Left: Predicted vs Actual =====
    ax = axes[0]
    
    pred_methods = [col for col in df.columns if col.endswith('_pred_emergence') and df[col].notna().any()]
    
    colors = {'max': 'blue', 'sum': 'red'}
    
    for method_col in pred_methods:
        method_name = method_col.replace('_pred_emergence', '')
        valid = df[method_col].notna()
        
        if valid.sum() == 0:
            continue
        
        color = colors.get(method_name, 'gray')
        ax.scatter(df.loc[valid, method_col], df.loc[valid, 'actual_emergence'],
                  alpha=0.6, s=80, label=method_name, color=color)
    
    # Diagonal line (perfect prediction)
    lims = [
        df[['actual_emergence'] + pred_methods].min().min(),
        df[['actual_emergence'] + pred_methods].max().max(),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='Perfect')
    
    ax.set_xlabel('Predicted Emergence (B tokens)', fontsize=12)
    ax.set_ylabel('Actual Emergence (B tokens)', fontsize=12)
    ax.set_title('Emergence Prediction Quality', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ===== Right: Prediction Error Distribution =====
    ax = axes[1]
    
    for method_col in pred_methods:
        method_name = method_col.replace('_pred_emergence', '')
        valid = df[method_col].notna()
        
        if valid.sum() == 0:
            continue
        
        errors = df.loc[valid, method_col] - df.loc[valid, 'actual_emergence']
        color = colors.get(method_name, 'gray')
        ax.hist(errors, bins=15, alpha=0.5, label=method_name, color=color)
    
    ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Prediction Error (B tokens)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Emergence Prediction Error Distribution', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_method_comparison_summary(
    results_df: pd.DataFrame,
    output_path: Path,
):
    """Create summary plot comparing all prediction methods."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    methods = ['multiplicative', 'min', 'mean', 'harmonic_mean', 'geometric_mean']
    method_cols = {m: f'{m}_r2' for m in methods if f'{m}_r2' in results_df.columns}
    
    if not method_cols:
        print("No prediction metrics found in results")
        return
    
    # ===== Top Left: R² by method =====
    ax = axes[0, 0]
    r2_data = [results_df[col].dropna() for col in method_cols.values()]
    ax.boxplot(r2_data, labels=list(method_cols.keys()))
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('Prediction Quality by Method', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # ===== Top Right: MAE by method =====
    ax = axes[0, 1]
    mae_cols = {m: f'{m}_mae' for m in methods if f'{m}_mae' in results_df.columns}
    mae_data = [results_df[col].dropna() for col in mae_cols.values()]
    ax.boxplot(mae_data, labels=list(mae_cols.keys()))
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('Prediction Error by Method', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # ===== Bottom Left: Best method per task =====
    ax = axes[1, 0]
    best_methods = []
    for _, row in results_df.iterrows():
        r2_values = {m: row[f'{m}_r2'] for m in methods if f'{m}_r2' in results_df.columns and pd.notna(row[f'{m}_r2'])}
        if r2_values:
            best_methods.append(max(r2_values, key=r2_values.get))
    
    if best_methods:
        method_counts = pd.Series(best_methods).value_counts()
        ax.bar(range(len(method_counts)), method_counts.values)
        ax.set_xticks(range(len(method_counts)))
        ax.set_xticklabels(method_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Best Method (highest R²) per Task', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
    
    # ===== Bottom Right: R² vs number of components =====
    ax = axes[1, 1]
    if 'n_components' in results_df.columns:
        for method, col in method_cols.items():
            ax.scatter(results_df['n_components'], results_df[col], 
                      alpha=0.6, s=60, label=method)
        ax.set_xlabel('Number of Components', fontsize=12)
        ax.set_ylabel('R²', fontsize=12)
        ax.set_title('Prediction Quality vs Task Complexity', fontsize=14)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Analysis
# ============================================================================

def analyze_compositional_predictions(
    results_dir: Path,
    output_dir: Path,
    methods: List[str] = ['all'],
    emergence_threshold: float = 0.5,
    smooth_sigma: float = 1.0,
):
    """Main analysis function.
    
    Args:
        results_dir: Directory containing accuracy_pivot_*.csv files
        output_dir: Output directory for plots and results
        methods: List of prediction methods to test
        emergence_threshold: Accuracy threshold for emergence detection
        smooth_sigma: Gaussian smoothing sigma (0 = no smoothing)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("COMPOSITIONAL TASK PREDICTION ANALYSIS")
    print("=" * 70)
    print(f"Results dir: {results_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Smoothing sigma: {smooth_sigma}")
    print()
    
    # Load all trajectories
    print("Loading trajectories...")
    trajectories = load_all_trajectories(results_dir)
    print(f"  Loaded {len(trajectories)} tasks")
    
    # Discover compositional tasks
    print("\nDiscovering compositional tasks...")
    compositional_tasks = discover_compositional_tasks(trajectories)
    print(f"  Found {len(compositional_tasks)} compositional tasks with all components")
    
    if len(compositional_tasks) == 0:
        print("No compositional tasks found!")
        return
    
    # Print task→component mapping
    print("\nCompositional task structure:")
    for comp_task, components in sorted(compositional_tasks.items()):
        print(f"  {comp_task}")
        for i, comp in enumerate(components, 1):
            print(f"    {i}. {comp}")
    
    # Analyze each compositional task
    results = []
    
    print("\nAnalyzing predictions...")
    for comp_task, component_names in sorted(compositional_tasks.items()):
        print(f"\n{comp_task}")
        
        actual_traj = trajectories[comp_task]
        component_trajs = [trajectories[name] for name in component_names]
        
        # Get common token grid (interpolate to finest resolution)
        all_tokens = np.unique(np.concatenate([t.tokens for t in [actual_traj] + component_trajs]))
        
        # Interpolate components to common grid
        component_accs_raw = [traj.interpolate(all_tokens) for traj in component_trajs]
        actual_acc_raw = actual_traj.interpolate(all_tokens)
        
        # Apply smoothing
        component_accs = [smooth_trajectory(acc, sigma=smooth_sigma) for acc in component_accs_raw]
        actual_acc = smooth_trajectory(actual_acc_raw, sigma=smooth_sigma)
        
        # Make predictions
        predictions = {}
        
        if 'all' in methods or 'multiplicative' in methods:
            predictions['multiplicative'] = predict_multiplicative(component_accs)
        
        if 'all' in methods or 'min' in methods:
            predictions['min'] = predict_min(component_accs)
        
        if 'all' in methods or 'mean' in methods:
            predictions['mean'] = predict_mean(component_accs)
        
        if 'all' in methods or 'harmonic_mean' in methods:
            predictions['harmonic_mean'] = predict_harmonic_mean(component_accs)
        
        if 'all' in methods or 'geometric_mean' in methods:
            predictions['geometric_mean'] = predict_geometric_mean(component_accs)
        
        # Linear is commented out — per-task fitted coefficients make it unfairly flexible
        # if 'all' in methods or 'linear' in methods:
        #     coeffs, r2, pred = fit_linear_combination(component_accs, actual_acc)
        #     predictions['linear'] = pred
        #     linear_coeffs = coeffs
        # else:
        linear_coeffs = None
        
        # Evaluate predictions
        metrics = {}
        print("  Prediction quality:")
        for method, pred_acc in predictions.items():
            metrics[method] = evaluate_prediction(pred_acc, actual_acc)
            m = metrics[method]
            print(f"    {method:15s}: R²={m['r2']:.3f}, MAE={m['mae']:.3f}, RMSE={m['rmse']:.3f}")
        
        # Print linear coefficients
        if linear_coeffs is not None:
            print(f"  Linear model: acc = ", end="")
            for i, (comp_name, coeff) in enumerate(zip(component_names, linear_coeffs[:-1])):
                comp_short = comp_name.split(':')[-1].split('_')[-1]  # Get last part of name
                sign = "+" if coeff >= 0 else ""
                print(f"{sign}{coeff:.3f}·{comp_short} ", end="")
            # Intercept
            sign = "+" if linear_coeffs[-1] >= 0 else ""
            print(f"{sign}{linear_coeffs[-1]:.3f}")
        
        # Find best method
        best_method = max(metrics.keys(), key=lambda m: metrics[m]['r2'])
        print(f"  → Best: {best_method} (R²={metrics[best_method]['r2']:.3f})")
        
        # Analyze emergence
        actual_emergence = None
        mask = actual_traj.accuracy >= emergence_threshold
        if mask.any():
            actual_emergence = float(actual_traj.tokens[mask.argmax()])
        
        pred_emergence_max = predict_emergence_max(component_trajs, emergence_threshold)
        pred_emergence_sum = predict_emergence_sum(component_trajs, emergence_threshold)
        
        if actual_emergence:
            if pred_emergence_max:
                error_max = pred_emergence_max - actual_emergence
                print(f"  Emergence (max): actual={actual_emergence:.0f}B, predicted={pred_emergence_max:.0f}B, error={error_max:+.0f}B")
            if pred_emergence_sum:
                error_sum = pred_emergence_sum - actual_emergence
                print(f"  Emergence (sum): actual={actual_emergence:.0f}B, predicted={pred_emergence_sum:.0f}B, error={error_sum:+.0f}B")
        
        # Save results
        result_row = {
            'task': comp_task,
            'n_components': len(component_names),
            'components': ' + '.join([c.split(':')[-1] for c in component_names]),
            'actual_emergence': actual_emergence,
            'max_pred_emergence': pred_emergence_max,
            'sum_pred_emergence': pred_emergence_sum,
            'best_method': best_method,
        }
        
        for method, pred_acc in predictions.items():
            m = metrics[method]
            result_row[f'{method}_r2'] = m['r2']
            result_row[f'{method}_mae'] = m['mae']
            result_row[f'{method}_rmse'] = m['rmse']
            result_row[f'{method}_max_error'] = m['max_error']
        
        results.append(result_row)
        
        # Plot
        plot_path = output_dir / f"{comp_task.replace(':', '_')}_prediction.png"
        plot_prediction_comparison(
            comp_task, actual_traj, component_trajs,
            predictions, metrics, plot_path, smooth_sigma=smooth_sigma
        )
    
    # Create summary dataframe
    results_df = pd.DataFrame(results)
    
    # Save results
    csv_path = output_dir / "prediction_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved results to: {csv_path}")
    
    # Plot emergence comparison
    emergence_plot_path = output_dir / "emergence_predictions.png"
    plot_emergence_comparison(results_df, emergence_plot_path)
    print(f"✅ Saved emergence plot to: {emergence_plot_path}")
    
    # Plot method comparison summary
    summary_plot_path = output_dir / "method_comparison_summary.png"
    plot_method_comparison_summary(results_df, summary_plot_path)
    print(f"✅ Saved method comparison to: {summary_plot_path}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for method in ['multiplicative', 'min', 'mean', 'harmonic_mean', 'geometric_mean']:
        if f'{method}_mae' in results_df.columns:
            r2_col = f'{method}_r2'
            mae_col = f'{method}_mae'
            print(f"\n{method.upper()} method:")
            print(f"  Mean R²:  {results_df[r2_col].mean():.3f}")
            print(f"  Mean MAE: {results_df[mae_col].mean():.3f}")
            print(f"  Best task (highest R²): {results_df.loc[results_df[r2_col].idxmax(), 'task']} (R²={results_df[r2_col].max():.3f})")
            print(f"  Worst task (lowest R²): {results_df.loc[results_df[r2_col].idxmin(), 'task']} (R²={results_df[r2_col].min():.3f})")
    
    # Best method summary
    if 'best_method' in results_df.columns:
        print(f"\nBEST METHOD DISTRIBUTION:")
        method_counts = results_df['best_method'].value_counts()
        for method, count in method_counts.items():
            pct = count / len(results_df) * 100
            print(f"  {method:15s}: {count:2d} tasks ({pct:5.1f}%)")
    
    # Emergence prediction summary
    for pred_col, pred_name in [('max_pred_emergence', 'MAX'), ('sum_pred_emergence', 'SUM')]:
        if pred_col in results_df.columns:
            valid = results_df[['actual_emergence', pred_col]].notna().all(axis=1)
            if valid.sum() > 0:
                errors = results_df.loc[valid, pred_col] - results_df.loc[valid, 'actual_emergence']
                print(f"\nEMERGENCE PREDICTION ({pred_name} method):")
                print(f"  Mean error: {errors.mean():+.1f}B tokens")
                print(f"  MAE: {errors.abs().mean():.1f}B tokens")
                print(f"  Predicted {valid.sum()}/{len(results_df)} emergences")
    
    print(f"\n✅ All plots saved to: {output_dir}")

    # Write human-readable summary.md
    write_summary_md(results_df, compositional_tasks, output_dir, results_dir, smooth_sigma)


def write_summary_md(
    results_df: pd.DataFrame,
    compositional_tasks: Dict[str, List[str]],
    output_dir: Path,
    results_dir: Path,
    smooth_sigma: float,
):
    """Write a human-readable summary.md of the prediction results."""
    lines = []
    lines.append("# Compositional Task Prediction — Summary\n")
    lines.append(f"> `results_dir`: `{results_dir}`  \n")
    lines.append(f"> `output_dir`: `{output_dir}`  \n")
    lines.append(f"> smoothing σ = {smooth_sigma}  \n")
    lines.append("\n---\n")

    methods = ['multiplicative', 'min', 'mean', 'harmonic_mean', 'geometric_mean']
    available = [m for m in methods if f'{m}_r2' in results_df.columns]

    # ── Overall method comparison ──────────────────────────────────────────
    lines.append("## Method Comparison (across all compositional tasks)\n")
    header = f"{'Method':<18} {'Mean R²':>8} {'Mean MAE':>9} {'Best task':>35} {'Worst task':>35}"
    lines.append("```\n")
    lines.append(header + "\n")
    lines.append("-" * len(header) + "\n")
    for m in available:
        r2s = results_df[f'{m}_r2']
        maes = results_df[f'{m}_mae']
        best_task  = results_df.loc[r2s.idxmax(), 'task']
        worst_task = results_df.loc[r2s.idxmin(), 'task']
        lines.append(
            f"{m:<18} {r2s.mean():>8.3f} {maes.mean():>9.3f}"
            f" {best_task:>35} {worst_task:>35}\n"
        )
    lines.append("```\n")

    # Best method distribution
    if 'best_method' in results_df.columns:
        lines.append("\n### Winning method per task\n")
        lines.append("```\n")
        for method, count in results_df['best_method'].value_counts().items():
            pct = count / len(results_df) * 100
            lines.append(f"  {method:<18}: {count:2d} tasks ({pct:5.1f}%)\n")
        lines.append("```\n")

    # ── Per-task breakdown ─────────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Per-Task Results\n")

    col_header = f"{'Task':<42} {'Components':<40} {'Best Method':<18}"
    for m in available:
        col_header += f" {m[:4]:>7}"
    col_header += f" {'Best R²':>8}"
    lines.append("```\n")
    lines.append(col_header + "\n")
    lines.append("-" * len(col_header) + "\n")

    for _, row in results_df.sort_values(f"{available[0]}_r2" if available else 'task', ascending=False).iterrows():
        best_r2 = max(row[f'{m}_r2'] for m in available)
        col_line = f"{row['task']:<42} {row['components']:<40} {row['best_method']:<18}"
        for m in available:
            col_line += f" {row[f'{m}_r2']:>7.3f}"
        col_line += f" {best_r2:>8.3f}"
        lines.append(col_line + "\n")
    lines.append("```\n")

    # ── Emergence predictions ──────────────────────────────────────────────
    for pred_col, pred_name in [('max_pred_emergence', 'MAX'), ('sum_pred_emergence', 'SUM')]:
        if pred_col not in results_df.columns:
            continue
        valid = results_df[['actual_emergence', pred_col]].notna().all(axis=1)
        if valid.sum() == 0:
            continue
        errors = results_df.loc[valid, pred_col] - results_df.loc[valid, 'actual_emergence']
        lines.append(f"\n## Emergence Prediction ({pred_name} method)\n")
        lines.append("```\n")
        lines.append(f"  Predicted {valid.sum()}/{len(results_df)} emergences\n")
        lines.append(f"  Mean error : {errors.mean():+.1f}B tokens\n")
        lines.append(f"  MAE        : {errors.abs().mean():.1f}B tokens\n")
        lines.append(f"  Max overest: {errors.max():+.1f}B tokens\n")
        lines.append(f"  Max underest: {errors.min():+.1f}B tokens\n")
        lines.append("```\n")
        lines.append("\n| Task | Actual | Predicted | Error |\n")
        lines.append("|------|--------|-----------|-------|\n")
        for _, row in results_df[valid].iterrows():
            err = row[pred_col] - row['actual_emergence']
            lines.append(
                f"| {row['task']} | {row['actual_emergence']:.0f}B"
                f" | {row[pred_col]:.0f}B | {err:+.0f}B |\n"
            )

    # ── Task structure ─────────────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Task → Component Mapping\n")
    lines.append("```\n")
    for comp_task, components in sorted(compositional_tasks.items()):
        lines.append(f"  {comp_task}\n")
        for i, c in enumerate(components, 1):
            lines.append(f"    {i}. {c}\n")
    lines.append("```\n")

    md_path = output_dir / "summary.md"
    md_path.write_text("".join(lines))
    print(f"✅ Saved summary to: {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict compositional task learning from component tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("-d", "--results_dir", type=str, required=True,
                        help="Directory containing accuracy_pivot_*.csv files")
    parser.add_argument("-o", "--output_dir", type=str, default="plots/compositional_prediction",
                        help="Output directory for plots and results")
    parser.add_argument("-m", "--method", nargs="+", default=["all"],
                        choices=["all", "multiplicative", "min", "mean", "harmonic_mean", "geometric_mean", "linear"],
                        help="Prediction methods to use")
    parser.add_argument("-t", "--threshold", type=float, default=0.5,
                        help="Emergence threshold (default: 0.5)")
    parser.add_argument("-s", "--smooth-sigma", type=float, default=1.0,
                        help="Gaussian smoothing sigma (0 = no smoothing, default: 1.0)")
    
    args = parser.parse_args()
    
    analyze_compositional_predictions(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir),
        methods=args.method,
        emergence_threshold=args.threshold,
        smooth_sigma=args.smooth_sigma,
    )


if __name__ == "__main__":
    main()

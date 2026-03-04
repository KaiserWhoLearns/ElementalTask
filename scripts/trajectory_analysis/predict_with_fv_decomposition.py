#!/usr/bin/env python3
"""Predict compositional task performance using function vector decomposition.

This script extends the behavioral baseline (predict_compositional_from_components.py)
by using FV-derived weights from the skill basis to improve predictions.

The key idea:
  1. At the mature checkpoint, extract FVs for all tasks and build a skill basis.
     (Already done by analyze_real_tasks.py)
  2. For each compositional task, compute its FV's similarity to each elemental FV.
     These similarities become weights w_i — how much each elemental skill contributes.
  3. Use these weights + elemental accuracy curves a_i(t) to predict compositional
     accuracy across all training checkpoints.

Three levels of prediction:
  - BEHAVIORAL BASELINE: Unweighted combination of known components (from COMPONENT_MAP)
  - FV-WEIGHTED: Use FV similarities as weights on known components
  - FV-DISCOVERED: Discover which elemental tasks matter purely from FV similarities
    (no human-specified decomposition needed)

Usage:
    python scripts/trajectory_analysis/predict_with_fv_decomposition.py \\
        --fv_dir function_vecs/results/olmo2_1b_correct_only \\
        --results_dir results/olmo2_continuous_1b_early_revised \\
        --output_dir plots/fv_prediction_1b
"""

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.stats import pearsonr

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from function_vecs.extract_function_vecs import (
    load_skill_basis,
    load_function_vec,
    SkillBasis,
    TaskFunctionVec,
)
from scripts.trajectory_analysis.predict_compositional_from_components import (
    TaskTrajectory,
    COMPONENT_MAP,
    load_all_trajectories,
    normalize_task_name,
    parse_compositional_task,
    get_component_tasks,
    predict_multiplicative,
    predict_min,
    predict_mean,
    evaluate_prediction,
    smooth_trajectory,
)


# ============================================================================
# FV Loading & Weight Computation
# ============================================================================

@dataclass
class FVDecomposition:
    """Decomposition of a compositional FV into elemental FV components."""
    comp_task: str
    # Weights from FV similarity (cosine sim to each elemental FV)
    elemental_weights: Dict[str, float]  # elemental_task_name -> weight
    # Known components from COMPONENT_MAP
    known_components: Optional[List[str]]
    # Reconstruction quality of the compositional FV from elemental FVs
    reconstruction_cosine: float = 0.0


def load_all_fvs(
    basis: SkillBasis,
    test_vec_dir: Path,
) -> Dict[str, np.ndarray]:
    """Load all FVs: reconstruct training FVs from basis, load test FVs from disk.

    Returns:
        Dict mapping task_name -> FV (unit-norm np.ndarray)
    """
    fvs = {}

    # Reconstruct training FVs from SVD: F = U @ diag(S) @ Vt
    F = basis.U @ np.diag(basis.S) @ basis.Vt  # (d_model, n_train)
    for i, name in enumerate(basis.task_names):
        fv = F[:, i]
        norm = np.linalg.norm(fv)
        if norm > 1e-10:
            fv = fv / norm
        fvs[name] = fv

    # Load test vectors from disk
    if test_vec_dir.exists():
        for npz_file in test_vec_dir.glob("*.npz"):
            try:
                vec = load_function_vec(str(npz_file))
                fvs[vec.task_name] = vec.function_vec
            except Exception as e:
                print(f"  Warning: Failed to load {npz_file.name}: {e}")

    return fvs


def compute_fv_weights(
    comp_fv: np.ndarray,
    elemental_fvs: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compute FV-derived weights: cosine similarity of compositional FV to each elemental FV.

    Since all FVs are L2-normalized, cosine similarity = dot product.

    Args:
        comp_fv: Compositional task's FV (unit norm)
        elemental_fvs: Dict of elemental task name -> FV (unit norm)

    Returns:
        Dict mapping elemental task name -> cosine similarity weight
    """
    weights = {}
    for name, efv in elemental_fvs.items():
        weights[name] = float(np.dot(comp_fv, efv))
    return weights


def compute_least_squares_weights(
    comp_fv: np.ndarray,
    elemental_fvs: Dict[str, np.ndarray],
) -> Tuple[Dict[str, float], float]:
    """Decompose compositional FV as linear combination of elemental FVs via least squares.

    v_comp ≈ Σ_i w_i * v_i

    Args:
        comp_fv: Compositional task's FV
        elemental_fvs: Dict of elemental task name -> FV

    Returns:
        (weights dict, reconstruction cosine similarity)
    """
    names = list(elemental_fvs.keys())
    if not names:
        return {}, 0.0

    # Build matrix: each column is an elemental FV
    E = np.column_stack([elemental_fvs[n] for n in names])  # (d_model, n_elemental)

    # Solve: comp_fv ≈ E @ w
    w, _, _, _ = np.linalg.lstsq(E, comp_fv, rcond=None)

    # Reconstruction quality
    recon = E @ w
    recon_norm = np.linalg.norm(recon)
    if recon_norm > 1e-10:
        cos_sim = float(np.dot(comp_fv, recon) / (np.linalg.norm(comp_fv) * recon_norm))
    else:
        cos_sim = 0.0

    weights = {name: float(w[i]) for i, name in enumerate(names)}
    return weights, cos_sim


def get_elemental_task_names(all_fvs: Dict[str, np.ndarray]) -> List[str]:
    """Get names of elemental (non-compositional) tasks that have FVs."""
    elemental = []
    for name in all_fvs:
        if not name.startswith("compositional"):
            elemental.append(name)
    return sorted(elemental)


def decompose_compositional_task(
    comp_task: str,
    all_fvs: Dict[str, np.ndarray],
    elemental_names: List[str],
    weight_method: str = "cosine",
) -> Optional[FVDecomposition]:
    """Decompose a compositional task into elemental components using FVs.

    Args:
        comp_task: Compositional task name
        all_fvs: All FVs (both elemental and compositional)
        elemental_names: List of elemental task names
        weight_method: "cosine" for dot-product weights, "lstsq" for least-squares

    Returns:
        FVDecomposition or None if the compositional FV is not available
    """
    if comp_task not in all_fvs:
        return None

    comp_fv = all_fvs[comp_task]
    elemental_fvs = {n: all_fvs[n] for n in elemental_names if n in all_fvs}

    # Compute weights
    if weight_method == "lstsq":
        weights, recon_cos = compute_least_squares_weights(comp_fv, elemental_fvs)
    else:  # cosine
        weights = compute_fv_weights(comp_fv, elemental_fvs)
        # Compute reconstruction quality using top components
        recon_cos = 0.0

    # Get known components from COMPONENT_MAP
    known_components = get_component_tasks(comp_task)

    return FVDecomposition(
        comp_task=comp_task,
        elemental_weights=weights,
        known_components=known_components,
        reconstruction_cosine=recon_cos,
    )


# ============================================================================
# FV-Weighted Prediction Models
# ============================================================================

def predict_fv_weighted_product(
    component_accs: List[np.ndarray],
    weights: List[float],
) -> np.ndarray:
    """Predict using FV-weighted product: Π_i a_i(t)^|w_i|.

    Higher weight = stronger dependency on that component.
    Weights are normalized so they sum to 1.
    """
    abs_weights = np.array([abs(w) for w in weights])
    total = abs_weights.sum()
    if total < 1e-10:
        return predict_mean(component_accs)
    norm_weights = abs_weights / total

    result = np.ones_like(component_accs[0])
    for acc, w in zip(component_accs, norm_weights):
        # Clamp to avoid log(0)
        safe_acc = np.maximum(acc, 1e-6)
        result *= np.power(safe_acc, w)
    return result


def predict_fv_weighted_min(
    component_accs: List[np.ndarray],
    weights: List[float],
) -> np.ndarray:
    """Predict using FV-weighted min: min_i (a_i(t) / need_i).

    Components with higher weights are harder to satisfy — the model
    needs more of that skill, so it's a tighter bottleneck.
    The prediction is rescaled so it stays in [0, 1].
    """
    abs_weights = np.array([abs(w) for w in weights])
    max_w = abs_weights.max()
    if max_w < 1e-10:
        return predict_min(component_accs)

    # Scale weights to [0, 1] range
    scaled = abs_weights / max_w

    # Weighted min: component with high weight and low accuracy is the bottleneck
    adjusted = []
    for acc, w in zip(component_accs, scaled):
        # Blend between "always 1" (w=0, doesn't matter) and raw accuracy (w=1)
        adjusted.append(1.0 * (1.0 - w) + acc * w)

    return np.minimum.reduce(adjusted)


def predict_fv_weighted_mean(
    component_accs: List[np.ndarray],
    weights: List[float],
) -> np.ndarray:
    """Predict using FV-weighted mean: Σ_i w_i * a_i(t) / Σ_i w_i."""
    abs_weights = np.array([abs(w) for w in weights])
    total = abs_weights.sum()
    if total < 1e-10:
        return predict_mean(component_accs)

    result = np.zeros_like(component_accs[0])
    for acc, w in zip(component_accs, abs_weights):
        result += w * acc
    return result / total


def predict_fv_logit_linear(
    component_accs: List[np.ndarray],
    weights: List[float],
) -> np.ndarray:
    """Predict by composing in logit space: σ(Σ_i w_i * logit(a_i(t))).

    This models the idea that skills combine multiplicatively in log-odds space.
    A natural model for AND-gate composition.
    """
    abs_weights = np.array([abs(w) for w in weights])
    total = abs_weights.sum()
    if total < 1e-10:
        return predict_mean(component_accs)
    norm_weights = abs_weights / total

    # Work in logit space
    logit_sum = np.zeros_like(component_accs[0])
    for acc, w in zip(component_accs, norm_weights):
        # Clamp to avoid log(0) or log(inf)
        safe_acc = np.clip(acc, 0.01, 0.99)
        logit_sum += w * np.log(safe_acc / (1.0 - safe_acc))

    # Sigmoid back
    return 1.0 / (1.0 + np.exp(-logit_sum))


# ============================================================================
# FV-Discovered Prediction (No COMPONENT_MAP)
# ============================================================================

def discover_components_from_fv(
    decomposition: FVDecomposition,
    trajectories: Dict[str, TaskTrajectory],
    top_k: int = 5,
    threshold: float = 0.1,
) -> Tuple[List[str], List[float]]:
    """Discover which elemental tasks are relevant purely from FV similarities.

    Returns the top-k elemental tasks with highest |weight| above threshold,
    filtered to only those that have accuracy trajectories.

    Returns:
        (task_names, weights) — sorted by |weight| descending
    """
    # Filter to tasks with trajectories and significant weight
    candidates = []
    for name, w in decomposition.elemental_weights.items():
        if name in trajectories and abs(w) >= threshold:
            candidates.append((name, w))

    # Sort by absolute weight, take top k
    candidates.sort(key=lambda x: abs(x[1]), reverse=True)
    candidates = candidates[:top_k]

    if not candidates:
        return [], []

    names, weights = zip(*candidates)
    return list(names), list(weights)


# ============================================================================
# Visualization
# ============================================================================

def plot_fv_prediction(
    comp_task: str,
    actual_traj: TaskTrajectory,
    component_trajs: List[TaskTrajectory],
    component_weights: List[float],
    predictions: Dict[str, np.ndarray],
    metrics: Dict[str, Dict[str, float]],
    discovered_components: Optional[List[str]],
    discovered_weights: Optional[List[float]],
    output_path: Path,
    smooth_sigma: float = 1.0,
):
    """Plot FV-weighted predictions vs behavioral baselines."""
    n_components = len(component_trajs)
    n_rows = max(n_components, 2)

    fig = plt.figure(figsize=(20, max(7, n_rows * 2.5)))
    gs = fig.add_gridspec(n_rows, 3, width_ratios=[3, 1, 1], hspace=0.4, wspace=0.3)

    # Left panel: predictions
    ax_main = fig.add_subplot(gs[:, 0])

    tokens = actual_traj.tokens

    # Plot actual
    actual_smooth = smooth_trajectory(actual_traj.accuracy, sigma=smooth_sigma)
    ax_main.scatter(tokens, actual_traj.accuracy, s=20, alpha=0.2, color='black', zorder=9)
    ax_main.plot(tokens, actual_smooth, 'k-', linewidth=3, alpha=0.8,
                 label='Actual', marker='o', markersize=5,
                 markevery=max(1, len(tokens) // 10), zorder=10)

    # Color scheme: 3 categories with consistent palettes
    #   Baseline (human-decided, unweighted) — grays, dashed
    #   FV + human mapping (known components, FV weights) — blues, solid
    #   FV-discovered (FV picks components & weights) — oranges/reds, dash-dot
    CATEGORY_COLORS = {
        'baseline': {'product': '#888888', 'min': '#aaaaaa', 'mean': '#666666'},
        'fv_known': {'fv_product': '#1f77b4', 'fv_min': '#6baed6',
                     'fv_mean': '#4292c6', 'fv_logit': '#08519c'},
        'fv_disc':  {'discovered_product': '#e6550d', 'discovered_min': '#fd8d3c'},
    }
    method_styles = {
        # Baseline (gray, dashed)
        'product': (CATEGORY_COLORS['baseline']['product'], '--', 1.5),
        'min': (CATEGORY_COLORS['baseline']['min'], '--', 1.5),
        'mean': (CATEGORY_COLORS['baseline']['mean'], '--', 1.5),
        # FV + human mapping (blue, solid)
        'fv_product': (CATEGORY_COLORS['fv_known']['fv_product'], '-', 2.0),
        'fv_min': (CATEGORY_COLORS['fv_known']['fv_min'], '-', 2.0),
        'fv_mean': (CATEGORY_COLORS['fv_known']['fv_mean'], '-', 2.0),
        'fv_logit': (CATEGORY_COLORS['fv_known']['fv_logit'], '-', 2.0),
        # FV-discovered (orange/red, dash-dot)
        'discovered_product': (CATEGORY_COLORS['fv_disc']['discovered_product'], '-.', 2.0),
        'discovered_min': (CATEGORY_COLORS['fv_disc']['discovered_min'], '-.', 2.0),
    }

    # Category display names for legend grouping
    CATEGORY_LABELS = {
        'baseline': 'Baseline (unweighted)',
        'fv_known': 'FV + known components',
        'fv_disc': 'FV-discovered components',
    }

    def _method_category(m):
        if m.startswith('discovered'):
            return 'fv_disc'
        elif m.startswith('fv_'):
            return 'fv_known'
        return 'baseline'

    # Plot predictions grouped by category
    sorted_methods = sorted(predictions.keys(),
                            key=lambda m: metrics[m]['r2'], reverse=True)

    # Group by category for legend
    from itertools import groupby
    cat_order = ['baseline', 'fv_known', 'fv_disc']
    methods_by_cat = {c: [] for c in cat_order}
    for method in sorted_methods:
        methods_by_cat[_method_category(method)].append(method)

    for cat in cat_order:
        if not methods_by_cat[cat]:
            continue
        # Add category header as invisible legend entry
        ax_main.plot([], [], ' ', label=f'── {CATEGORY_LABELS[cat]} ──')
        for method in methods_by_cat[cat]:
            pred = predictions[method]
            r2 = metrics[method]['r2']
            mae = metrics[method]['mae']
            color, ls, lw = method_styles.get(method, ('gray', ':', 1.0))
            ax_main.plot(tokens, pred, color=color, linestyle=ls, linewidth=lw,
                         alpha=0.8, label=f'  {method} (R²={r2:.3f})')

    ax_main.set_xlabel('Training Tokens (B)', fontsize=12)
    ax_main.set_ylabel('Accuracy', fontsize=12)
    ax_main.set_title(f'{comp_task}', fontsize=13, fontweight='bold')
    ax_main.legend(fontsize=7, loc='best', ncol=1, framealpha=0.9)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylim(-0.05, 1.05)

    # Middle column: Component accuracy panels
    comp_colors = plt.cm.Set2(np.linspace(0, 0.6, n_components))
    for i, comp_traj in enumerate(component_trajs):
        if i >= n_rows:
            break
        ax = fig.add_subplot(gs[i, 1])
        comp_interp = comp_traj.interpolate(tokens)
        comp_smooth = smooth_trajectory(comp_interp, sigma=smooth_sigma)
        label = comp_traj.task_name.replace('simple_icl:', '').replace('simple_icl_', '')

        ax.plot(tokens, comp_smooth, '-', linewidth=2, color=comp_colors[i])
        ax.scatter(tokens, comp_interp, s=10, alpha=0.2, color=comp_colors[i])

        w = component_weights[i] if i < len(component_weights) else 0
        ax.set_title(f'{label}\nw={w:.3f}', fontsize=9, color=comp_colors[i], fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        if i == n_rows - 1:
            ax.set_xlabel('Tokens (B)', fontsize=9)

    # Right column: FV weight bar chart
    ax_bar = fig.add_subplot(gs[:, 2])
    all_weights = sorted(
        [(n, w) for n, w in zip(
            [t.task_name for t in component_trajs], component_weights
        )],
        key=lambda x: abs(x[1]), reverse=True
    )
    if all_weights:
        names_short = [n.replace('simple_icl:', '').replace('textfrct:', 'tf:')
                       for n, _ in all_weights]
        vals = [w for _, w in all_weights]
        colors = ['red' if v > 0 else 'blue' for v in vals]
        y_pos = range(len(names_short))
        ax_bar.barh(y_pos, vals, color=colors, alpha=0.7)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(names_short, fontsize=8)
        ax_bar.set_xlabel('FV Weight (cosine sim)', fontsize=9)
        ax_bar.set_title('Component Weights', fontsize=10)
        ax_bar.axvline(x=0, color='k', linewidth=0.5)
        ax_bar.grid(True, alpha=0.3, axis='x')
        ax_bar.invert_yaxis()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_fv_vs_baseline_summary(
    results_df: pd.DataFrame,
    output_path: Path,
):
    """Summary plot: does FV weighting improve over behavioral baselines?"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Consistent 3-category color scheme
    CAT_FACE = {'baseline': '#d9d9d9', 'fv_known': '#c6dbef', 'fv_disc': '#fdd0a2'}
    CAT_EDGE = {'baseline': '#888888', 'fv_known': '#1f77b4', 'fv_disc': '#e6550d'}

    def _cat(m):
        if m.startswith('discovered'):
            return 'fv_disc'
        elif m.startswith('fv_'):
            return 'fv_known'
        return 'baseline'

    # Gather method pairs for comparison
    baseline_methods = ['product', 'min', 'mean']
    fv_methods = ['fv_product', 'fv_min', 'fv_mean', 'fv_logit']
    discovered_methods = ['discovered_product', 'discovered_min']
    all_methods = baseline_methods + fv_methods + discovered_methods

    present_methods = [m for m in all_methods if f'{m}_r2' in results_df.columns]

    # ===== Top Left: R² comparison =====
    ax = axes[0, 0]
    r2_data = []
    labels = []
    for m in present_methods:
        col = f'{m}_r2'
        data = results_df[col].dropna()
        if len(data) > 0:
            r2_data.append(data.values)
            labels.append(m)
    if r2_data:
        bp = ax.boxplot(r2_data, tick_labels=labels, patch_artist=True)
        for patch, method in zip(bp['boxes'], labels):
            c = _cat(method)
            patch.set_facecolor(CAT_FACE[c])
            patch.set_edgecolor(CAT_EDGE[c])
            patch.set_linewidth(1.5)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('Prediction Quality (R²)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    # Add category legend
    from matplotlib.patches import Patch
    cat_legend = [
        Patch(facecolor=CAT_FACE['baseline'], edgecolor=CAT_EDGE['baseline'], label='Baseline (unweighted)'),
        Patch(facecolor=CAT_FACE['fv_known'], edgecolor=CAT_EDGE['fv_known'], label='FV + known components'),
        Patch(facecolor=CAT_FACE['fv_disc'], edgecolor=CAT_EDGE['fv_disc'], label='FV-discovered'),
    ]
    ax.legend(handles=cat_legend, fontsize=8, loc='lower right')

    # ===== Top Right: MAE comparison =====
    ax = axes[0, 1]
    mae_data = []
    labels = []
    for m in present_methods:
        col = f'{m}_mae'
        if col in results_df.columns:
            data = results_df[col].dropna()
            if len(data) > 0:
                mae_data.append(data.values)
                labels.append(m)
    if mae_data:
        bp = ax.boxplot(mae_data, tick_labels=labels, patch_artist=True)
        for patch, method in zip(bp['boxes'], labels):
            c = _cat(method)
            patch.set_facecolor(CAT_FACE[c])
            patch.set_edgecolor(CAT_EDGE[c])
            patch.set_linewidth(1.5)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('Prediction Error (MAE)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # ===== Bottom Left: Pairwise improvement (fv vs baseline) =====
    ax = axes[1, 0]
    for baseline, fv_method, color, marker in [
        ('product', 'fv_product', CAT_EDGE['fv_known'], 'o'),
        ('min', 'fv_min', '#6baed6', 's'),
        ('mean', 'fv_mean', '#4292c6', '^'),
    ]:
        b_col = f'{baseline}_r2'
        f_col = f'{fv_method}_r2'
        if b_col in results_df.columns and f_col in results_df.columns:
            valid = results_df[[b_col, f_col]].notna().all(axis=1)
            if valid.sum() > 0:
                ax.scatter(results_df.loc[valid, b_col],
                           results_df.loc[valid, f_col],
                           alpha=0.6, s=60, label=f'{baseline} → {fv_method}',
                           color=color, marker=marker)
    lim = [-0.05, 1.05]
    ax.plot(lim, lim, 'k--', alpha=0.5, linewidth=1, label='No improvement')
    ax.set_xlabel('Baseline R²', fontsize=12)
    ax.set_ylabel('FV-Weighted R²', fontsize=12)
    ax.set_title('Per-Task: Does FV Weighting Help?', fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    # ===== Bottom Right: Best method distribution =====
    ax = axes[1, 1]
    if 'best_method' in results_df.columns:
        method_counts = results_df['best_method'].value_counts()
        bar_face = [CAT_FACE[_cat(m)] for m in method_counts.index]
        bar_edge = [CAT_EDGE[_cat(m)] for m in method_counts.index]
        ax.bar(range(len(method_counts)), method_counts.values,
               color=bar_face, edgecolor=bar_edge, linewidth=1.5)
        ax.set_xticks(range(len(method_counts)))
        ax.set_xticklabels(method_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Best Method per Task', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Predict compositional task performance using FV decomposition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--fv_dir", type=str, required=True,
                        help="Directory with skill_basis.npz and test_vectors/ "
                             "(output of analyze_real_tasks.py)")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory with accuracy_pivot_*.csv files")
    parser.add_argument("--output_dir", type=str, default="plots/fv_prediction",
                        help="Output directory for plots and results")
    parser.add_argument("--weight_method", type=str, default="cosine",
                        choices=["cosine", "lstsq"],
                        help="How to compute FV weights (default: cosine)")
    parser.add_argument("--discover_top_k", type=int, default=5,
                        help="Number of top FV-similar elemental tasks for discovery mode")
    parser.add_argument("--discover_threshold", type=float, default=0.1,
                        help="Minimum |weight| to include a discovered component")
    parser.add_argument("--smooth_sigma", type=float, default=1.0,
                        help="Gaussian smoothing sigma (0 = no smoothing)")
    parser.add_argument("--exclude_pattern", type=str, nargs='*', default=['reverse', 'last'],
                        help="Exclude compositional tasks whose name contains any of these substrings "
                             "(e.g., --exclude_pattern reverse). Default: ['reverse', 'last']")

    args = parser.parse_args()

    fv_dir = Path(args.fv_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FV-WEIGHTED COMPOSITIONAL PREDICTION")
    print("=" * 70)
    print(f"FV dir:      {fv_dir}")
    print(f"Results dir: {results_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Weight method: {args.weight_method}")
    print(f"Smoothing σ:   {args.smooth_sigma}")
    if args.exclude_pattern:
        print(f"Excluding:     {args.exclude_pattern}")
    print()

    # ── Load data ────────────────────────────────────────────────────
    print("Loading skill basis...")
    basis = load_skill_basis(str(fv_dir / "skill_basis.npz"))
    print(f"  Basis: {basis.U.shape[1]} components from {len(basis.task_names)} training tasks")
    print(f"  Method: {basis.method}")

    test_vec_dir = fv_dir / "test_vectors"
    print(f"\nLoading all FVs (train from basis + test from {test_vec_dir})...")
    all_fvs = load_all_fvs(basis, test_vec_dir)
    print(f"  Loaded {len(all_fvs)} total FVs")

    elemental_names = get_elemental_task_names(all_fvs)
    comp_names = [n for n in all_fvs if n.startswith("compositional")]
    print(f"  Elemental: {len(elemental_names)}, Compositional: {len(comp_names)}")

    print("\nLoading accuracy trajectories...")
    trajectories = load_all_trajectories(results_dir)
    print(f"  Loaded {len(trajectories)} trajectory files")

    # ── Decompose each compositional task ────────────────────────────
    print("\n" + "=" * 70)
    print("DECOMPOSING COMPOSITIONAL TASKS")
    print("=" * 70)

    decompositions = {}
    for comp_task in sorted(comp_names):
        decomp = decompose_compositional_task(
            comp_task, all_fvs, elemental_names,
            weight_method=args.weight_method,
        )
        if decomp is not None:
            decompositions[comp_task] = decomp

    print(f"\nDecomposed {len(decompositions)} compositional tasks")

    # ── Apply exclusion filter ──
    if args.exclude_pattern:
        decompositions_filtered = {}
        excluded = []
        for k, v in decompositions.items():
            if any(pat in k for pat in args.exclude_pattern):
                excluded.append(k)
            else:
                decompositions_filtered[k] = v
        decompositions = decompositions_filtered
        print(f"  Excluded {len(excluded)} tasks matching {args.exclude_pattern}:")
        for t in excluded:
            print(f"    - {t}")
        print(f"  Remaining: {len(decompositions)} tasks")

    # Print decomposition details
    for comp_task, decomp in sorted(decompositions.items()):
        print(f"\n  {comp_task}")
        if decomp.known_components:
            print(f"    Known components: {decomp.known_components}")

        # Show top FV weights
        sorted_weights = sorted(decomp.elemental_weights.items(),
                                key=lambda x: abs(x[1]), reverse=True)
        print(f"    Top FV similarities:")
        for name, w in sorted_weights[:8]:
            marker = " ★" if decomp.known_components and name in decomp.known_components else ""
            print(f"      {name:40s}: {w:+.4f}{marker}")

    # ── Run predictions ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PREDICTING COMPOSITIONAL ACCURACY TRAJECTORIES")
    print("=" * 70)

    results = []

    for comp_task, decomp in sorted(decompositions.items()):
        # Check we have trajectory data
        if comp_task not in trajectories:
            print(f"\n  ⚠️  No trajectory for {comp_task}, skipping")
            continue

        actual_traj = trajectories[comp_task]
        tokens = actual_traj.tokens
        actual_smooth = smooth_trajectory(actual_traj.accuracy, sigma=args.smooth_sigma)

        print(f"\n{'─' * 60}")
        print(f"{comp_task}")
        print(f"{'─' * 60}")

        predictions = {}
        all_metrics = {}

        # ── Level 1: Behavioral baselines (unweighted, known components) ──
        known_components = decomp.known_components
        if known_components and all(c in trajectories for c in known_components):
            comp_trajs = [trajectories[c] for c in known_components]
            comp_accs = [smooth_trajectory(t.interpolate(tokens), sigma=args.smooth_sigma)
                         for t in comp_trajs]

            predictions['product'] = predict_multiplicative(comp_accs)
            predictions['min'] = predict_min(comp_accs)
            predictions['mean'] = predict_mean(comp_accs)

            # ── Level 2: FV-weighted (known components + FV weights) ──
            fv_weights = [decomp.elemental_weights.get(c, 0.0) for c in known_components]

            predictions['fv_product'] = predict_fv_weighted_product(comp_accs, fv_weights)
            predictions['fv_min'] = predict_fv_weighted_min(comp_accs, fv_weights)
            predictions['fv_mean'] = predict_fv_weighted_mean(comp_accs, fv_weights)
            predictions['fv_logit'] = predict_fv_logit_linear(comp_accs, fv_weights)

            print(f"  Known components: {[c.split(':')[-1] for c in known_components]}")
            print(f"  FV weights:       {[f'{w:.3f}' for w in fv_weights]}")
        else:
            missing = [c for c in (known_components or []) if c not in trajectories]
            if missing:
                print(f"  ⚠️  Missing trajectories for known components: {missing}")

        # ── Level 3: FV-discovered components ──
        disc_names, disc_weights = discover_components_from_fv(
            decomp, trajectories,
            top_k=args.discover_top_k,
            threshold=args.discover_threshold,
        )

        if disc_names:
            disc_trajs = [trajectories[n] for n in disc_names]
            disc_accs = [smooth_trajectory(t.interpolate(tokens), sigma=args.smooth_sigma)
                         for t in disc_trajs]

            predictions['discovered_product'] = predict_fv_weighted_product(disc_accs, disc_weights)
            predictions['discovered_min'] = predict_fv_weighted_min(disc_accs, disc_weights)

            print(f"  Discovered components (top {args.discover_top_k}, thresh={args.discover_threshold}):")
            for n, w in zip(disc_names, disc_weights):
                print(f"    {n:40s}: {w:+.4f}")

        # ── Evaluate all predictions ──
        print(f"\n  Prediction quality:")
        for method, pred in predictions.items():
            m = evaluate_prediction(pred, actual_smooth)
            all_metrics[method] = m
            print(f"    {method:25s}: R²={m['r2']:.3f}, MAE={m['mae']:.3f}")

        if all_metrics:
            best = max(all_metrics, key=lambda m: all_metrics[m]['r2'])
            print(f"  → Best: {best} (R²={all_metrics[best]['r2']:.3f})")

        # ── Save row ──
        result_row = {
            'task': comp_task,
            'n_known_components': len(known_components) if known_components else 0,
            'known_components': ' + '.join([c.split(':')[-1] for c in known_components]) if known_components else '',
            'n_discovered_components': len(disc_names),
            'discovered_components': ' + '.join([n.split(':')[-1] for n in disc_names]),
            'best_method': best if all_metrics else '',
        }
        for method, m in all_metrics.items():
            result_row[f'{method}_r2'] = m['r2']
            result_row[f'{method}_mae'] = m['mae']
            result_row[f'{method}_rmse'] = m['rmse']
            result_row[f'{method}_max_error'] = m['max_error']

        # Add FV weights for known components
        if known_components:
            for c in known_components:
                c_short = c.split(':')[-1]
                result_row[f'fv_weight_{c_short}'] = decomp.elemental_weights.get(c, 0.0)

        results.append(result_row)

        # ── Plot ──
        if known_components and all(c in trajectories for c in known_components):
            comp_trajs_for_plot = [trajectories[c] for c in known_components]
            fv_weights_for_plot = [decomp.elemental_weights.get(c, 0.0) for c in known_components]
        else:
            comp_trajs_for_plot = []
            fv_weights_for_plot = []

        plot_path = output_dir / f"{comp_task.replace(':', '_')}_fv_prediction.png"
        plot_fv_prediction(
            comp_task, actual_traj,
            comp_trajs_for_plot, fv_weights_for_plot,
            predictions, all_metrics,
            disc_names, disc_weights,
            plot_path, smooth_sigma=args.smooth_sigma,
        )

    # ── Summary ──────────────────────────────────────────────────────
    if not results:
        print("\nNo results to summarize!")
        return

    results_df = pd.DataFrame(results)

    csv_path = output_dir / "fv_prediction_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved results to: {csv_path}")

    # Summary plot
    summary_path = output_dir / "fv_vs_baseline_summary.png"
    plot_fv_vs_baseline_summary(results_df, summary_path)
    print(f"✅ Saved summary plot to: {summary_path}")

    # ── Print summary statistics ──
    print("\n" + "=" * 70)
    print("SUMMARY (all tasks, varying N per method)")
    print("=" * 70)

    baseline_methods = ['product', 'min', 'mean']
    fv_methods = ['fv_product', 'fv_min', 'fv_mean', 'fv_logit']
    discovered_methods = ['discovered_product', 'discovered_min']
    all_methods = baseline_methods + fv_methods + discovered_methods

    print(f"\n{'Method':<25s} {'Mean R²':>8s} {'Mean MAE':>9s} {'N':>4s} {'Best':>5s}")
    print("-" * 56)

    for method in all_methods:
        r2_col = f'{method}_r2'
        mae_col = f'{method}_mae'
        if r2_col in results_df.columns:
            r2 = results_df[r2_col].dropna()
            mae = results_df[mae_col].dropna() if mae_col in results_df.columns else pd.Series()
            n_best = (results_df['best_method'] == method).sum() if 'best_method' in results_df.columns else 0
            prefix = "  " if method.startswith('fv_') or method.startswith('discovered') else ""
            print(f"{prefix}{method:<23s} {r2.mean():>8.3f} {mae.mean():>9.3f} {len(r2):>4d} {n_best:>5d}")

    # ── Fair head-to-head comparison ──
    # Only average over tasks where ALL methods have predictions (apples-to-apples)
    print(f"\n{'=' * 70}")
    print("FAIR COMPARISON (only tasks where all methods have predictions)")
    print(f"{'=' * 70}")

    r2_cols = [f'{m}_r2' for m in all_methods if f'{m}_r2' in results_df.columns]
    fair_mask = results_df[r2_cols].notna().all(axis=1)
    n_fair = fair_mask.sum()
    fair_tasks = results_df.loc[fair_mask, 'task'].tolist()

    print(f"\n  Tasks with all methods ({n_fair}): {[t.split(':')[-1] for t in fair_tasks]}")

    if n_fair > 0:
        fair_df = results_df[fair_mask]

        print(f"\n{'Method':<25s} {'Mean R²':>8s} {'Mean MAE':>9s} {'Best':>5s}")
        print("-" * 52)

        for method in all_methods:
            r2_col = f'{method}_r2'
            mae_col = f'{method}_mae'
            if r2_col in fair_df.columns:
                r2 = fair_df[r2_col]
                mae = fair_df[mae_col] if mae_col in fair_df.columns else pd.Series()
                # Recompute best_method for fair subset
                fair_r2_cols = {m: f'{m}_r2' for m in all_methods if f'{m}_r2' in fair_df.columns}
                fair_r2_values = fair_df[[c for c in fair_r2_cols.values()]]
                fair_best = fair_r2_values.idxmax(axis=1).str.replace('_r2$', '', regex=True)
                n_best = (fair_best == method).sum()
                prefix = "  " if method.startswith('fv_') or method.startswith('discovered') else ""
                print(f"{prefix}{method:<23s} {r2.mean():>8.3f} {mae.mean():>9.3f} {n_best:>5d}")

        # Show per-task breakdown for fair subset
        print(f"\n  Per-task R² (fair subset):")
        print(f"  {'Task':<30s}", end="")
        short_methods = all_methods
        for m in short_methods:
            if f'{m}_r2' in fair_df.columns:
                print(f" {m:>12s}", end="")
        print(f" {'best':>14s}")
        print(f"  {'─' * 30}", end="")
        for m in short_methods:
            if f'{m}_r2' in fair_df.columns:
                print(f" {'─' * 12}", end="")
        print(f" {'─' * 14}")

        for _, row in fair_df.iterrows():
            task_short = row['task'].split(':')[-1] if ':' in row['task'] else row['task']
            print(f"  {task_short:<30s}", end="")
            r2_vals = {}
            for m in short_methods:
                r2_col = f'{m}_r2'
                if r2_col in fair_df.columns:
                    val = row[r2_col]
                    r2_vals[m] = val
                    print(f" {val:>12.3f}", end="")
            best_m = max(r2_vals, key=r2_vals.get) if r2_vals else ''
            print(f" {best_m:>14s}")
    else:
        print("\n  ⚠️  No tasks have predictions from all methods.")
        print("      (Some tasks are missing known component trajectories.)")

    # Improvement analysis
    print(f"\n{'─' * 52}")
    print("IMPROVEMENT OVER BASELINES")
    print(f"{'─' * 52}")
    for baseline, fv_method in [('product', 'fv_product'), ('min', 'fv_min'), ('mean', 'fv_mean')]:
        b_col = f'{baseline}_r2'
        f_col = f'{fv_method}_r2'
        if b_col in results_df.columns and f_col in results_df.columns:
            valid = results_df[[b_col, f_col]].notna().all(axis=1)
            if valid.sum() > 0:
                diff = results_df.loc[valid, f_col] - results_df.loc[valid, b_col]
                n_improved = (diff > 0).sum()
                n_total = valid.sum()
                print(f"  {baseline} → {fv_method}:")
                print(f"    Improved: {n_improved}/{n_total} tasks ({n_improved/n_total*100:.0f}%)")
                print(f"    Mean ΔR²: {diff.mean():+.4f}")

    # Best method distribution
    if 'best_method' in results_df.columns:
        print(f"\nBEST METHOD DISTRIBUTION:")
        method_counts = results_df['best_method'].value_counts()
        for method, count in method_counts.items():
            pct = count / len(results_df) * 100
            cat = "FV" if method.startswith('fv_') else ("DISC" if method.startswith('discovered') else "BASE")
            print(f"  [{cat:4s}] {method:25s}: {count:2d} tasks ({pct:5.1f}%)")

    # ── Write summary.md ──────────────────────────────────────────────
    write_summary_md(results_df, output_dir, args)

    print(f"\n✅ All results saved to: {output_dir}")


# ============================================================================
# Summary Writer
# ============================================================================

def write_summary_md(results_df: pd.DataFrame, output_dir: Path, args) -> None:
    """Write a markdown summary of FV prediction results to summary.md."""
    baseline_methods = ['product', 'min', 'mean']
    fv_methods = ['fv_product', 'fv_min', 'fv_mean', 'fv_logit']
    discovered_methods = ['discovered_product', 'discovered_min']
    all_methods = baseline_methods + fv_methods + discovered_methods

    summary_path = output_dir / "summary.md"
    buf = []
    w = buf.append

    w("# FV-Weighted Compositional Prediction — Summary")
    w("")
    w(f"**FV dir:** `{args.fv_dir}`  ")
    w(f"**Results dir:** `{args.results_dir}`  ")
    w(f"**Weight method:** `{args.weight_method}`  ")
    w(f"**Smoothing σ:** `{args.smooth_sigma}`  ")
    w(f"**Tasks evaluated:** {len(results_df)}")
    w("")

    # ── Method comparison table ──
    w("## Method Comparison (all tasks)")
    w("")
    w(f"| Category | Method | Mean R² | Mean MAE | N | # Best |")
    w(f"|----------|--------|--------:|---------:|--:|-------:|")
    for method in all_methods:
        r2_col = f'{method}_r2'
        mae_col = f'{method}_mae'
        if r2_col not in results_df.columns:
            continue
        r2 = results_df[r2_col].dropna()
        mae = results_df[mae_col].dropna() if mae_col in results_df.columns else pd.Series()
        n_best = (results_df.get('best_method', pd.Series()) == method).sum()
        if method.startswith('fv_'):
            cat = 'FV+known'
        elif method.startswith('discovered'):
            cat = 'FV-disc'
        else:
            cat = 'Baseline'
        mean_r2 = f'{r2.mean():.3f}' if len(r2) > 0 else 'N/A'
        mean_mae = f'{mae.mean():.3f}' if len(mae) > 0 else 'N/A'
        w(f"| {cat} | `{method}` | {mean_r2} | {mean_mae} | {len(r2)} | {n_best} |")
    w("")

    # ── Fair comparison ──
    r2_cols = [f'{m}_r2' for m in all_methods if f'{m}_r2' in results_df.columns]
    fair_mask = results_df[r2_cols].notna().all(axis=1)
    n_fair = fair_mask.sum()
    if n_fair > 0:
        fair_df = results_df[fair_mask]
        w(f"## Fair Comparison ({n_fair} tasks with all methods)")
        w("")
        fair_tasks_short = [t.split(':')[-1] for t in fair_df['task'].tolist()]
        w(f"Tasks: {', '.join(f'`{t}`' for t in fair_tasks_short)}")
        w("")
        w(f"| Category | Method | Mean R² | Mean MAE |")
        w(f"|----------|--------|--------:|---------:|")
        for method in all_methods:
            r2_col = f'{method}_r2'
            mae_col = f'{method}_mae'
            if r2_col not in fair_df.columns:
                continue
            r2 = fair_df[r2_col]
            mae = fair_df[mae_col] if mae_col in fair_df.columns else pd.Series()
            if method.startswith('fv_'):
                cat = 'FV+known'
            elif method.startswith('discovered'):
                cat = 'FV-disc'
            else:
                cat = 'Baseline'
            w(f"| {cat} | `{method}` | {r2.mean():.3f} | {mae.mean():.3f} |")
        w("")

        # Per-task breakdown
        w("### Per-Task R² (fair subset)")
        w("")
        present_methods = [m for m in all_methods if f'{m}_r2' in fair_df.columns]
        header = '| Task | ' + ' | '.join(f'`{m}`' for m in present_methods) + ' | Best |'
        sep = '|------|' + '|'.join(['------:'] * len(present_methods)) + '|------|'
        w(header)
        w(sep)
        for _, row in fair_df.iterrows():
            task_short = row['task'].split(':')[-1] if ':' in row['task'] else row['task']
            r2_vals = {m: row[f'{m}_r2'] for m in present_methods if f'{m}_r2' in fair_df.columns}
            best_m = max(r2_vals, key=r2_vals.get) if r2_vals else ''
            vals_str = ' | '.join(f'{r2_vals[m]:.3f}' for m in present_methods)
            w(f'| `{task_short}` | {vals_str} | `{best_m}` |')
        w("")

    # ── Improvement over baselines ──
    w("## FV Weighting vs Baseline")
    w("")
    w("| Baseline | FV Method | Improved (N/Total) | Mean ΔR² |")
    w("|----------|-----------|-------------------:|---------:|")
    for baseline, fv_method in [('product', 'fv_product'), ('min', 'fv_min'), ('mean', 'fv_mean')]:
        b_col = f'{baseline}_r2'
        f_col = f'{fv_method}_r2'
        if b_col in results_df.columns and f_col in results_df.columns:
            valid = results_df[[b_col, f_col]].notna().all(axis=1)
            if valid.sum() > 0:
                diff = results_df.loc[valid, f_col] - results_df.loc[valid, b_col]
                n_improved = (diff > 0).sum()
                n_total = valid.sum()
                w(f"| `{baseline}` | `{fv_method}` | {n_improved}/{n_total} ({n_improved/n_total*100:.0f}%) | {diff.mean():+.4f} |")
    w("")

    # ── Best method distribution ──
    if 'best_method' in results_df.columns:
        w("## Best Method Distribution")
        w("")
        w("| Method | Category | Count | % |")
        w("|--------|----------|------:|--:|")
        method_counts = results_df['best_method'].value_counts()
        for method, count in method_counts.items():
            pct = count / len(results_df) * 100
            cat = 'FV+known' if method.startswith('fv_') else ('FV-disc' if method.startswith('discovered') else 'Baseline')
            w(f"| `{method}` | {cat} | {count} | {pct:.1f}% |")
        w("")

    # ── Per-task FV weights for known components ──
    fv_weight_cols = [c for c in results_df.columns if c.startswith('fv_weight_')]
    if fv_weight_cols:
        w("## Per-Task FV Weights (Known Components)")
        w("")
        w("| Task | Known Components | " + " | ".join(c.replace('fv_weight_', '') for c in fv_weight_cols) + " |")
        w("|------|-----------------|" + "|".join(["------:"] * len(fv_weight_cols)) + "|")
        for _, row in results_df.iterrows():
            task_short = row['task'].split(':')[-1] if ':' in row['task'] else row['task']
            comp_str = row.get('known_components', '')
            vals = ' | '.join(
                f'{row[c]:.3f}' if pd.notna(row.get(c)) else 'N/A'
                for c in fv_weight_cols
            )
            w(f"| `{task_short}` | {comp_str} | {vals} |")
        w("")

    with open(summary_path, 'w') as f:
        f.write('\n'.join(buf) + '\n')
    print(f"✅ Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()

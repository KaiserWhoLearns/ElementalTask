#!/usr/bin/env python3
"""
Comprehensive analysis of function vectors from real ICL tasks.

This script:
1. Discovers all tasks that support ICL
2. Extracts function vectors from a subset (training tasks)
3. Builds a skill basis using SVD
4. Tests reconstruction of held-out tasks (test tasks)
5. Analyzes epsilon-ranks and similarity patterns
6. Saves detailed results and visualizations
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tasks.base_task import BaseTask, TaskConfig
from function_vecs.extract_function_vecs import (
    ExtractConfig,
    extract_task_function_vec,
    stack_function_vecs,
    build_skill_basis,
    save_function_vec,
    save_skill_basis,
    Headset,
    TaskFunctionVec,
)


def load_task_performance(
    results_dir: str,
    checkpoint: str = "main",
    task_names: List[str] = None
) -> Dict[str, float]:
    """
    Load final accuracy for each task from pivot CSV files.
    
    Args:
        results_dir: Path to results directory (e.g., "results/olmo2_continuous_1b_early_revised")
        checkpoint: Checkpoint to filter by (e.g., "main")
        task_names: Optional list of task names to load (loads all if None)
    
    Returns:
        Dict mapping task_name -> accuracy (0.0-1.0)
    """
    import pandas as pd
    
    results_path = Path(results_dir)
    performance = {}
    
    # Load from accuracy_pivot_*.csv files
    for pivot_file in results_path.glob("accuracy_pivot_*.csv"):
        try:
            df = pd.read_csv(pivot_file)
            
            # Get task name from column (third column after model, checkpoint)
            task_cols = [c for c in df.columns if c not in ['model', 'checkpoint']]
            if not task_cols:
                continue
            task_name = task_cols[0]
            
            # Filter to the requested checkpoint
            df_ckpt = df[df['checkpoint'] == checkpoint]
            if df_ckpt.empty:
                continue
            
            # Get accuracy
            acc = df_ckpt[task_name].values[0]
            
            if task_names and task_name not in task_names:
                continue
            
            performance[task_name] = float(acc)
            
        except Exception as e:
            # Skip files that can't be parsed
            continue
    
    return performance


def discover_icl_tasks() -> List[BaseTask]:
    """Discover all tasks that support ICL format, including subtasks."""
    print("\n" + "="*70)
    print("DISCOVERING ICL TASKS")
    print("="*70)
    
    from tasks.registry import discover_tasks, get_task, list_tasks
    
    # Discover base tasks
    discover_tasks()
    base_tasks = list_tasks()
    print(f"\nFound {len(base_tasks)} registered base tasks")
    
    # Define all tasks to try, including subtasks
    # These are the tasks we want to analyze
    task_names_to_try = [
        # Base tasks
        "basic_arithmetic",
        "copying",
        "token_reversal",
        "string_analogy",
        # "ignoring_context",  # May have format issues
        # "ioi_task",  # Known incompatible format
        
        # simple_icl subtasks
        "simple_icl:uppercase",
        "simple_icl:lowercase",
        "simple_icl:first_letter",
        "simple_icl:last_letter",
        "simple_icl:translate_eng_fr",
        "simple_icl:translate_fr_eng",
        "simple_icl:translate_eng_sp",
        "simple_icl:translate_sp_eng",
        "simple_icl:present_to_gerund",
        "simple_icl:singular_to_plural",
        "simple_icl:country_to_capital",
        "simple_icl:country_to_currency",
        
        # textfrct subtasks (objective ones with enough data)
        "textfrct:CV1",
        "textfrct:CV2",
        "textfrct:CV3",
        "textfrct:I1",
        "textfrct:I2",
        "textfrct:MA2",
        "textfrct:MA3",
        "textfrct:RG1",
        "textfrct:RG2",
        "textfrct:RG3",
        "textfrct:RL1",
        "textfrct:RL3",
        "textfrct:V1",
        "textfrct:V2",
        "textfrct:V3",
        "textfrct:V4",
        "textfrct:V5",
        
        # compositional subtasks
        "compositional:upper_reverse",
        "compositional:lower_reverse",
        "compositional:reverse_upper",
        "compositional:reverse_lower",
        "compositional:upper_first",
        "compositional:upper_last",
        "compositional:lower_first",
        "compositional:lower_last",
        "compositional:first_upper",
        "compositional:last_upper",
        "compositional:gerund_upper",
        "compositional:gerund_reverse",
        "compositional:plural_upper",
        "compositional:plural_reverse",
    ]
    
    print(f"\nAttempting to load {len(task_names_to_try)} tasks...")
    
    icl_tasks = []
    for task_name in task_names_to_try:
        try:
            task = get_task(task_name)
            
            # Store the full task name (including subtask) as an attribute
            # This is needed because task.config.name doesn't include the subtask
            task._full_name = task_name
            
            # Try to get sample data to verify task works
            try:
                sample_data = task.get_split("test")
                n_examples = len(sample_data) if sample_data else 0
                if n_examples > 0:
                    icl_tasks.append(task)
                    print(f"  ✓ {task_name} ({n_examples} examples)")
                else:
                    print(f"  ✗ {task_name} (no data)")
            except Exception as data_err:
                print(f"  ✗ {task_name} (data error: {data_err})")
                
        except Exception as e:
            print(f"  ✗ {task_name} (error: {e})")
    
    print(f"\n✓ Found {len(icl_tasks)} ICL-compatible tasks")
    return icl_tasks


def get_task_display_name(task: BaseTask) -> str:
    """Get the full display name for a task, including subtask if applicable."""
    # Check for our custom _full_name attribute first
    if hasattr(task, '_full_name'):
        return task._full_name
    # Fall back to config.name
    return task.config.name


def split_tasks(tasks: List[BaseTask], train_ratio: float = 0.7, seed: int = 42):
    """Split tasks into training and test sets."""
    np.random.seed(seed)
    n_train = max(1, int(len(tasks) * train_ratio))
    
    indices = np.random.permutation(len(tasks))
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_tasks = [tasks[i] for i in train_indices]
    test_tasks = [tasks[i] for i in test_indices]
    
    return train_tasks, test_tasks


def extract_function_vectors(
    tasks: List[BaseTask],
    config: ExtractConfig,
    headset: Headset,
    model,
    tokenizer,
    desc: str = "tasks"
):
    """Extract function vectors from a list of tasks.
    
    Tasks with no correct instances (when only_correct=True) will be skipped.
    """
    print(f"\nExtracting function vectors from {len(tasks)} {desc}...")
    if config.only_correct:
        print("  (strict mode: tasks with 0 correct instances will be skipped)")
    
    function_vecs = []
    skipped_no_correct = 0
    for i, task in enumerate(tasks, 1):
        try:
            task_name = get_task_display_name(task)
            print(f"  [{i}/{len(tasks)}] {task_name}...", end=" ")
            fv = extract_task_function_vec(task, config, headset, model, tokenizer)
            # Also store the full task name in the function vector
            fv.task_name = task_name
            function_vecs.append(fv)
            norm = np.linalg.norm(fv.function_vec)
            print(f"✓ (norm={norm:.4f})")
        except ValueError as e:
            if "no correct instances" in str(e).lower():
                print(f"⚠ SKIPPED (no correct instances)")
                skipped_no_correct += 1
            else:
                print(f"✗ ERROR: {e}")
            continue
        except Exception as e:
            print(f"✗ ERROR: {e}")
            continue
    
    print(f"✓ Successfully extracted {len(function_vecs)}/{len(tasks)} function vectors")
    if skipped_no_correct > 0:
        print(f"  ({skipped_no_correct} tasks skipped due to no correct instances)")
    return function_vecs


def analyze_epsilon_ranks(
    test_vecs: List,
    basis,
    epsilons: List[float] = [0.9, 0.5, 0.1, 0.01]
) -> Dict[str, Any]:
    """Analyze epsilon-ranks for test tasks."""
    print("\n" + "="*70)
    print("EPSILON-RANK ANALYSIS")
    print("="*70)
    
    print(f"\nEpsilon thresholds: {epsilons}")
    print("Metric: cosine distance (1 - cosine_similarity)\n")
    
    results = {}
    
    for vec in test_vecs:
        task_name = vec.task_name
        results[task_name] = {}
        
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print('='*60)
        
        # Get detailed reconstruction info - pass the TaskFunctionVec object
        details = basis.epsilon_rank(
            vec,  # Pass the whole TaskFunctionVec object
            epsilon=epsilons[-1],  # Use strictest epsilon for details
            metric="cosine",
            return_details=True
        )
        
        results[task_name]['projections'] = details['projections']
        results[task_name]['cosine_errors'] = details['cosine_errors']
        
        # Show projection onto basis
        print(f"\nProjection onto basis (top 5 components):")
        print(f"  {details['projections'][:5]}")
        
        # Show reconstruction quality by k
        print(f"\nReconstruction quality by number of components (k):")
        max_k = len(details['cosine_errors'])
        k_values = [1, 2, 3, 4, 5, 7, 10, 15, 20]
        # Add max_k if not in list
        if max_k not in k_values and max_k <= 30:
            k_values.append(max_k)
        k_values.sort()
        
        for k in k_values:
            if k < len(details['cosine_errors']):
                cos_err = details['cosine_errors'][k]
                cos_sim = 1.0 - cos_err
                print(f"  k={k:2d}: cosine_sim={cos_sim:.4f}, cosine_dist={cos_err:.4f}")
        
        # Compute epsilon-ranks for all thresholds
        print(f"\nEpsilon-ranks for different thresholds (lower ε = stricter):")
        print(f"  (Note: ε is cosine distance threshold, so ε=0.1 means cosine_sim ≥ 0.9)")
        epsilon_ranks = {}
        for eps in epsilons:
            rank = basis.epsilon_rank(vec, epsilon=eps, metric="cosine")  # Pass vec object
            epsilon_ranks[eps] = rank
            required_sim = 1.0 - eps
            print(f"  ε={eps:5.3f} (sim≥{required_sim:.2f}): k={rank}")
        
        results[task_name]['epsilon_ranks'] = epsilon_ranks
    
    return results


def compute_similarity_matrix(function_vecs: List) -> np.ndarray:
    """Compute pairwise cosine similarities between function vectors."""
    n = len(function_vecs)
    similarity_matrix = np.zeros((n, n))
    
    for i, fv_i in enumerate(function_vecs):
        for j, fv_j in enumerate(function_vecs):
            # Vectors are already L2-normalized, so dot product = cosine similarity
            similarity_matrix[i, j] = np.dot(fv_i.function_vec, fv_j.function_vec)
    
    return similarity_matrix


def print_summary(
    train_tasks: List[BaseTask],
    test_tasks: List[BaseTask],
    basis,
    results: Dict[str, Any],
    similarity_matrix: np.ndarray,
    test_vecs: List,
    train_vecs: List
):
    """Print comprehensive summary of results."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Basis info
    print("\n--- Skill Basis ---")
    print(f"Training tasks: {len(train_tasks)}")
    print(f"Basis dimensions: {basis.U.shape}")
    print(f"Components: {basis.U.shape[1]}")
    
    var_ratios = basis.explained_variance_ratio()
    print(f"\nExplained variance:")
    for k in [1, 2, 3, 5, min(10, len(var_ratios))]:
        if k <= len(var_ratios):
            print(f"  Top {k:2d} component(s): {var_ratios[k-1]:6.2%}")
    
    # Get unique epsilons from results (sorted)
    all_epsilons = set()
    for task_name, data in results.items():
        all_epsilons.update(data['epsilon_ranks'].keys())
    sorted_epsilons = sorted(all_epsilons, reverse=True)
    
    # Training task rankings
    print("\n--- Training Task Reconstruction Quality ---")
    print("(Used to build basis - shows task complexity)")
    
    # Build header
    header = f"{'Task':<30s} {'Type':<6s}"
    for eps in sorted_epsilons:
        header += f" {f'ε={eps:.2f}':>7s}"
    header += f" {'Best Sim':>10s}"
    print(header)
    print("-" * len(header))
    
    # Collect training task scores
    train_task_names = {get_task_display_name(t) for t in train_tasks}
    train_scores = []
    for vec in train_vecs:
        task_name = vec.task_name
        if task_name in results:
            epsilon_ranks = results[task_name]['epsilon_ranks']
            cos_errors = results[task_name]['cosine_errors']
            best_sim = 1.0 - cos_errors[-1]
            train_scores.append((task_name, epsilon_ranks, best_sim, 'TRAIN'))
    
    train_scores.sort(key=lambda x: x[2], reverse=True)  # Sort by best similarity
    
    for task_name, epsilon_ranks, best_sim, task_type in train_scores:
        row = f"{task_name:<30s} {task_type:<6s}"
        for eps in sorted_epsilons:
            rank = epsilon_ranks.get(eps, '-')
            row += f" {str(rank):>7s}"
        row += f" {best_sim:>10.4f}"
        print(row)
    
    # Test task rankings
    print("\n--- Test Task Reconstruction Quality ---")
    print("(Held-out tasks - tests generalization)")
    
    print(header)
    print("-" * len(header))
    
    # Sort by best similarity (k=max components)
    test_scores = []
    for vec in test_vecs:
        task_name = vec.task_name
        if task_name in results:
            epsilon_ranks = results[task_name]['epsilon_ranks']
            cos_errors = results[task_name]['cosine_errors']
            best_sim = 1.0 - cos_errors[-1]
            test_scores.append((task_name, epsilon_ranks, best_sim, 'TEST'))
    
    test_scores.sort(key=lambda x: x[2], reverse=True)  # Sort by best similarity
    
    for task_name, epsilon_ranks, best_sim, task_type in test_scores:
        row = f"{task_name:<30s} {task_type:<6s}"
        for eps in sorted_epsilons:
            rank = epsilon_ranks.get(eps, '-')
            row += f" {str(rank):>7s}"
        row += f" {best_sim:>10.4f}"
        print(row)
    
    # Combined complexity ranking (by epsilon-rank at moderate threshold)
    print("\n--- Task Complexity Ranking (by ε=0.05 epsilon-rank) ---")
    print("Lower rank = simpler/more fundamental task")
    
    # Use a moderate epsilon that shows differentiation (not too strict)
    ranking_eps = 0.05  # Requires similarity ≥ 0.95
    # Fall back to next available if 0.05 not in results
    if ranking_eps not in sorted_epsilons:
        ranking_eps = sorted_epsilons[-1] if len(sorted_epsilons) > 0 else 0.1
    
    all_scores = train_scores + test_scores
    all_scores.sort(key=lambda x: x[1].get(ranking_eps, 999))  # Sort by epsilon-rank
    
    print(f"\n{'Rank':<6s} {'Task':<30s} {'Type':<6s} ε={ranking_eps:.2f} {'Best Sim':>10s}")
    print("-" * 65)
    
    for i, (task_name, epsilon_ranks, best_sim, task_type) in enumerate(all_scores, 1):
        rank_val = epsilon_ranks.get(ranking_eps, '-')
        print(f"{i:<6d} {task_name:<30s} {task_type:<6s} {str(rank_val):>7s} {best_sim:>10.4f}")
    
    # Test task detailed reconstruction summary
    if len(test_scores) > 0:
        print("\n--- Test Task Reconstruction Summary (All Epsilon Levels) ---")
        print("(Epsilon = max allowed cosine distance; lower ε = stricter threshold)")
        
        # Build header with all epsilon levels
        header = f"{'Task':<30s}"
        for eps in sorted_epsilons:
            header += f" {f'ε={eps:.2f}':>7s}"
        header += f" {'Best Sim':>10s}"
        print(header)
        print("-" * len(header))
        
        # Sort test tasks by best similarity (descending)
        test_scores_sorted = sorted(test_scores, key=lambda x: x[2], reverse=True)
        
        for task_name, epsilon_ranks, best_sim, _ in test_scores_sorted:
            row = f"{task_name:<30s}"
            for eps in sorted_epsilons:
                rank = epsilon_ranks.get(eps, '-')
                row += f" {str(rank):>7s}"
            row += f" {best_sim:>10.4f}"
            print(row)
        
        print(f"\nInterpretation:")
        print(f"  - ε-rank: Number of basis components needed to achieve similarity ≥ (1 - ε)")
        print(f"  - Lower ε-rank at given threshold = simpler/more fundamental task")
        print(f"  - Best Sim: Maximum achievable similarity with all {basis.U.shape[1]} components")
    
    # Similarity patterns
    print("\n--- Test Task Similarities (Top 5 Pairs) ---")
    n_test = len(test_vecs)
    if n_test > 1:
        pairs = []
        for i in range(n_test):
            for j in range(i+1, n_test):
                sim = similarity_matrix[i, j]
                pairs.append((test_vecs[i].task_name, test_vecs[j].task_name, sim))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        for name1, name2, sim in pairs[:5]:
            print(f"  {name1:20s} ↔ {name2:20s}: {sim:.4f}")
    
    print("\n" + "="*70)


def interpret_principal_components(
    basis,
    train_fvs: List[TaskFunctionVec],
    test_fvs: List[TaskFunctionVec],
    n_components: int = 10,
    top_k: int = 5
):
    """
    Interpret principal components by analyzing task loadings.
    
    Shows which tasks load most heavily on each PC and identifies patterns
    that may reveal what each dimension captures (e.g., case transformation,
    reversal, translation, etc.)
    """
    from collections import defaultdict
    
    print("\n" + "=" * 70)
    print("PRINCIPAL COMPONENT INTERPRETATION")
    print("=" * 70)
    
    # Combine all FVs
    all_fvs = {fv.task_name: fv for fv in train_fvs + test_fvs}
    task_names = list(all_fvs.keys())
    n_tasks = len(task_names)
    n_components = min(n_components, basis.U.shape[1], n_tasks)
    
    print(f"\nAnalyzing {n_components} principal components across {n_tasks} tasks")
    
    # Compute loadings: project each FV onto each PC
    loadings = np.zeros((n_tasks, n_components))
    for i, task_name in enumerate(task_names):
        fv = all_fvs[task_name].function_vec
        for j in range(n_components):
            pc = basis.U[:, j]
            loadings[i, j] = np.dot(fv, pc)
    
    # Helper to categorize tasks
    def get_category(task_name):
        t = task_name.lower()
        if any(x in t for x in ['upper', 'lower', 'reverse', 'first_letter', 'copying']):
            return 'string_manip'
        if 'translate' in t:
            return 'translation'
        if any(x in t for x in ['plural', 'singular', 'gerund']):
            return 'grammatical'
        if any(x in t for x in ['capital', 'currency']):
            return 'factual'
        if 'arithmetic' in t:
            return 'arithmetic'
        if 'textfrct' in t:
            # Sub-categorize textfrct
            if ':v' in t:
                return 'textfrct:vocab'
            elif ':rg' in t:
                return 'textfrct:reading'
            elif ':rl' in t:
                return 'textfrct:reasoning'
            elif ':ma' in t:
                return 'textfrct:math'
            elif ':cv' in t:
                return 'textfrct:comprehension'
            elif ':i' in t:
                return 'textfrct:inference'
            return 'textfrct:other'
        if 'compositional' in t:
            return 'compositional'
        return 'other'
    
    # Print explained variance summary
    total_var = np.sum(basis.S ** 2)
    cumulative_var = 0
    print(f"\nExplained variance by component:")
    for pc_idx in range(min(10, n_components)):
        var = (basis.S[pc_idx] ** 2) / total_var * 100
        cumulative_var += var
        print(f"  PC{pc_idx + 1:2d}: {var:5.2f}% (cumulative: {cumulative_var:5.2f}%)")
    
    # Analyze each PC
    for pc_idx in range(n_components):
        pc_loadings = loadings[:, pc_idx]
        explained_var = (basis.S[pc_idx] ** 2) / total_var * 100
        
        # Sort by loading
        sorted_idx = np.argsort(pc_loadings)
        
        print(f"\n{'─' * 70}")
        print(f"PC {pc_idx + 1} (explains {explained_var:.2f}% variance)")
        print(f"{'─' * 70}")
        
        print(f"\n  HIGH loadings (tasks that activate this component):")
        for i in sorted_idx[-top_k:][::-1]:
            cat = get_category(task_names[i])
            print(f"    {pc_loadings[i]:+.4f}  {task_names[i]:<40} [{cat}]")
        
        print(f"\n  LOW loadings (tasks anti-correlated with this component):")
        for i in sorted_idx[:top_k]:
            cat = get_category(task_names[i])
            print(f"    {pc_loadings[i]:+.4f}  {task_names[i]:<40} [{cat}]")
        
        # Category means
        cat_loadings = defaultdict(list)
        for i, t in enumerate(task_names):
            cat_loadings[get_category(t)].append(pc_loadings[i])
        
        cat_means = {cat: np.mean(vals) for cat, vals in cat_loadings.items()}
        cat_means_sorted = sorted(cat_means.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n  Category mean loadings (sorted by |loading|):")
        for cat, mean in cat_means_sorted[:7]:  # Top 7 categories
            n_tasks_cat = len(cat_loadings[cat])
            print(f"    {mean:+.4f}  {cat:<25} (n={n_tasks_cat})")
        
        # Try to interpret based on patterns
        pos_tasks = [task_names[i] for i in sorted_idx[-top_k:]]
        neg_tasks = [task_names[i] for i in sorted_idx[:top_k]]
        
        interpretations = []
        
        # Check for upper/lower contrast
        upper_pos = sum(1 for t in pos_tasks if 'upper' in t.lower())
        lower_pos = sum(1 for t in pos_tasks if 'lower' in t.lower())
        upper_neg = sum(1 for t in neg_tasks if 'upper' in t.lower())
        lower_neg = sum(1 for t in neg_tasks if 'lower' in t.lower())
        
        if upper_pos > 1 and lower_neg > 1:
            interpretations.append("uppercase ↔ lowercase")
        elif lower_pos > 1 and upper_neg > 1:
            interpretations.append("lowercase ↔ uppercase")
        
        # Check for reverse
        reverse_pos = sum(1 for t in pos_tasks if 'reverse' in t.lower())
        reverse_neg = sum(1 for t in neg_tasks if 'reverse' in t.lower())
        if reverse_pos > 1 and reverse_neg == 0:
            interpretations.append("+ reversal operations")
        elif reverse_neg > 1 and reverse_pos == 0:
            interpretations.append("- reversal operations")
        
        # Check for translation
        trans_pos = sum(1 for t in pos_tasks if 'translate' in t.lower())
        trans_neg = sum(1 for t in neg_tasks if 'translate' in t.lower())
        if trans_pos > 1:
            interpretations.append("+ translation tasks")
        elif trans_neg > 1:
            interpretations.append("- translation tasks")
        
        # Check for grammatical
        gram_pos = sum(1 for t in pos_tasks if any(x in t.lower() for x in ['plural', 'gerund']))
        gram_neg = sum(1 for t in neg_tasks if any(x in t.lower() for x in ['plural', 'gerund']))
        if gram_pos > 1:
            interpretations.append("+ grammatical transforms")
        elif gram_neg > 1:
            interpretations.append("- grammatical transforms")
        
        # Check for factual
        fact_pos = sum(1 for t in pos_tasks if any(x in t.lower() for x in ['capital', 'currency']))
        fact_neg = sum(1 for t in neg_tasks if any(x in t.lower() for x in ['capital', 'currency']))
        if fact_pos > 1:
            interpretations.append("+ factual lookup")
        elif fact_neg > 1:
            interpretations.append("- factual lookup")
        
        # Check for textfrct
        tf_pos = sum(1 for t in pos_tasks if 'textfrct' in t.lower())
        tf_neg = sum(1 for t in neg_tasks if 'textfrct' in t.lower())
        if tf_pos >= 3:
            interpretations.append("+ textfrct tasks")
        elif tf_neg >= 3:
            interpretations.append("- textfrct tasks")
        
        # Check for first/last
        first_pos = sum(1 for t in pos_tasks if 'first' in t.lower())
        last_pos = sum(1 for t in pos_tasks if 'last' in t.lower())
        first_neg = sum(1 for t in neg_tasks if 'first' in t.lower())
        last_neg = sum(1 for t in neg_tasks if 'last' in t.lower())
        if first_pos > 1 and last_neg > 1:
            interpretations.append("first ↔ last position")
        elif last_pos > 1 and first_neg > 1:
            interpretations.append("last ↔ first position")
        
        if interpretations:
            print(f"\n  💡 Interpretation: {'; '.join(interpretations)}")
        else:
            print(f"\n  💡 Interpretation: (no clear pattern detected)")
    
    # Print correlation matrix between task features and PCs
    print(f"\n{'=' * 70}")
    print("PC-FEATURE CORRELATIONS")
    print(f"{'=' * 70}")
    
    # Create binary features
    features = {
        'has_upper': np.array([1 if 'upper' in t.lower() else 0 for t in task_names]),
        'has_lower': np.array([1 if 'lower' in t.lower() else 0 for t in task_names]),
        'has_reverse': np.array([1 if 'reverse' in t.lower() else 0 for t in task_names]),
        'is_translation': np.array([1 if 'translate' in t.lower() else 0 for t in task_names]),
        'is_factual': np.array([1 if any(x in t.lower() for x in ['capital', 'currency']) else 0 for t in task_names]),
        'is_grammatical': np.array([1 if any(x in t.lower() for x in ['plural', 'gerund']) else 0 for t in task_names]),
        'is_textfrct': np.array([1 if 'textfrct' in t.lower() else 0 for t in task_names]),
        'is_simple_icl': np.array([1 if 'simple_icl' in t.lower() else 0 for t in task_names]),
        'is_compositional': np.array([1 if 'compositional' in t.lower() else 0 for t in task_names]),
    }
    
    # Print header
    header = f"{'Feature':<18}"
    for i in range(min(8, n_components)):
        header += f" PC{i+1:>5}"
    print(header)
    print("-" * len(header))
    
    for feat_name, feat_vals in features.items():
        if feat_vals.sum() == 0 or feat_vals.sum() == len(feat_vals):
            continue  # Skip if no variance
        
        row = f"{feat_name:<18}"
        for pc_idx in range(min(8, n_components)):
            corr = np.corrcoef(feat_vals, loadings[:, pc_idx])[0, 1]
            # Highlight strong correlations
            if abs(corr) > 0.5:
                row += f" {corr:+.2f}*"
            else:
                row += f" {corr:+.2f} "
        print(row)
    
    print("\n(* indicates |correlation| > 0.5)")
    
    return loadings, task_names


def visualize_task_projections(
    train_tasks: List[BaseTask],
    test_tasks: List[BaseTask],
    basis,
    output_dir: Path,
    model_name: str = "model"
):
    """Create visualizations of task projections in skill basis space.
    
    Uses SVD-based skill basis (uncentered decomposition of L2-normalized function vectors).
    Similar to the visualizations in test_basic_icl_tasks.py but for real tasks.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("\n⚠️  matplotlib not available - skipping visualization")
        return
    
    # Try to import adjustText for better label positioning
    try:
        from adjustText import adjust_text
        use_adjust_text = True
        print("✓ Using adjustText for cleaner label positioning")
    except ImportError:
        use_adjust_text = False
        print("⚠️  adjustText not available - labels may overlap. Install with: pip install adjustText")
    
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Get task projections (coefficients in the skill basis)
    # basis.Vt is (k, n_tasks) where n_tasks are the training tasks that were successfully extracted
    projections = basis.Vt.T  # (n_tasks, k)
    
    # Use task names from the basis (these are the tasks that were actually extracted)
    # This handles the case where some tasks were skipped due to no correct instances
    task_names = basis.task_names
    
    # Get explained variance ratios
    var_ratios = basis.explained_variance_ratio()
    
    print(f"\nProjections shape: {projections.shape}")
    print(f"Number of tasks: {len(task_names)}")
    print(f"Number of basis components: {basis.U.shape[1]}")
    print(f"Using uncentered SVD decomposition on L2-normalized function vectors")
    
    # Ensure we have at least 3 components
    if projections.shape[1] < 3:
        print("⚠️  Need at least 3 skill basis components for visualization")
        return
    
    # Create output directory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # ========== 2D Plots ==========
    print("\n📊 Creating 2D projection plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Component 1 vs Component 2
    ax = axes[0, 0]
    scatter = ax.scatter(projections[:, 0], projections[:, 1], s=100, alpha=0.7, c='steelblue')
    texts = []
    for i, name in enumerate(task_names):
        texts.append(ax.text(projections[i, 0], projections[i, 1], name, 
                            fontsize=7, alpha=0.8))
    if use_adjust_text:
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.5), ax=ax)
    ax.set_xlabel(f'Basis Component 1 ({var_ratios[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'Basis Component 2 ({var_ratios[1]:.1%} variance)', fontsize=12)
    ax.set_title('Task Projections: Component 1 vs 2', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
    
    # Component 1 vs Component 3
    ax = axes[0, 1]
    scatter = ax.scatter(projections[:, 0], projections[:, 2], s=100, alpha=0.7, c='darkorange')
    texts = []
    for i, name in enumerate(task_names):
        texts.append(ax.text(projections[i, 0], projections[i, 2], name, 
                            fontsize=7, alpha=0.8))
    if use_adjust_text:
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.5), ax=ax)
    ax.set_xlabel(f'Basis Component 1 ({var_ratios[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'Basis Component 3 ({var_ratios[2]:.1%} variance)', fontsize=12)
    ax.set_title('Task Projections: Component 1 vs 3', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
    
    # Component 2 vs Component 3
    ax = axes[1, 0]
    scatter = ax.scatter(projections[:, 1], projections[:, 2], s=100, alpha=0.7, c='forestgreen')
    texts = []
    for i, name in enumerate(task_names):
        texts.append(ax.text(projections[i, 1], projections[i, 2], name, 
                            fontsize=7, alpha=0.8))
    if use_adjust_text:
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.5), ax=ax)
    ax.set_xlabel(f'Basis Component 2 ({var_ratios[1]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'Basis Component 3 ({var_ratios[2]:.1%} variance)', fontsize=12)
    ax.set_title('Task Projections: Component 2 vs 3', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
    
    # Explained variance plot
    ax = axes[1, 1]
    k = min(20, len(var_ratios))
    ax.bar(range(1, k+1), var_ratios[:k], alpha=0.7, color='purple')
    ax.plot(range(1, k+1), np.cumsum(var_ratios[:k]), 'ro-', linewidth=2, markersize=6, label='Cumulative')
    ax.set_xlabel('Skill Basis Component', fontsize=12)
    ax.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax.set_title('Explained Variance by Component', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(range(1, k+1, max(1, k//10)))
    
    plt.tight_layout()
    plot_2d_path = viz_dir / f"task_basis_2d_{model_name}.png"
    plt.savefig(plot_2d_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved 2D skill basis plots to: {plot_2d_path}")
    plt.close()
    
    # ========== 3D Plot ==========
    print("📊 Creating 3D projection plot...")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Categorize tasks by cognitive/computational function
    def categorize_task(name):
        name_lower = name.lower()
        
        # 1. Arithmetic/Mathematical
        if any(kw in name_lower for kw in ['arithmetic', 'add_one', 'math']):
            return 'arithmetic', '#e74c3c'  # red
        
        # 2. String Manipulation (algorithmic, no semantics)
        if any(kw in name_lower for kw in [
            'capitalization', 'uppercase', 'lowercase',
            'first_letter', 'last_letter', 'first_character', 'last_character',
            'reverse', 'reversal', 'string_length', 'vowel_count',
            'scrambled_words', 'incomplete_words', 'hidden_words'
        ]):
            return 'string_manipulation', '#3498db'  # blue
        
        # 3. Factual Lookup/Memorization
        if any(kw in name_lower for kw in [
            'translate', 'country_to_capital', 'country_to_currency'
        ]):
            return 'factual_lookup', '#2ecc71'  # green
        
        # 4. Grammatical (morphology, POS, linguistic rules)
        if any(kw in name_lower for kw in [
            'singular_to_plural', 'present_to_gerund', 'part_of_speech'
        ]):
            return 'grammatical', '#9b59b6'  # purple
        
        # 5. Semantic Relations (meaning-based reasoning)
        if any(kw in name_lower for kw in [
            'opposites', 'nonsense_syllogisms', 'inference', 
            'analogy', 'rhyming', 'deciphering_languages'
        ]):
            return 'semantic_reasoning', '#f39c12'  # orange
        
        # 6. Vocabulary/Word Knowledge
        if any(kw in name_lower for kw in [
            'vocabulary_test', 'controlled_association', 
            'first_and_last_name', 'objest-number'
        ]):
            return 'vocabulary', '#1abc9c'  # teal
        
        # 7. Pattern Recognition
        if any(kw in name_lower for kw in [
            'letter_sets', 'locations_test', 'copying'
        ]):
            return 'pattern_recognition', '#e67e22'  # dark orange
        
        # 8. Meta-Cognitive (attention, context manipulation)
        if any(kw in name_lower for kw in [
            'ignoring_context', 'ioi_task'
        ]):
            return 'meta_cognitive', '#34495e'  # dark gray
        
        # Default
        return 'other', '#95a5a6'  # light gray
    
    task_colors = []
    task_categories = []
    for name in task_names:
        cat, color = categorize_task(name)
        task_categories.append(cat)
        task_colors.append(color)
    
    scatter = ax.scatter(projections[:, 0], projections[:, 1], projections[:, 2],
                        c=task_colors, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # For 3D plots, adjustText doesn't work well, so we use conditional labeling
    # Only label tasks that are well-separated (avoid clutter)
    if use_adjust_text:
        # Label every task but with smaller font
        for i, name in enumerate(task_names):
            ax.text(projections[i, 0], projections[i, 1], projections[i, 2],
                   name, fontsize=6, alpha=0.7)
    else:
        # Label all tasks
        for i, name in enumerate(task_names):
            ax.text(projections[i, 0], projections[i, 1], projections[i, 2],
                   name, fontsize=7, alpha=0.8)
    
    ax.set_xlabel(f'Basis Component 1 ({var_ratios[0]:.1%})', fontsize=12, labelpad=10)
    ax.set_ylabel(f'Basis Component 2 ({var_ratios[1]:.1%})', fontsize=12, labelpad=10)
    ax.set_zlabel(f'Basis Component 3 ({var_ratios[2]:.1%})', fontsize=12, labelpad=10)
    ax.set_title('Task Projections in 3D Skill Basis Space', fontsize=16, fontweight='bold', pad=20)
    
    # Add color legend with refined categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Arithmetic'),
        Patch(facecolor='#3498db', label='String Manipulation'),
        Patch(facecolor='#2ecc71', label='Factual Lookup'),
        Patch(facecolor='#9b59b6', label='Grammatical'),
        Patch(facecolor='#f39c12', label='Semantic Reasoning'),
        Patch(facecolor='#1abc9c', label='Vocabulary'),
        Patch(facecolor='#e67e22', label='Pattern Recognition'),
        Patch(facecolor='#34495e', label='Meta-Cognitive'),
        Patch(facecolor='#95a5a6', label='Other')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    plot_3d_path = viz_dir / f"task_basis_3d_{model_name}.png"
    plt.savefig(plot_3d_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved 3D skill basis plot to: {plot_3d_path}")
    plt.close()
    
    # Note about interactivity
    if not use_adjust_text:
        print("\n💡 Tip: For interactive rotatable 3D plots, consider using plotly")
        print("   (We can add this in the future for better exploration)")
    
    # ========== Task Category Analysis ==========
    print("\n📊 Analyzing task categories...")
    
    # Group tasks by refined cognitive/computational categories
    categories = {}
    for i, name in enumerate(task_names):
        cat, _ = categorize_task(name)
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(i)
    
    print(f"\nTask categories found: {list(categories.keys())}")
    print(f"\nTasks per category:")
    for cat in sorted(categories.keys()):
        print(f"  {cat:25s}: {len(categories[cat]):2d} tasks")
    
    # Compute centroids and distances
    print("\nCategory centroids in skill basis space:")
    centroids = {}
    for cat, indices in categories.items():
        if len(indices) > 0:
            centroid = projections[indices].mean(axis=0)
            centroids[cat] = centroid
            print(f"  {cat:25s}: [{centroid[0]:7.3f}, {centroid[1]:7.3f}, {centroid[2]:7.3f}] ({len(indices)} tasks)")
    
    print("\nPairwise centroid distances (in skill basis space):")
    cat_names = sorted(centroids.keys())
    for i, cat1 in enumerate(cat_names):
        for cat2 in cat_names[i+1:]:
            dist = np.linalg.norm(centroids[cat1] - centroids[cat2])
            print(f"  {cat1:25s} ↔ {cat2:25s}: {dist:.3f}")
    
    print(f"\n✓ Visualizations saved to: {viz_dir}")
    print("="*70)


def visualize_task_projections_interactive(
    train_tasks: List[BaseTask],
    test_tasks: List[BaseTask],
    basis,
    output_dir: Path,
    model_name: str = "model",
    performance: Dict[str, float] = None,
    color_by: str = "both"
):
    """Create interactive plotly visualization of task projections.
    
    Args:
        train_tasks: Training tasks used to build basis
        test_tasks: Held-out test tasks
        basis: Skill basis from SVD
        output_dir: Output directory for HTML files
        model_name: Model name for filenames
        performance: Optional dict mapping task_name -> accuracy (0.0-1.0)
        color_by: "category", "performance", or "both" (creates separate plots)
    
    This is called after matplotlib plots are generated to avoid blocking
    the main visualizations if plotly is not available.
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
    except ImportError:
        print("\n⚠️  plotly not available - skipping interactive visualization")
        print("   Install with: pip install plotly")
        return
    
    print("\n" + "="*70)
    print("CREATING INTERACTIVE PLOTLY VISUALIZATION")
    print("="*70)
    
    # Get task projections
    projections = basis.Vt.T  # (n_tasks, k)
    
    # Use task names from the basis (these are the tasks that were actually extracted)
    task_names = basis.task_names
    var_ratios = basis.explained_variance_ratio()
    
    # Categorize tasks
    def categorize_task(name):
        name_lower = name.lower()
        
        if any(kw in name_lower for kw in ['arithmetic', 'add_one', 'math']):
            return 'Arithmetic', '#e74c3c'
        if any(kw in name_lower for kw in [
            'capitalization', 'uppercase', 'lowercase',
            'first_letter', 'last_letter', 'first_character', 'last_character',
            'reverse', 'reversal', 'string_length', 'vowel_count',
            'scrambled_words', 'incomplete_words', 'hidden_words'
        ]):
            return 'String Manipulation', '#3498db'
        if any(kw in name_lower for kw in [
            'translate', 'country_to_capital', 'country_to_currency'
        ]):
            return 'Factual Lookup', '#2ecc71'
        if any(kw in name_lower for kw in [
            'singular_to_plural', 'present_to_gerund', 'part_of_speech'
        ]):
            return 'Grammatical', '#9b59b6'
        if any(kw in name_lower for kw in [
            'opposites', 'nonsense_syllogisms', 'inference', 
            'analogy', 'rhyming', 'deciphering_languages'
        ]):
            return 'Semantic Reasoning', '#f39c12'
        if any(kw in name_lower for kw in [
            'vocabulary_test', 'controlled_association', 
            'first_and_last_name', 'objest-number'
        ]):
            return 'Vocabulary', '#1abc9c'
        if any(kw in name_lower for kw in [
            'letter_sets', 'locations_test', 'copying'
        ]):
            return 'Pattern Recognition', '#e67e22'
        if any(kw in name_lower for kw in [
            'ignoring_context', 'ioi_task'
        ]):
            return 'Meta-Cognitive', '#34495e'
        
        # Default
        return 'Other', '#95a5a6'
    
    # Build data for plotly
    categories = []
    cat_colors = []
    for name in task_names:
        cat, color = categorize_task(name)
        categories.append(cat)
        cat_colors.append(color)
    
    # Get performance values if available
    perf_values = []
    has_performance = performance is not None and len(performance) > 0
    if has_performance:
        for name in task_names:
            perf_values.append(performance.get(name, np.nan))
        perf_values = np.array(perf_values)
        n_with_perf = np.sum(~np.isnan(perf_values))
        print(f"  Performance data available for {n_with_perf}/{len(task_names)} tasks")
    
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Helper to create a 3D plot
    def create_3d_plot(colors, color_mode, title_suffix, colorbar_title=None):
        if color_mode == "performance":
            marker_dict = dict(
                size=10,
                color=colors,
                colorscale='RdYlGn',  # Red (low) to Green (high)
                cmin=0,
                cmax=1,
                colorbar=dict(title=colorbar_title or 'Accuracy', x=1.02),
                opacity=0.8,
                line=dict(color='black', width=0.5)
            )
        else:
            marker_dict = dict(
                size=8,
                color=colors,
                opacity=0.8,
                line=dict(color='black', width=0.5)
            )
        
        # Build hover text with performance info
        hover_texts = []
        for i, name in enumerate(task_names):
            text = f'<b>{name}</b><br>Category: {categories[i]}'
            if has_performance and not np.isnan(perf_values[i]):
                text += f'<br>Accuracy: {perf_values[i]:.1%}'
            hover_texts.append(text)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=projections[:, 0],
            y=projections[:, 1],
            z=projections[:, 2],
            mode='markers+text',
            marker=marker_dict,
            text=task_names,
            textposition='top center',
            textfont=dict(size=8),
            customdata=hover_texts,
            hovertemplate='%{customdata}<br>C1: %{x:.3f}<br>C2: %{y:.3f}<br>C3: %{z:.3f}<extra></extra>',
            name='Tasks'
        )])
        
        fig.update_layout(
            title=dict(
                text=f'Task Projections in 3D Skill Basis Space {title_suffix}',
                font=dict(size=18, family='Arial, sans-serif'),
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(
                    title=f'Basis Component 1 ({var_ratios[0]:.1%} variance)',
                    backgroundcolor='rgb(230, 230, 230)',
                    gridcolor='white',
                    showbackground=True
                ),
                zaxis=dict(
                    title=f'Basis Component 3 ({var_ratios[2]:.1%} variance)',
                    backgroundcolor='rgb(230, 230, 230)',
                    gridcolor='white',
                    showbackground=True
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            width=1200,
            height=900,
            showlegend=False,
            hovermode='closest',
            font=dict(family='Arial, sans-serif', size=12)
        )
        
        # Add category legend for category-colored plots
        if color_mode == "category":
            legend_text = "<b>Categories:</b><br>"
            unique_cats = {}
            for cat, color in zip(categories, cat_colors):
                if cat not in unique_cats:
                    unique_cats[cat] = color
            for cat, color in sorted(unique_cats.items()):
                legend_text += f'<span style="color:{color}">■</span> {cat}<br>'
            
            fig.add_annotation(
                text=legend_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor='left', yanchor='top',
                showarrow=False,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=10)
            )
        
        return fig
    
    # Create category-colored plot
    if color_by in ["category", "both"]:
        fig_cat = create_3d_plot(cat_colors, "category", "(by Category)")
        html_path = viz_dir / f"task_basis_3d_interactive_{model_name}.html"
        fig_cat.write_html(str(html_path))
        print(f"✓ Saved interactive 3D plot (by category) to: {html_path}")
    
    # Create performance-colored plot if data available
    if has_performance and color_by in ["performance", "both"]:
        # Replace NaN with 0.5 for display (gray in RdYlGn)
        perf_display = np.where(np.isnan(perf_values), 0.5, perf_values)
        fig_perf = create_3d_plot(perf_display.tolist(), "performance", "(by Accuracy)", "Accuracy")
        html_path_perf = viz_dir / f"task_basis_3d_performance_{model_name}.html"
        fig_perf.write_html(str(html_path_perf))
        print(f"✓ Saved interactive 3D plot (by performance) to: {html_path_perf}")
        
        # Also analyze performance clustering
        analyze_performance_clustering(projections, perf_values, task_names, categories)
    elif color_by == "performance" and not has_performance:
        print("  ⚠️  No performance data available - skipping performance-colored plot")
    
    # Create 2D interactive plots
    print("\n📊 Creating interactive 2D plots...")
    
    # Helper to create 2D plot
    def create_2d_plots(colors, color_mode):
        fig_2d = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                f'Component 1 vs 2',
                f'Component 1 vs 3',
                f'Component 2 vs 3'
            ),
            horizontal_spacing=0.1
        )
        
        marker_dict = dict(size=10, color=colors, opacity=0.8, line=dict(color='black', width=0.5))
        if color_mode == "performance":
            marker_dict.update(colorscale='RdYlGn', cmin=0, cmax=1)
        
        # Plot 1: Component 1 vs 2
        fig_2d.add_trace(
            go.Scatter(
                x=projections[:, 0],
                y=projections[:, 1],
                mode='markers+text',
                marker=marker_dict,
                text=task_names,
                textposition='top center',
                textfont=dict(size=7),
                hovertemplate='<b>%{text}</b><br>C1: %{x:.3f}<br>C2: %{y:.3f}<extra></extra>',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Plot 2: Component 1 vs 3
        fig_2d.add_trace(
            go.Scatter(
                x=projections[:, 0],
                y=projections[:, 2],
                mode='markers+text',
                marker=marker_dict,
                text=task_names,
                textposition='top center',
                textfont=dict(size=7),
                hovertemplate='<b>%{text}</b><br>C1: %{x:.3f}<br>C3: %{y:.3f}<extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Plot 3: Component 2 vs 3
        fig_2d.add_trace(
            go.Scatter(
                x=projections[:, 1],
                y=projections[:, 2],
                mode='markers+text',
                marker=marker_dict,
                text=task_names,
                textposition='top center',
                textfont=dict(size=7),
                hovertemplate='<b>%{text}</b><br>C2: %{x:.3f}<br>C3: %{y:.3f}<extra></extra>',
                showlegend=False
            ),
            row=1, col=3
        )
        
        # Update axes labels
        fig_2d.update_xaxes(title_text=f'Component 1 ({var_ratios[0]:.1%})', row=1, col=1)
        fig_2d.update_yaxes(title_text=f'Component 2 ({var_ratios[1]:.1%})', row=1, col=1)
        fig_2d.update_xaxes(title_text=f'Component 1 ({var_ratios[0]:.1%})', row=1, col=2)
        fig_2d.update_yaxes(title_text=f'Component 3 ({var_ratios[2]:.1%})', row=1, col=2)
        fig_2d.update_xaxes(title_text=f'Component 2 ({var_ratios[1]:.1%})', row=1, col=3)
        fig_2d.update_yaxes(title_text=f'Component 3 ({var_ratios[2]:.1%})', row=1, col=3)
        
        suffix = "(by Category)" if color_mode == "category" else "(by Accuracy)"
        fig_2d.update_layout(
            title_text=f'Task Projections in Skill Basis Space - 2D Views {suffix}',
            height=500,
            width=1800,
            showlegend=False,
            font=dict(family='Arial, sans-serif', size=11)
        )
        
        return fig_2d
    
    # Create 2D plots
    if color_by in ["category", "both"]:
        fig_2d = create_2d_plots(cat_colors, "category")
        html_2d_path = viz_dir / f"task_basis_2d_interactive_{model_name}.html"
        fig_2d.write_html(str(html_2d_path))
        print(f"✓ Saved interactive 2D plots (by category) to: {html_2d_path}")
    
    if has_performance and color_by in ["performance", "both"]:
        perf_display = np.where(np.isnan(perf_values), 0.5, perf_values)
        fig_2d_perf = create_2d_plots(perf_display.tolist(), "performance")
        html_2d_perf_path = viz_dir / f"task_basis_2d_performance_{model_name}.html"
        fig_2d_perf.write_html(str(html_2d_perf_path))
        print(f"✓ Saved interactive 2D plots (by performance) to: {html_2d_perf_path}")
    
    print("\n" + "="*70)
    print("✓ Interactive visualizations complete!")
    print("="*70)
    print("\n💡 Tips for interactive plots:")
    print("  - Hover over points to see task names and coordinates")
    print("  - Drag to rotate (3D) or pan (2D)")
    print("  - Scroll to zoom in/out")
    print("  - Double-click to reset view")
    print("  - Click and drag to select region for zoom")


def analyze_performance_clustering(
    projections: np.ndarray,
    perf_values: np.ndarray,
    task_names: List[str],
    categories: List[str],
    threshold: float = 0.5
):
    """Analyze if hard tasks cluster in FV space."""
    from scipy import stats
    
    print("\n" + "=" * 70)
    print("PERFORMANCE CLUSTERING ANALYSIS")
    print("=" * 70)
    
    valid_mask = ~np.isnan(perf_values)
    n_valid = valid_mask.sum()
    
    if n_valid < 5:
        print(f"  Not enough tasks with performance data ({n_valid}/5 required)")
        return
    
    hard_mask = (perf_values < threshold) & valid_mask
    easy_mask = (perf_values >= threshold) & valid_mask
    
    print(f"\nDifficulty split (threshold: {threshold:.0%}):")
    print(f"  Hard tasks (< {threshold:.0%}): {hard_mask.sum()}")
    print(f"  Easy tasks (≥ {threshold:.0%}): {easy_mask.sum()}")
    
    if hard_mask.sum() > 1 and easy_mask.sum() > 1:
        # Compute within-group distances
        hard_points = projections[hard_mask, :3]  # Use first 3 PCs
        easy_points = projections[easy_mask, :3]
        
        hard_dists = []
        for i in range(len(hard_points)):
            for j in range(i+1, len(hard_points)):
                hard_dists.append(np.linalg.norm(hard_points[i] - hard_points[j]))
        
        easy_dists = []
        for i in range(len(easy_points)):
            for j in range(i+1, len(easy_points)):
                easy_dists.append(np.linalg.norm(easy_points[i] - easy_points[j]))
        
        cross_dists = []
        for hp in hard_points:
            for ep in easy_points:
                cross_dists.append(np.linalg.norm(hp - ep))
        
        print(f"\nDistance analysis (in PC1-3 space):")
        print(f"  Avg distance within hard tasks: {np.mean(hard_dists):.4f}")
        print(f"  Avg distance within easy tasks: {np.mean(easy_dists):.4f}")
        print(f"  Avg distance between hard/easy: {np.mean(cross_dists):.4f}")
        
        if np.mean(hard_dists) < np.mean(cross_dists) * 0.7:
            print("\n  ✓ Hard tasks appear to CLUSTER together!")
        else:
            print("\n  ✗ Hard tasks do NOT strongly cluster together")
    
    # Correlation with PC dimensions
    print(f"\nCorrelation of accuracy with PC dimensions:")
    for i in range(min(5, projections.shape[1])):
        corr, pval = stats.pearsonr(projections[valid_mask, i], perf_values[valid_mask])
        sig = "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  PC{i+1} vs Accuracy: r={corr:+.3f} (p={pval:.3f}){sig}")
    
    # Hardest tasks
    print(f"\nHardest 5 tasks:")
    sorted_indices = np.argsort(perf_values)
    for idx in sorted_indices[:5]:
        if not np.isnan(perf_values[idx]):
            print(f"  {perf_values[idx]:5.1%}  {task_names[idx]:<40} [{categories[idx]}]")
    
    # Easiest tasks
    print(f"\nEasiest 5 tasks:")
    for idx in sorted_indices[-5:][::-1]:
        if not np.isnan(perf_values[idx]):
            print(f"  {perf_values[idx]:5.1%}  {task_names[idx]:<40} [{categories[idx]}]")
    
    # Category breakdown
    print(f"\nAccuracy by category:")
    from collections import defaultdict
    cat_perfs = defaultdict(list)
    for i, cat in enumerate(categories):
        if valid_mask[i]:
            cat_perfs[cat].append(perf_values[i])
    
    cat_stats = [(cat, np.mean(perfs), np.std(perfs), len(perfs)) 
                 for cat, perfs in cat_perfs.items()]
    cat_stats.sort(key=lambda x: x[1])  # Sort by mean accuracy
    
    for cat, mean, std, n in cat_stats:
        print(f"  {mean:5.1%} ± {std:4.1%}  {cat:<25} (n={n})")


def main():
    parser = argparse.ArgumentParser(description="Analyze function vectors from real ICL tasks")
    parser.add_argument("--model", type=str, default="distilgpt2",
                       help="Model name (default: distilgpt2)")
    parser.add_argument("--checkpoint", type=str, default="main",
                       help="Model checkpoint/revision (default: main)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (cpu/cuda, default: cpu)")
    parser.add_argument("--layer", type=int, default=5,
                       help="Layer to analyze (default: 5)")
    parser.add_argument("--num-heads", type=int, default=4,
                       help="Number of heads to use (default: 4)")
    parser.add_argument("--num-samples", type=int, default=8,
                       help="Samples per task (default: 8)")
    parser.add_argument("--use-synthetic-tests", action="store_true",
                       help="Use all real tasks for basis, test with synthetic tasks (default: False)")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Ratio of tasks for training (default: 0.7, ignored if --use-synthetic-tests)")
    parser.add_argument("--k-components", type=int, default=None,
                       help="Number of PCA components (default: num_train_tasks)")
    parser.add_argument("--output-dir", type=str, default="real_task_analysis",
                       help="Output directory (default: real_task_analysis)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--epsilons", type=float, nargs="+",
                       default=[0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0.01],
                       help="Epsilon thresholds (default: 0.5 0.3 0.2 0.15 0.1 0.05 0.01)")
    
    # Correct-instance filtering options
    parser.add_argument("--only-correct", action="store_true",
                       help="Only use correct instances for FV extraction (requires --results-dir)")
    parser.add_argument("--results-dir", type=str, default=None,
                       help="Path to evaluation results dir for correct-instance filtering "
                            "(e.g., results/olmo2_continuous_1b_early_revised)")
    
    # Visualization options
    parser.add_argument("--color-by", type=str, default="both",
                       choices=["category", "performance", "both"],
                       help="How to color tasks in visualizations (default: both)")
    
    args = parser.parse_args()
    
    # Validate filtering args
    if args.only_correct and not args.results_dir:
        print("⚠️  Warning: --only-correct requires --results-dir. Disabling filtering.")
        args.only_correct = False
    
    # Validate filtering args
    if args.only_correct and not args.results_dir:
        print("⚠️  Warning: --only-correct requires --results-dir. Disabling filtering.")
        args.only_correct = False
    
    print("="*70)
    print("REAL TASK FUNCTION VECTOR ANALYSIS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {args.device}")
    print(f"  Layer: {args.layer}")
    print(f"  Heads: {args.num_heads}")
    print(f"  Samples per task: {args.num_samples}")
    print(f"  Only correct instances: {args.only_correct}")
    if args.only_correct:
        print(f"  Results dir: {args.results_dir}")
    print(f"  Use synthetic tests: {args.use_synthetic_tests}")
    if not args.use_synthetic_tests:
        print(f"  Train ratio: {args.train_ratio}")
    print(f"  Seed: {args.seed}")
    
    # Phase 1: Discover ICL tasks
    icl_tasks = discover_icl_tasks()
    
    if len(icl_tasks) < 2:
        print("\n❌ Need at least 2 ICL tasks to perform analysis!")
        return
    
    # Phase 2: Split into train/test
    print("\n" + "="*70)
    print("SPLITTING TASKS")
    print("="*70)
    
    if args.use_synthetic_tests:
        # Use ALL real tasks for training basis, PLUS some simple ICL test tasks
        print("\nAdding simple ICL test tasks to basis...")
        from tests.test_basic_icl_tasks import (
            SimpleArithmeticTask,
            SimpleNegationTask,
            SimpleCapitalizationTask,
            SimpleRhymingTask,
            FirstCharacterTask,
            LastCharacterTask,
            ReverseStringTask,
            AddOneTask,
            StringLengthTask,
            VowelCountTask,
            #ReverseCapitalizeTask,
        )
        
        # Add non-test simple ICL tasks to training
        # Excluding test tasks (see below)
        train_tasks = icl_tasks + [
            SimpleArithmeticTask(),
            SimpleCapitalizationTask(),  # Component of composite task
            FirstCharacterTask(),
            StringLengthTask(),
        ]
        
        # Test tasks covering different transformation types
        # Held out from training to measure generalization
        test_tasks = [
            # Arithmetic operations
            AddOneTask(),              # Increment: tests numeric transformation
            
            # Semantic operations
            SimpleNegationTask(),      # Semantic opposites: tests word meanings
            
            # String manipulations
            ReverseStringTask(),       # Reverse: tests position manipulation (component of composite)
            LastCharacterTask(),       # Extract last char: tests indexing
            VowelCountTask(),          # Count vowels: tests counting/filtering
            
            # Pattern recognition
            SimpleRhymingTask(),       # Rhyme detection: tests phonetic patterns
            
            # Composite task (should need multiple components)
           #CompositeReverseCapitalizeTask(),  # Reverse + Capitalize: tests composition
        ]
        
        print(f"\nUsing {len(icl_tasks)} real tasks + {len(train_tasks) - len(icl_tasks)} simple ICL tasks for basis training")
        print(f"Testing with {len(test_tasks)} held-out synthetic tasks (including 1 composite)")
    else:
        # Original behavior: split real tasks
        train_tasks, test_tasks = split_tasks(icl_tasks, args.train_ratio, args.seed)
    
    print(f"\nTraining tasks ({len(train_tasks)}):")
    for task in train_tasks:
        print(f"  • {get_task_display_name(task)}")
    
    print(f"\nTest tasks ({len(test_tasks)}):")
    for task in test_tasks:
        print(f"  • {get_task_display_name(task)}")
    
    # Phase 3: Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    
    print(f"\nLoading {args.model} (checkpoint: {args.checkpoint})...")
    # Note: use_fast=False is a workaround for older tokenizers library versions
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.checkpoint, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.checkpoint, trust_remote_code=True).eval()
    if args.device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    print("✓ Model loaded")
    
    # Normalize layer index (support negative indexing)
    from function_vecs.extract_function_vecs import get_blocks, extract_informative_heads
    blocks = get_blocks(model)
    num_layers = len(blocks)
    layer_idx = args.layer
    if layer_idx < 0:
        layer_idx = num_layers + layer_idx
    print(f"Using layer {layer_idx} (out of {num_layers} layers)")
    
    # Setup extraction config
    config = ExtractConfig(
        model_name=args.model,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=4,
        num_samples_per_task=args.num_samples,
        layers=[layer_idx],
        topk_heads=args.num_heads,
        only_correct=args.only_correct,
        results_dir=args.results_dir,
    )
    
    # Phase 3.5: Select informative heads
    print("\n" + "="*70)
    print("SELECTING INFORMATIVE ATTENTION HEADS")
    print("="*70)
    print(f"\nSelecting top-{args.num_heads} most informative heads from layer {layer_idx}...")
    print("(Using AIE metric on training tasks with correct instances)")
    
    # Use all training tasks for screening, require at least 5 to succeed
    headset = extract_informative_heads(
        config, 
        train_tasks, 
        max_screen_tasks=None,  # Try all tasks
        min_screen_tasks=5,     # Need at least 5 successful
    )
    
    print(f"\n✓ Selected {len(headset.heads)} heads:")
    for layer, head in sorted(headset.heads):
        print(f"  Layer {layer}, Head {head}")
    
    # Phase 4: Extract training function vectors
    print("\n" + "="*70)
    print("PHASE 4: EXTRACTING TRAINING FUNCTION VECTORS")
    print("="*70)
    
    train_vecs = extract_function_vectors(
        train_tasks, config, headset, model, tokenizer, desc="training tasks"
    )
    
    if len(train_vecs) < 2:
        print("\n❌ Need at least 2 successful training extractions!")
        return
    
    # Phase 5: Build skill basis
    print("\n" + "="*70)
    print("BUILDING SKILL BASIS")
    print("="*70)
    
    task_matrix = stack_function_vecs(train_vecs)
    print(f"\nTask matrix shape: {task_matrix.V.shape}")
    
    k = args.k_components if args.k_components else len(train_vecs)
    k = min(k, len(train_vecs))  # Can't have more components than tasks
    
    basis = build_skill_basis(task_matrix, method="svd", k=k, center=False)
    print(f"Basis: {basis.U.shape}, k={k}")
    
    # Phase 6: Extract test function vectors
    print("\n" + "="*70)
    print("EXTRACTING TEST FUNCTION VECTORS")
    print("="*70)
    
    test_vecs = extract_function_vectors(
        test_tasks, config, headset, model, tokenizer, desc="test tasks"
    )
    
    if len(test_vecs) == 0:
        print("\n❌ No successful test extractions!")
        return
    
    # Phase 7: Analyze epsilon-ranks for TEST tasks
    print("\n" + "="*70)
    print("ANALYZING TEST TASKS")
    print("="*70)
    test_results = analyze_epsilon_ranks(test_vecs, basis, args.epsilons)
    
    # Phase 7.5: Analyze epsilon-ranks for TRAINING tasks
    print("\n" + "="*70)
    print("ANALYZING TRAINING TASKS (for comparison)")
    print("="*70)
    print("Note: These tasks were used to build the basis")
    train_results = analyze_epsilon_ranks(train_vecs, basis, args.epsilons)
    
    # Combine results
    results = {**test_results, **train_results}
    
    # Phase 8: Compute similarities
    print("\n" + "="*70)
    print("COMPUTING SIMILARITIES")
    print("="*70)
    
    similarity_matrix = compute_similarity_matrix(test_vecs)
    print(f"\nSimilarity matrix shape: {similarity_matrix.shape}")
    
    # Phase 9: Print summary
    print_summary(train_tasks, test_tasks, basis, results, similarity_matrix, test_vecs, train_vecs)
    
    # Phase 9.1: Interpret principal components
    interpret_principal_components(basis, train_vecs, test_vecs, n_components=10, top_k=5)
    
    # Phase 9.2: Load performance data if available
    performance_data = None
    if args.results_dir:
        print("\n" + "="*70)
        print("LOADING PERFORMANCE DATA")
        print("="*70)
        performance_data = load_task_performance(args.results_dir)
        if performance_data:
            print(f"✓ Loaded performance data for {len(performance_data)} tasks")
            # Show some stats
            accuracies = [v for v in performance_data.values() if v is not None]
            if accuracies:
                print(f"  Mean accuracy: {np.mean(accuracies):.1%}")
                print(f"  Range: {min(accuracies):.1%} - {max(accuracies):.1%}")
        else:
            print("⚠️  No performance data found in results dir")
    
    # Phase 9.5: Create visualizations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Sanitize model name for filename (remove slashes, special chars)
    model_name_sanitized = args.model.replace('/', '_').replace('\\', '_')
    
    # First generate matplotlib plots (always works)
    visualize_task_projections(train_tasks, test_tasks, basis, output_dir, model_name_sanitized)
    
    # Then try to generate interactive plotly plots (optional, won't crash if fails)
    try:
        visualize_task_projections_interactive(
            train_tasks, test_tasks, basis, output_dir, model_name_sanitized,
            performance=performance_data,
            color_by=args.color_by
        )
    except Exception as e:
        print(f"\n⚠️  Could not generate interactive plotly visualizations: {e}")
        print("   (This is optional - matplotlib plots were generated successfully)")
    
    # Phase 10: Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Save basis
    basis_path = output_dir / "skill_basis.npz"
    save_skill_basis(basis, str(basis_path))
    print(f"✓ Saved basis to: {basis_path}")
    
    # Save test vectors
    test_vec_dir = output_dir / "test_vectors"
    test_vec_dir.mkdir(exist_ok=True)
    for vec in test_vecs:
        vec_path = test_vec_dir / f"{vec.task_name}.npz"
        save_function_vec(vec, str(vec_path))
    print(f"✓ Saved {len(test_vecs)} test vectors to: {test_vec_dir}")
    
    # Save summary results
    import json
    summary = {
        "config": {
            "model": args.model,
            "layer": args.layer,
            "num_heads": args.num_heads,
            "num_samples": args.num_samples,
            "train_ratio": args.train_ratio,
            "k_components": k,
            "seed": args.seed
        },
        "train_tasks": [get_task_display_name(t) for t in train_tasks],
        "test_tasks": [get_task_display_name(t) for t in test_tasks],
        "epsilon_ranks": {
            name: {str(eps): int(rank) for eps, rank in data['epsilon_ranks'].items()}
            for name, data in results.items()
        },
        "explained_variance": basis.explained_variance_ratio().tolist(),
        "similarity_matrix": similarity_matrix.tolist()
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to: {summary_path}")
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE!")
    print("="*70)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"\nYou can now run further analysis:")
    print(f"  python function_vecs/experiments/analyze_epsilon_rank.py \\")
    print(f"    --basis {basis_path} \\")
    print(f"    --vecs {test_vec_dir}/*.npz \\")
    print(f"    --output {output_dir}/visualizations")


if __name__ == "__main__":
    main()

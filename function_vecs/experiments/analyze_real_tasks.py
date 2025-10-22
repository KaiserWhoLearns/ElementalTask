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

from tasks.registry import TaskRegistry
from tasks.base_task import BaseTask, TaskConfig
from function_vecs.extract_function_vecs import (
    ExtractConfig,
    extract_task_function_vec,
    stack_function_vecs,
    build_skill_basis,
    save_function_vec,
    save_skill_basis,
    Headset,
)


def discover_icl_tasks() -> List[BaseTask]:
    """Discover all tasks that support ICL format."""
    print("\n" + "="*70)
    print("DISCOVERING ICL TASKS")
    print("="*70)
    
    registry = TaskRegistry()
    registry.discover_tasks()
    all_task_names = registry.list_tasks()
    
    print(f"\nFound {len(all_task_names)} registered tasks")
    print("Checking ICL support...")
    
    # Import factory functions
    try:
        from tasks.implementations.copying_task import make_copying_task
        from tasks.implementations.ignoring_context_task import make_ignoring_context_task
        from tasks.implementations.string_analogy_task import make_string_analogy_task
        from tasks.implementations.basic_arithmetic import create_basic_arithmetic_task
        from tasks.implementations.ioi_task import create_ioi_task
        from tasks.implementations.token_reversal import create_token_reversal_task
        from tasks.implementations.pos_id import create_pos_task
        
        factories = {
            'copying': lambda: make_copying_task(use_generator=True),
            'ignoring_context': lambda: make_ignoring_context_task(use_generator=True),
            'string_analogy': lambda: make_string_analogy_task(use_generator=True),
            'basic_arithmetic': lambda: create_basic_arithmetic_task(),
            'ioi_task': lambda: create_ioi_task(),
            'token_reversal': lambda: create_token_reversal_task(),
            'part_of_speech': lambda: create_pos_task(),
        }
    except ImportError as e:
        print(f"Warning: Could not import factory functions: {e}")
        factories = {}
    
    # Try loading tasks from config files for data-dependent tasks
    config_based_tasks = {}
    try:
        import json
        from pathlib import Path
        config_dir = Path(__file__).parent.parent.parent / "tasks" / "configs"
        
        # Try loading simple_icl from config - but we'll split it by category
        simple_icl_config_path = config_dir / "simple_icl_tasks.json"
        if simple_icl_config_path.exists():
            with open(simple_icl_config_path) as f:
                simple_icl_config_data = json.load(f)
                # Check if data file exists
                data_path = Path(simple_icl_config_data.get("data_path", ""))
                if not data_path.is_absolute():
                    data_path = Path(__file__).parent.parent.parent / data_path
                if data_path.exists():
                    # Load the CSV to see categories
                    import pandas as pd
                    df = pd.read_csv(data_path)
                    if 'category_name' in df.columns:
                        categories = df['category_name'].unique()
                        print(f"  Found {len(categories)} categories in simple_icl: {list(categories)}")
                        
                        # Create separate config for each category
                        for category in categories:
                            category_data = df[df['category_name'] == category].to_dict('records')
                            category_config_data = simple_icl_config_data.copy()
                            category_config_data['name'] = f"simple_icl_{category}"
                            category_config_data['in_memory_data'] = category_data
                            category_config_data['data_format'] = "memory"
                            # Remove data_path since we're using in_memory_data
                            category_config_data.pop('data_path', None)
                            config_based_tasks[f'simple_icl_{category}'] = TaskConfig(**category_config_data)
                            print(f"    ✓ Created config for simple_icl_{category} ({len(category_data)} examples)")
        
        # Try loading textfrct - split by category similar to simple_icl
        textfrct_path = Path(__file__).parent.parent.parent / "dataset" / "TextFRCT.csv"
        if textfrct_path.exists():
            import pandas as pd
            df = pd.read_csv(textfrct_path)
            if 'category_name' in df.columns:
                # Only keep objective tasks (filter out LLMEval)
                df = df[~df['answer'].astype(str).str.contains('<LLMEval>', na=False)]
                
                categories = df['category_name'].unique()
                print(f"  Found {len(categories)} objective categories in textfrct")
                
                # Create separate task for each category (limit to categories with enough data)
                for category in categories:
                    category_data = df[df['category_name'] == category].to_dict('records')
                    if len(category_data) >= 5:  # Only include categories with at least 5 examples
                        category_config_data = {
                            'name': f"textfrct_{category.lower().replace(' ', '_')}",
                            'input_column': 'question',
                            'output_column': 'answer',
                            'in_memory_data': category_data,
                            'data_format': 'memory'
                        }
                        config_based_tasks[category_config_data['name']] = TaskConfig(**category_config_data)
                        print(f"    ✓ Created config for {category_config_data['name']} ({len(category_data)} examples)")
    except Exception as e:
        print(f"  Warning: Could not load config-based tasks: {e}")
    
    icl_tasks = []
    for task_name in all_task_names:
        # Skip the base simple_icl and textfrct tasks - we'll use the category-specific ones
        if task_name == 'simple_icl' and len([k for k in config_based_tasks if k.startswith('simple_icl_')]) > 0:
            print(f"  ⊘ {task_name} (using category-specific versions instead)")
            continue
        if task_name == 'textfrct' and len([k for k in config_based_tasks if k.startswith('textfrct_')]) > 0:
            print(f"  ⊘ {task_name} (using category-specific versions instead)")
            continue
            
        try:
            # Try using factory function first if available
            if task_name in factories:
                try:
                    task = factories[task_name]()
                    # Verify it has data
                    sample_data = task.get_split("test")
                    if sample_data and len(sample_data) > 0:
                        icl_tasks.append(task)
                        print(f"  ✓ {task_name} (factory, {len(sample_data)} examples)")
                        continue
                except Exception as factory_error:
                    print(f"  ⚠ {task_name} factory failed: {factory_error}, trying default...")
            
            # Try config-based instantiation for data-dependent tasks
            if task_name in config_based_tasks:
                try:
                    task_class = registry.get_task_class(task_name.split('_')[0] if '_' in task_name else task_name)
                    task = task_class(config_based_tasks[task_name])
                    sample_data = task.get_split("test")
                    if sample_data and len(sample_data) > 0:
                        icl_tasks.append(task)
                        print(f"  ✓ {task_name} (config, {len(sample_data)} examples)")
                        continue
                except Exception as config_error:
                    print(f"  ⚠ {task_name} config failed: {config_error}, trying default...")
            
            # Fall back to standard instantiation
            task_class = registry.get_task_class(task_name)
            config = TaskConfig(
                name=task_name,
                input_column="input",
                output_column="output"
            )
            task = task_class(config)
            
            # Check if task supports ICL and has data
            if hasattr(task, 'supports_icl') and task.supports_icl:
                # Try to get a sample split to verify task has data
                try:
                    sample_data = task.get_split("test")
                    if sample_data and len(sample_data) > 0:
                        icl_tasks.append(task)
                        print(f"  ✓ {task_name} ({len(sample_data)} examples)")
                    else:
                        print(f"  ✗ {task_name} (no data available)")
                except Exception as data_error:
                    print(f"  ✗ {task_name} (data error: {data_error})")
            else:
                print(f"  ✗ {task_name} (no ICL support)")
                
        except Exception as e:
            print(f"  ✗ {task_name} (error: {e})")
    
    # Add the category-specific tasks from config_based_tasks
    for cat_task_name in config_based_tasks:
        if cat_task_name not in [t.config.name for t in icl_tasks]:
            try:
                # Determine the base task class
                if cat_task_name.startswith('simple_icl_'):
                    # Get the base task name (e.g., 'simple_icl' from 'simple_icl_uppercase')
                    base_name = 'simple_icl'
                elif cat_task_name.startswith('textfrct_'):
                    # For textfrct, use the SimpleTask class with the config
                    base_name = 'simple'
                else:
                    continue
                    
                task_class = registry.get_task_class(base_name)
                task = task_class(config_based_tasks[cat_task_name])
                sample_data = task.get_split("test")
                if sample_data and len(sample_data) > 0:
                    icl_tasks.append(task)
                    # Already printed above
            except Exception as e:
                print(f"  ✗ {cat_task_name} (config load error: {e})")
    
    print(f"\n✓ Found {len(icl_tasks)} ICL-compatible tasks")
    return icl_tasks


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
    """Extract function vectors from a list of tasks."""
    print(f"\nExtracting function vectors from {len(tasks)} {desc}...")
    
    function_vecs = []
    for i, task in enumerate(tasks, 1):
        try:
            print(f"  [{i}/{len(tasks)}] {task.config.name}...", end=" ")
            fv = extract_task_function_vec(task, config, headset, model, tokenizer)
            function_vecs.append(fv)
            norm = np.linalg.norm(fv.function_vec)
            print(f"✓ (norm={norm:.4f})")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            continue
    
    print(f"✓ Successfully extracted {len(function_vecs)}/{len(tasks)} function vectors")
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
    test_vecs: List
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
    
    # Test task rankings
    print("\n--- Test Task Reconstruction Quality ---")
    
    # Get unique epsilons from results (sorted)
    all_epsilons = set()
    for task_name, data in results.items():
        all_epsilons.update(data['epsilon_ranks'].keys())
    sorted_epsilons = sorted(all_epsilons, reverse=True)
    
    # Build header
    header = f"{'Task':<25s}"
    for eps in sorted_epsilons:
        header += f" {f'ε={eps:.2f}':>7s}"
    header += f" {'Best Sim':>10s}"
    print(header)
    print("-" * len(header))
    
    # Sort by best similarity (k=max components)
    task_scores = []
    for vec in test_vecs:
        task_name = vec.task_name
        epsilon_ranks = results[task_name]['epsilon_ranks']
        
        # Best similarity is 1 - cosine_error at max k
        cos_errors = results[task_name]['cosine_errors']
        best_sim = 1.0 - cos_errors[-1]
        
        task_scores.append((task_name, epsilon_ranks, best_sim))
    
    task_scores.sort(key=lambda x: x[2], reverse=True)  # Sort by best similarity
    
    for task_name, epsilon_ranks, best_sim in task_scores:
        row = f"{task_name:<25s}"
        for eps in sorted_epsilons:
            rank = epsilon_ranks.get(eps, '-')
            row += f" {str(rank):>7s}"
        row += f" {best_sim:>10.4f}"
        print(row)
    
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


def main():
    parser = argparse.ArgumentParser(description="Analyze function vectors from real ICL tasks")
    parser.add_argument("--model", type=str, default="distilgpt2",
                       help="Model name (default: distilgpt2)")
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
    
    args = parser.parse_args()
    
    print("="*70)
    print("REAL TASK FUNCTION VECTOR ANALYSIS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Device: {args.device}")
    print(f"  Layer: {args.layer}")
    print(f"  Heads: {args.num_heads}")
    print(f"  Samples per task: {args.num_samples}")
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
        print(f"  • {task.config.name}")
    
    print(f"\nTest tasks ({len(test_tasks)}):")
    for task in test_tasks:
        print(f"  • {task.config.name}")
    
    # Phase 3: Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    
    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).eval()
    if args.device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    print("✓ Model loaded")
    
    # Normalize layer index (support negative indexing)
    from function_vecs.extract_function_vecs import get_blocks
    blocks = get_blocks(model)
    num_layers = len(blocks)
    layer_idx = args.layer
    if layer_idx < 0:
        layer_idx = num_layers + layer_idx
    print(f"Using layer {layer_idx} (out of {num_layers} layers)")
    
    # Setup extraction config
    config = ExtractConfig(
        model_name=args.model,
        device=args.device,
        batch_size=4,
        num_samples_per_task=args.num_samples,
        layers=[layer_idx],
        topk_heads=args.num_heads
    )
    
    # Create headset
    heads = [(layer_idx, h) for h in range(args.num_heads)]
    headset = Headset(mode="topk", heads=heads)
    
    # Phase 4: Extract training function vectors
    print("\n" + "="*70)
    print("EXTRACTING TRAINING FUNCTION VECTORS")
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
    
    # Phase 7: Analyze epsilon-ranks
    results = analyze_epsilon_ranks(test_vecs, basis, args.epsilons)
    
    # Phase 8: Compute similarities
    print("\n" + "="*70)
    print("COMPUTING SIMILARITIES")
    print("="*70)
    
    similarity_matrix = compute_similarity_matrix(test_vecs)
    print(f"\nSimilarity matrix shape: {similarity_matrix.shape}")
    
    # Phase 9: Print summary
    print_summary(train_tasks, test_tasks, basis, results, similarity_matrix, test_vecs)
    
    # Phase 10: Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
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
        "train_tasks": [t.config.name for t in train_tasks],
        "test_tasks": [t.config.name for t in test_tasks],
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

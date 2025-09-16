#!/usr/bin/env python3
"""
Plotting script for analyzing checkpoint performance curves from measure_ckpt_interp_perf.py results.

This script creates performance curves showing how different interpretation tasks evolve
across model checkpoints during training.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import argparse
import os
from typing import List, Tuple, Optional

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def extract_checkpoint_info(checkpoint_name: str) -> Tuple[int, str]:
    """
    Extract sorting key and display name from checkpoint name.
    
    Args:
        checkpoint_name: Name of the checkpoint
        
    Returns:
        Tuple of (sort_key, display_name)
    """
    # Handle OLMo-style: stage1-step140000-tokens294B
    olmo_match = re.search(r'stage(\d+).*step(\d+).*tokens(\d+)', checkpoint_name.lower())
    if olmo_match:
        stage = int(olmo_match.group(1))
        step = int(olmo_match.group(2))
        tokens = int(olmo_match.group(3))
        sort_key = stage * 10000000 + step  # Ensure stage ordering
        display_name = f"S{stage}-{tokens}B"
        return sort_key, display_name
    
    # Handle Crystal-style: CrystalCoder_phase1_checkpoint_055500
    crystal_match = re.search(r'phase(\d+)_checkpoint_(\d+)', checkpoint_name.lower())
    if crystal_match:
        phase = int(crystal_match.group(1))
        checkpoint_num = int(crystal_match.group(2))
        sort_key = phase * 1000000 + checkpoint_num
        display_name = f"P{phase}-{checkpoint_num}"
        return sort_key, display_name
    
    # Handle standard step-based: step1000, step1000-tokens4B
    step_match = re.search(r'step(\d+)', checkpoint_name.lower())
    if step_match:
        step = int(step_match.group(1))
        # Try to extract tokens if present
        token_match = re.search(r'tokens(\d+)', checkpoint_name.lower())
        if token_match:
            tokens = int(token_match.group(1))
            display_name = f"{tokens}B"
        else:
            display_name = f"Step {step}"
        return step, display_name
    
    # Fallback: use the checkpoint name as-is
    return 0, checkpoint_name

def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load CSV data and prepare it for plotting."""
    df = pd.read_csv(csv_path)
    
    # Extract checkpoint information for sorting and display
    checkpoint_info = [extract_checkpoint_info(ckpt) for ckpt in df['checkpoint']]
    df['sort_key'] = [info[0] for info in checkpoint_info]
    df['display_name'] = [info[1] for info in checkpoint_info]
    
    # Sort by sort_key
    df = df.sort_values('sort_key').reset_index(drop=True)
    
    return df

def plot_performance_curves(df: pd.DataFrame, output_path: str = None, 
                          figsize: Tuple[int, int] = (12, 8),
                          task_groups: Optional[List[List[str]]] = None):
    """
    Plot performance curves for all tasks.
    
    Args:
        df: DataFrame with checkpoint results
        output_path: Path to save the plot
        figsize: Figure size
        task_groups: Optional grouping of tasks for separate subplots
    """
    # Get task columns (excluding metadata columns)
    task_columns = [col for col in df.columns 
                   if col not in ['checkpoint', 'sort_key', 'display_name']]
    
    if task_groups is None:
        # Single plot with all tasks
        plt.figure(figsize=figsize)
        
        # Plot each task
        for task in task_columns:
            plt.plot(range(len(df)), df[task], marker='o', label=task, linewidth=2, markersize=4)
        
        plt.xlabel('Checkpoint')
        plt.ylabel('Performance Score')
        plt.title('Model Performance Across Checkpoints')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Set x-axis labels
        plt.xticks(range(len(df)), df['display_name'], rotation=45, ha='right')
        plt.ylim(0, 1.05)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {output_path}")
        else:
            plt.show()
    
    else:
        # Multiple subplots for different task groups
        n_groups = len(task_groups)
        _, axes = plt.subplots(n_groups, 1, figsize=(figsize[0], figsize[1] * n_groups // 2))
        if n_groups == 1:
            axes = [axes]
        
        group_names = [
            "Character Transformations",
            "Translation Tasks", 
            "Linguistic Transformations",
            "Knowledge Tasks"
        ]
        
        for i, (group, group_name) in enumerate(zip(task_groups, group_names[:n_groups])):
            ax = axes[i]
            
            for task in group:
                if task in task_columns:
                    ax.plot(range(len(df)), df[task], marker='o', label=task, linewidth=2, markersize=4)
            
            ax.set_xlabel('Checkpoint')
            ax.set_ylabel('Performance Score')
            ax.set_title(f'{group_name} Performance Across Checkpoints')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df['display_name'], rotation=45, ha='right')
            ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved grouped plot to {output_path}")
        else:
            plt.show()

def plot_task_categories(df: pd.DataFrame, output_path: str = None):
    """Plot performance curves grouped by task categories."""
    
    # Define task groups
    task_groups = [
        ['uppercase', 'lowercase', 'first_letter', 'last_letter'],
        ['translate_eng_fr', 'translate_fr_eng', 'translate_eng_sp', 'translate_sp_eng'],
        ['present_to_gerund', 'singular_to_plural'],
        ['country_to_capital', 'country_to_currency']
    ]
    
    plot_performance_curves(df, output_path, figsize=(14, 10), task_groups=task_groups)

def plot_summary_stats(df: pd.DataFrame, output_path: str = None):
    """Plot summary statistics across checkpoints."""
    # Get task columns
    task_columns = [col for col in df.columns 
                   if col not in ['checkpoint', 'sort_key', 'display_name']]
    
    # Calculate summary stats
    df['mean_performance'] = df[task_columns].mean(axis=1)
    df['std_performance'] = df[task_columns].std(axis=1)
    df['min_performance'] = df[task_columns].min(axis=1)
    df['max_performance'] = df[task_columns].max(axis=1)
    
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot mean with error bars
    ax1.errorbar(range(len(df)), df['mean_performance'], 
                yerr=df['std_performance'], 
                marker='o', capsize=5, capthick=2, linewidth=2)
    ax1.fill_between(range(len(df)), df['min_performance'], df['max_performance'], 
                    alpha=0.3, label='Min-Max Range')
    ax1.set_xlabel('Checkpoint')
    ax1.set_ylabel('Performance Score')
    ax1.set_title('Mean Performance Across All Tasks')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['display_name'], rotation=45, ha='right')
    ax1.set_ylim(0, 1.05)
    
    # Plot standard deviation
    ax2.plot(range(len(df)), df['std_performance'], marker='o', linewidth=2, color='red')
    ax2.set_xlabel('Checkpoint')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Performance Variability Across Tasks')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['display_name'], rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary stats plot to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot checkpoint performance curves")
    parser.add_argument("--csv_path", type=str, required=True,
                       help="Path to the CSV file with checkpoint results")
    parser.add_argument("--output_dir", type=str, default="output/analysis/plots",
                       help="Directory to save plots")
    parser.add_argument("--plot_type", type=str, choices=['all', 'curves', 'grouped', 'summary'],
                       default='all', help="Type of plot to generate")
    parser.add_argument("--figsize", nargs=2, type=int, default=[12, 8],
                       help="Figure size (width height)")
    
    args = parser.parse_args()
    
    # Load and prepare data
    print(f"Loading data from {args.csv_path}")
    df = load_and_prepare_data(args.csv_path)
    print(f"Loaded {len(df)} checkpoints with {len([c for c in df.columns if c not in ['checkpoint', 'sort_key', 'display_name']])} tasks")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots based on type
    base_name = os.path.splitext(os.path.basename(args.csv_path))[0]
    
    if args.plot_type in ['all', 'curves']:
        output_path = os.path.join(args.output_dir, f"{base_name}_performance_curves.png")
        plot_performance_curves(df, output_path, tuple(args.figsize))
    
    if args.plot_type in ['all', 'grouped']:
        output_path = os.path.join(args.output_dir, f"{base_name}_grouped_curves.png")
        plot_task_categories(df, output_path)
    
    if args.plot_type in ['all', 'summary']:
        output_path = os.path.join(args.output_dir, f"{base_name}_summary_stats.png")
        plot_summary_stats(df, output_path)
    
    print("Plotting completed!")

if __name__ == "__main__":
    main()
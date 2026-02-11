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

def extract_checkpoint_info(checkpoint_name: str) -> Tuple[int, str, Optional[float]]:
    """
    Extract sorting key, display name, and FLOPs from checkpoint name.
    
    Args:
        checkpoint_name: Name of the checkpoint
        
    Returns:
        Tuple of (sort_key, display_name, flops_in_billions)
    """
    # Handle OLMo-style: stage1-step140000-tokens294B
    olmo_match = re.search(r'stage(\d+).*step(\d+).*tokens(\d+)', checkpoint_name.lower())
    if olmo_match:
        stage = int(olmo_match.group(1))
        step = int(olmo_match.group(2))
        tokens = int(olmo_match.group(3))
        sort_key = stage * 10000000 + step  # Ensure stage ordering
        display_name = f"S{stage}-{tokens}B"
        # Estimate FLOPs: 6 * num_params * num_tokens
        # Assuming OLMo-2 7B model: 7B params
        flops = 6 * 7 * tokens  # Result in 10^21 FLOPs
        return sort_key, display_name, flops
    
    # Handle Crystal-style: CrystalCoder_phase1_checkpoint_055500
    crystal_match = re.search(r'phase(\d+)_checkpoint_(\d+)', checkpoint_name.lower())
    if crystal_match:
        phase = int(crystal_match.group(1))
        checkpoint_num = int(crystal_match.group(2))
        sort_key = phase * 1000000 + checkpoint_num
        display_name = f"P{phase}-{checkpoint_num}"
        # No token info for Crystal checkpoints
        return sort_key, display_name, None
    
    # Handle standard step-based: step1000, step1000-tokens4B
    step_match = re.search(r'step(\d+)', checkpoint_name.lower())
    if step_match:
        step = int(step_match.group(1))
        # Try to extract tokens if present
        token_match = re.search(r'tokens(\d+)', checkpoint_name.lower())
        if token_match:
            tokens = int(token_match.group(1))
            display_name = f"{tokens}B"
            # Estimate FLOPs (need to know model size, assuming 7B)
            flops = 6 * 7 * tokens
        else:
            display_name = f"Step {step}"
            flops = None
        return step, display_name, flops
    
    # Fallback: use the checkpoint name as-is
    return 0, checkpoint_name, None

def load_and_prepare_data(csv_path: str, model_params_b: float = 7.0) -> pd.DataFrame:
    """Load CSV data and prepare it for plotting.
    
    Args:
        csv_path: Path to CSV file
        model_params_b: Model parameters in billions (default: 7.0 for OLMo-2 7B)
    """
    df = pd.read_csv(csv_path)
    
    # Extract checkpoint information for sorting and display
    checkpoint_info = [extract_checkpoint_info(ckpt) for ckpt in df['checkpoint']]
    df['sort_key'] = [info[0] for info in checkpoint_info]
    df['display_name'] = [info[1] for info in checkpoint_info]
    df['flops_1e21'] = [info[2] if info[2] is not None else None for info in checkpoint_info]
    
    # Sort by sort_key
    df = df.sort_values('sort_key').reset_index(drop=True)
    
    return df

def plot_performance_curves(df: pd.DataFrame, output_path: str = None, 
                          figsize: Tuple[int, int] = (12, 8),
                          task_groups: Optional[List[List[str]]] = None,
                          x_axis: str = 'checkpoint'):
    """
    Plot performance curves for all tasks.
    
    Args:
        df: DataFrame with checkpoint results
        output_path: Path to save the plot
        figsize: Figure size
        task_groups: Optional grouping of tasks for separate subplots
        x_axis: 'checkpoint' or 'flops' for x-axis type
    """
    # Get task columns (excluding metadata columns)
    task_columns = [col for col in df.columns 
                   if col not in ['checkpoint', 'sort_key', 'display_name', 'flops_1e21']]
    
    if task_groups is None:
        # Single plot with all tasks
        plt.figure(figsize=figsize)
        
        # Determine x values based on axis type
        if x_axis == 'flops' and 'flops_1e21' in df.columns:
            # Filter out rows without FLOP data and sort by FLOPs
            df_plot = df.dropna(subset=['flops_1e21']).sort_values('flops_1e21')
            if df_plot.empty:
                print("Warning: No FLOP data available. Falling back to checkpoint axis.")
                x_axis = 'checkpoint'
            else:
                x_values = df_plot['flops_1e21']
        else:
            df_plot = df
            x_values = range(len(df_plot))
        
        # Plot each task
        for task in task_columns:
            if x_axis == 'flops' and 'flops_1e21' in df.columns and not df_plot.empty:
                plt.plot(x_values, df_plot[task], marker='o', label=task, linewidth=2, markersize=4)
            else:
                plt.plot(x_values, df_plot[task], marker='o', label=task, linewidth=2, markersize=4)
        
        if x_axis == 'flops' and 'flops_1e21' in df.columns and not df_plot.empty:
            plt.xlabel('FLOPs (10²¹)')
            plt.title('Model Performance vs Compute (FLOPs)')
        else:
            plt.xlabel('Checkpoint')
            plt.title('Model Performance Across Checkpoints')
            
        plt.ylabel('Performance Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Set x-axis labels
        if x_axis == 'checkpoint' or df_plot.empty:
            plt.xticks(range(len(df_plot)), df_plot['display_name'], rotation=45, ha='right')
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
        
        # Determine x values based on axis type
        if x_axis == 'flops' and 'flops_1e21' in df.columns:
            df_plot = df.dropna(subset=['flops_1e21']).sort_values('flops_1e21')
            if df_plot.empty:
                print("Warning: No FLOP data available. Falling back to checkpoint axis.")
                x_axis = 'checkpoint'
                x_values = range(len(df))
                df_plot = df
            else:
                x_values = df_plot['flops_1e21']
        else:
            df_plot = df
            x_values = range(len(df_plot))
        
        for i, (group, group_name) in enumerate(zip(task_groups, group_names[:n_groups])):
            ax = axes[i]
            
            for task in group:
                if task in task_columns:
                    ax.plot(x_values, df_plot[task], marker='o', label=task, linewidth=2, markersize=4)
            
            if x_axis == 'flops' and 'flops_1e21' in df.columns and not df_plot.empty:
                ax.set_xlabel('FLOPs (10²¹)')
                ax.set_title(f'{group_name} Performance vs Compute (FLOPs)')
            else:
                ax.set_xlabel('Checkpoint')
                ax.set_title(f'{group_name} Performance Across Checkpoints')
                
            ax.set_ylabel('Performance Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if x_axis == 'checkpoint' or df_plot.empty:
                ax.set_xticks(range(len(df_plot)))
                ax.set_xticklabels(df_plot['display_name'], rotation=45, ha='right')
            ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved grouped plot to {output_path}")
        else:
            plt.show()

def plot_task_categories(df: pd.DataFrame, output_path: str = None, x_axis: str = 'checkpoint'):
    """Plot performance curves grouped by task categories."""
    
    # Define task groups
    task_groups = [
        ['uppercase', 'lowercase', 'first_letter', 'last_letter'],
        ['translate_eng_fr', 'translate_fr_eng', 'translate_eng_sp', 'translate_sp_eng'],
        ['present_to_gerund', 'singular_to_plural'],
        ['country_to_capital', 'country_to_currency']
    ]
    
    plot_performance_curves(df, output_path, figsize=(14, 10), task_groups=task_groups, x_axis=x_axis)

def plot_summary_stats(df: pd.DataFrame, output_path: str = None, x_axis: str = 'checkpoint'):
    """Plot summary statistics across checkpoints."""
    # Get task columns
    task_columns = [col for col in df.columns 
                   if col not in ['checkpoint', 'sort_key', 'display_name', 'flops_1e21']]
    
    # Calculate summary stats
    df['mean_performance'] = df[task_columns].mean(axis=1)
    df['std_performance'] = df[task_columns].std(axis=1)
    df['min_performance'] = df[task_columns].min(axis=1)
    df['max_performance'] = df[task_columns].max(axis=1)
    
    # Determine x values based on axis type
    if x_axis == 'flops' and 'flops_1e21' in df.columns:
        df_plot = df.dropna(subset=['flops_1e21']).sort_values('flops_1e21')
        if df_plot.empty:
            print("Warning: No FLOP data available. Falling back to checkpoint axis.")
            x_axis = 'checkpoint'
            x_values = range(len(df))
            df_plot = df
        else:
            x_values = df_plot['flops_1e21']
    else:
        df_plot = df
        x_values = range(len(df_plot))
    
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot mean with error bars
    ax1.errorbar(x_values, df_plot['mean_performance'], 
                yerr=df_plot['std_performance'], 
                marker='o', capsize=5, capthick=2, linewidth=2)
    ax1.fill_between(x_values, df_plot['min_performance'], df_plot['max_performance'], 
                    alpha=0.3, label='Min-Max Range')
    
    if x_axis == 'flops' and 'flops_1e21' in df.columns and not df_plot.empty:
        ax1.set_xlabel('FLOPs (10²¹)')
        ax1.set_title('Mean Performance vs Compute (FLOPs)')
    else:
        ax1.set_xlabel('Checkpoint')
        ax1.set_title('Mean Performance Across All Tasks')
        
    ax1.set_ylabel('Performance Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if x_axis == 'checkpoint' or df_plot.empty:
        ax1.set_xticks(range(len(df_plot)))
        ax1.set_xticklabels(df_plot['display_name'], rotation=45, ha='right')
    ax1.set_ylim(0, 1.05)
    
    # Plot standard deviation
    ax2.plot(x_values, df_plot['std_performance'], marker='o', linewidth=2, color='red')
    
    if x_axis == 'flops' and 'flops_1e21' in df.columns and not df_plot.empty:
        ax2.set_xlabel('FLOPs (10²¹)')
        ax2.set_title('Performance Variability vs Compute (FLOPs)')
    else:
        ax2.set_xlabel('Checkpoint')
        ax2.set_title('Performance Variability Across Tasks')
        
    ax2.set_ylabel('Standard Deviation')
    ax2.grid(True, alpha=0.3)
    
    if x_axis == 'checkpoint' or df_plot.empty:
        ax2.set_xticks(range(len(df_plot)))
        ax2.set_xticklabels(df_plot['display_name'], rotation=45, ha='right')
    
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
    parser.add_argument("--x_axis", type=str, choices=['checkpoint', 'flops'], default='checkpoint',
                       help="X-axis type: checkpoint order or FLOPs")
    parser.add_argument("--model_params_b", type=float, default=7.0,
                       help="Model parameters in billions (for FLOP calculation, default: 7.0)")
    
    args = parser.parse_args()
    
    # Load and prepare data
    print(f"Loading data from {args.csv_path}")
    df = load_and_prepare_data(args.csv_path, model_params_b=args.model_params_b)
    print(f"Loaded {len(df)} checkpoints with {len([c for c in df.columns if c not in ['checkpoint', 'sort_key', 'display_name', 'flops_1e21']])} tasks")
    
    if args.x_axis == 'flops':
        flop_count = df['flops_1e21'].notna().sum() if 'flops_1e21' in df.columns else 0
        print(f"Found FLOP data for {flop_count}/{len(df)} checkpoints")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots based on type
    base_name = os.path.splitext(os.path.basename(args.csv_path))[0]
    
    if args.plot_type in ['all', 'curves']:
        suffix = '_flops' if args.x_axis == 'flops' else ''
        output_path = os.path.join(args.output_dir, f"{base_name}_performance_curves{suffix}.png")
        plot_performance_curves(df, output_path, tuple(args.figsize), x_axis=args.x_axis)
    
    if args.plot_type in ['all', 'grouped']:
        suffix = '_flops' if args.x_axis == 'flops' else ''
        output_path = os.path.join(args.output_dir, f"{base_name}_grouped_curves{suffix}.png")
        plot_task_categories(df, output_path, x_axis=args.x_axis)
    
    if args.plot_type in ['all', 'summary']:
        suffix = '_flops' if args.x_axis == 'flops' else ''
        output_path = os.path.join(args.output_dir, f"{base_name}_summary_stats{suffix}.png")
        plot_summary_stats(df, output_path, x_axis=args.x_axis)
    
    print("Plotting completed!")

if __name__ == "__main__":
    main()
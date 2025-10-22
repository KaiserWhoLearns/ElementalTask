#!/usr/bin/env python3
"""
Evaluate model checkpoints on interpretation tasks from run_interp.ipynb.

This script supports various checkpoint naming conventions:
- Standard step-based: step1000, step1000-tokens4B, etc.
- Crystal model: CrystalCoder_phase1_checkpoint_055500
- General checkpoint patterns with numbers

Usage examples:
    # List available checkpoints
    python scripts/measure_ckpt_interp_perf.py --model_id LLM360/Crystal --list_checkpoints_only
    
    # Evaluate 10 uniformly sampled checkpoints
    python scripts/measure_ckpt_interp_perf.py --model_id LLM360/Crystal --use_vllm
    
    # Evaluate specific checkpoints
    python scripts/measure_ckpt_interp_perf.py --model_id LLM360/Crystal --use_vllm \
        --checkpoints CrystalCoder_phase1_checkpoint_055500 CrystalCoder_phase3_checkpoint_027728
    
    # Evaluate checkpoints 5 through 10 (0-indexed, inclusive)
    python scripts/measure_ckpt_interp_perf.py --model_id LLM360/Crystal --use_vllm \
        --num_checkpoints 20 --begin 5 --end 10
    
    # Evaluate all checkpoints from index 10 onwards
    python scripts/measure_ckpt_interp_perf.py --model_id LLM360/Crystal --use_vllm \
        --num_checkpoints 20 --begin 10
    
    # Evaluate checkpoints from beginning up to index 5
    python scripts/measure_ckpt_interp_perf.py --model_id LLM360/Crystal --use_vllm \
        --num_checkpoints 20 --end 5
"""

import os
import sys
import argparse
import re
import pandas as pd
import torch
import vllm
from tqdm import tqdm
from huggingface_hub import list_repo_refs

sys.path.append(os.getcwd())
from scripts.inference import load_model_revision

# Task categories and examples from run_interp.ipynb
task_categories_to_examples = {
    "uppercase": ["a -> A", "c -> C"],
    "lowercase": ["A -> a", "C -> c"],
    "first_letter": ["the cat went up the tree -> t", "elephants are cool -> e"],
    "last_letter": ["the cat went up the tree -> e", "elephants are cool -> l"],
    "translate_eng_fr": ["hello -> bonjour", "goodbye -> au revoir"],
    "translate_fr_eng": ["bonjour -> hello", "au revoir -> goodbye"],
    "translate_eng_sp": ["hello -> hola", "goodbye -> adiós"],
    "translate_sp_eng": ["hola -> hello", "adiós -> goodbye"],
    "present_to_gerund": ["run -> running", "swim -> swimming"],
    "singular_to_plural": ["cat -> cats", "dog -> dogs"],
    "country_to_capital": ["France -> Paris", "Germany -> Berlin"],
    "country_to_currency": ["France -> Euro", "United States -> Dollar"],
}

def craft_icl(category):
    """Create in-context learning prompt for a given category."""
    examples = task_categories_to_examples[category]
    prompt = ""
    for example in examples:
        prompt += f"{example}\n"
    return prompt

def get_uniformly_sampled_checkpoints(model_id, num_checkpoints=10):
    """Get uniformly sampled checkpoints from HuggingFace model repository."""
    try:
        # Get all branches/revisions from the model repository
        refs = list_repo_refs(model_id)
        
        # Filter for checkpoint branches and extract sorting keys
        checkpoint_branches = []
        for branch in refs.branches:
            branch_name = branch.name
            
            # Skip main branch
            if branch_name == "main":
                continue
                
            # Handle Crystal model pattern: CrystalCoder_phase{N}_checkpoint_{NUM}
            crystal_match = re.search(r'crystalcoder_phase(\d+)_checkpoint_(\d+)', branch_name.lower())
            if crystal_match:
                phase = int(crystal_match.group(1))
                checkpoint_num = int(crystal_match.group(2))
                # Use phase * 1000000 + checkpoint_num for sorting to ensure phase ordering
                sort_key = phase * 1000000 + checkpoint_num
                checkpoint_branches.append((sort_key, branch_name, f"Phase {phase}, Checkpoint {checkpoint_num}"))
                continue
            
            # Handle stage/phase patterns with steps (e.g., stage1_step1000, phase2_step500)
            stage_step_match = re.search(r'(?:stage|phase)(\d+).*?step(\d+)', branch_name.lower())
            if stage_step_match:
                stage = int(stage_step_match.group(1))
                step_num = int(stage_step_match.group(2))
                # Use stage * 1000000 + step_num for proper ordering
                sort_key = stage * 1000000 + step_num
                checkpoint_branches.append((sort_key, branch_name, f"Stage {stage}, Step {step_num}"))
                continue
            
            # Handle standard step-based patterns (step1000, step1000-tokens4B, etc.)
            step_match = re.search(r'step(\d+)', branch_name.lower())
            if step_match:
                step_num = int(step_match.group(1))
                checkpoint_branches.append((step_num, branch_name, f"Step {step_num}"))
                continue
                
            # Handle checkpoint patterns with stage (e.g., checkpoint_stage1_100, stage1_checkpoint100)
            stage_checkpoint_match = re.search(r'(?:stage|phase)(\d+).*?(?:checkpoint|ckpt|epoch).*?(\d+)', branch_name.lower())
            if not stage_checkpoint_match:
                stage_checkpoint_match = re.search(r'(?:checkpoint|ckpt|epoch).*?(?:stage|phase)(\d+).*?(\d+)', branch_name.lower())
            
            if stage_checkpoint_match:
                stage = int(stage_checkpoint_match.group(1))
                checkpoint_num = int(stage_checkpoint_match.group(2))
                sort_key = stage * 1000000 + checkpoint_num
                checkpoint_branches.append((sort_key, branch_name, f"Stage {stage}, Checkpoint {checkpoint_num}"))
                continue
                
            # Handle other checkpoint patterns without stage
            if any(pattern in branch_name.lower() for pattern in ['checkpoint', 'epoch', 'ckpt']):
                # Try to extract any number for sorting
                num_match = re.search(r'(\d+)', branch_name)
                if num_match:
                    num = int(num_match.group(1))
                    checkpoint_branches.append((num, branch_name, f"Checkpoint {num}"))
        
        # Sort by sort key (first by stage/phase, then by checkpoint number)
        checkpoint_branches.sort(key=lambda x: x[0])
        
        if not checkpoint_branches:
            print(f"Warning: No checkpoint branches found for {model_id}")
            available_branches = [b.name for b in refs.branches]
            print("Available branches:", available_branches)
            # Return main branch as fallback
            return ["main"]
        
        # If we have fewer checkpoints than requested, return all
        if len(checkpoint_branches) <= num_checkpoints:
            selected_checkpoints = [branch[1] for branch in checkpoint_branches]
        else:
            # Uniformly sample checkpoints
            indices = []
            n = len(checkpoint_branches)
            
            # Calculate step size for uniform sampling
            step_size = (n - 1) / (num_checkpoints - 1)
            
            for i in range(num_checkpoints):
                index = int(round(i * step_size))
                indices.append(min(index, n - 1))  # Ensure we don't go out of bounds
            
            # Remove duplicates while preserving order
            seen = set()
            unique_indices = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)
            
            selected_checkpoints = [checkpoint_branches[i][1] for i in unique_indices]
        
        print(f"Found {len(checkpoint_branches)} checkpoints, sampling {len(selected_checkpoints)}:")
        for ckpt in selected_checkpoints:
            # Find the description for this checkpoint
            desc = next((c[2] for c in checkpoint_branches if c[1] == ckpt), "Unknown")
            print(f"  - {ckpt} ({desc})")
        
        return selected_checkpoints
        
    except Exception as e:
        print(f"Error accessing HuggingFace Hub for {model_id}: {e}")
        print("Please provide checkpoints manually using --checkpoints argument")
        return []

def evaluate_checkpoint_on_tasks(model_id, checkpoint, data, use_vllm=True):
    """Evaluate a single checkpoint on all tasks."""
    print(f"\nEvaluating checkpoint: {checkpoint}")
    
    # Load model
    if use_vllm:
        model = vllm.LLM(
            model=model_id,
            tokenizer=model_id,
            revision=checkpoint,
            tokenizer_mode="auto",
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=True,
        )
        
        sampling_params = vllm.SamplingParams(
            temperature=0,
            max_tokens=10,
        )
    else:
        model, tokenizer = load_model_revision(model_id, checkpoint)
    
    # Prepare prompts and answers
    all_prompts = []
    all_answers = []
    all_categories = []
    
    for _, row in data.iterrows():
        category = row['category_name']
        icl_prompt = craft_icl(category)
        icl_prompt += f"{row['question']} ->"
        all_prompts.append(icl_prompt)
        all_answers.append(row['answer'])
        all_categories.append(category)
    
    # Generate outputs
    if use_vllm:
        outputs = model.generate(all_prompts, sampling_params)
        outputs = [it.outputs[0].text for it in outputs]
    else:
        outputs = []
        for prompt in tqdm(all_prompts, desc="Generating"):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=10, temperature=0)
            
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # Extract only the generated part
            outputs.append(generated_text[len(prompt):].strip())
    
    # Calculate scores per category
    cat_to_score = {}
    for category, output, answer in zip(all_categories, outputs, all_answers):
        if category not in cat_to_score:
            cat_to_score[category] = []
        
        # Check if the first line of output matches the answer
        output_clean = output.split("\n")[0].strip()
        answer_clean = answer.strip()
        is_correct = output_clean == answer_clean
        cat_to_score[category].append(1 if is_correct else 0)
    
    # Calculate average scores
    results = {}
    for category, scores in cat_to_score.items():
        score = sum(scores) / len(scores) if scores else 0
        results[category] = score
        print(f"  {category}: {score:.2f}")
    
    # Clean up model from memory
    if use_vllm:
        del model
        torch.cuda.empty_cache()
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints on interpretation tasks")
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier (e.g., LLM360/Crystal)")
    parser.add_argument("--data_path", type=str, default="dataset/simple.csv", 
                       help="Path to the simple.csv dataset")
    parser.add_argument("--num_checkpoints", type=int, default=10, 
                       help="Number of checkpoints to evaluate")
    parser.add_argument("--output_path", type=str, default="output/checkpoint_interp_results.csv",
                       help="Path to save results")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for inference")
    parser.add_argument("--checkpoints", nargs="+", type=str, default=None,
                       help="Specific checkpoints to evaluate (overrides automatic sampling)")
    parser.add_argument("--list_checkpoints_only", action="store_true",
                       help="Only list available checkpoints without running evaluation")
    parser.add_argument("--begin", type=int, default=-1,
                       help="Start evaluation at the begin-th checkpoint (0-indexed). Use -1 to disable range filtering")
    parser.add_argument("--end", type=int, default=-1,
                       help="End evaluation at the end-th checkpoint (0-indexed, inclusive). Use -1 to disable range filtering")
    
    args = parser.parse_args()
    
    # If only listing checkpoints, do that and exit
    if args.list_checkpoints_only:
        print(f"Listing available checkpoints for {args.model_id}...")
        if "Crystal" in args.model_id:
            print("Note: Crystal model uses phase-based training with checkpoints like 'CrystalCoder_phase1_checkpoint_055500'")
        checkpoints = get_uniformly_sampled_checkpoints(args.model_id, num_checkpoints=1000)  # Get all
        if checkpoints:
            print(f"\nTotal available checkpoints: {len(checkpoints)}")
        return
    
    # Load data
    print(f"Loading data from {args.data_path}")
    data = pd.read_csv(args.data_path)
    if 'index' in data.columns:
        data = data.drop(columns=['index'])
    
    # Get checkpoints
    if args.checkpoints:
        checkpoints = args.checkpoints
    elif args.begin == -1:
        checkpoints = get_uniformly_sampled_checkpoints(args.model_id, args.num_checkpoints)
    else:
        # Get all available checkpoints
        checkpoints = get_uniformly_sampled_checkpoints(args.model_id, num_checkpoints=100000)  # Get all

    
    # Apply begin/end filtering if specified
    if args.begin != -1:
        if args.begin >= len(checkpoints):
            print(f"ERROR: --begin ({args.begin}) is out of range. Total checkpoints: {len(checkpoints)}")
            return
        
        if args.end != -1:
            # Both begin and end specified
            if args.end >= len(checkpoints):
                print(f"ERROR: --end ({args.end}) is out of range. Total checkpoints: {len(checkpoints)}")
                return
            if args.begin > args.end:
                print(f"ERROR: --begin ({args.begin}) must be less than or equal to --end ({args.end})")
                return
            
            checkpoints = checkpoints[args.begin:args.end + 1]  # +1 because end is inclusive
            print(f"Evaluating checkpoints from index {args.begin} to {args.end} (inclusive)")
        else:
            # Only begin specified
            checkpoints = checkpoints[args.begin:]
            print(f"Evaluating checkpoints from index {args.begin} to the end")
    elif args.end != -1:
        # Only end specified
        if args.end >= len(checkpoints):
            print(f"ERROR: --end ({args.end}) is out of range. Total checkpoints: {len(checkpoints)}")
            return
        checkpoints = checkpoints[:args.end + 1]  # +1 because end is inclusive
        print(f"Evaluating checkpoints from the beginning to index {args.end} (inclusive)")
    
    print(f"Evaluating {len(checkpoints)} checkpoints on {len(task_categories_to_examples)} tasks")
    
    # Evaluate each checkpoint
    all_results = []
    for i, checkpoint in enumerate(checkpoints):
        print(f"\n=== Evaluating checkpoint {i+1}/{len(checkpoints)}: {checkpoint} ===")
        
        results = evaluate_checkpoint_on_tasks(
            args.model_id, 
            checkpoint, 
            data,
            use_vllm=args.use_vllm
        )
        
        # Add checkpoint info to results
        results['checkpoint'] = checkpoint
        all_results.append(results)
        
        # Print current results
        print(f"Results for {checkpoint}:")
        for task, score in results.items():
            if task != 'checkpoint':
                print(f"  {task}: {score:.3f}")
    
    # Convert to DataFrame and save
    if not all_results:
        print("ERROR: No results to save!")
        return
        
    results_df = pd.DataFrame(all_results)
    
    # Debug: Print DataFrame info
    print(f"\nDataFrame shape: {results_df.shape}")
    print(f"DataFrame columns: {list(results_df.columns)}")
    
    # Reorder columns to have checkpoint first
    cols = ['checkpoint'] + [col for col in results_df.columns if col != 'checkpoint']
    results_df = results_df[cols]
    
    # Save results
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        results_df.to_csv(args.output_path, index=False)
        print(f"\nResults saved to {args.output_path}")
        
        # Verify file was created
        if os.path.exists(args.output_path):
            file_size = os.path.getsize(args.output_path)
            print(f"File size: {file_size} bytes")
            
            # Show first few lines of CSV
            print("\nFirst few lines of saved CSV:")
            with open(args.output_path, 'r') as f:
                for i, line in enumerate(f):
                    if i < 5:  # Show first 5 lines
                        print(f"  {line.strip()}")
                    else:
                        break
        else:
            print("ERROR: CSV file was not created!")
    except Exception as e:
        print(f"ERROR saving results: {e}")
        return
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for task in task_categories_to_examples.keys():
        if task in results_df.columns:
            print(f"{task}:")
            print(f"  Mean: {results_df[task].mean():.3f}")
            print(f"  Std:  {results_df[task].std():.3f}")
            print(f"  Min:  {results_df[task].min():.3f}")
            print(f"  Max:  {results_df[task].max():.3f}")

if __name__ == "__main__":
    main()
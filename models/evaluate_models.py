import os
import sys
import argparse
import tqdm
import pdb
import torch
import vllm
from datasets import Dataset
sys.path.append(os.getcwd())
from scripts.inference import load_model_revision
from tasks.registry import get_task

def preprocess_5shot(dataset):
    # Sample 5 instances from the dataset
    sampled_instances = dataset.shuffle(seed=42).select(range(5))

    # Remove the sampled instance from the dataset
    dataset = dataset.filter(lambda x: x not in sampled_instances)
    prompt = "Provide a response based on the following examples:\n"
    for instance in sampled_instances:
        prompt += f"Input: {instance['input']}\n{instance['output']}\n"

    def prompt_formatting(instance):
        # Format the prompt
        instance["prompt"] = prompt + f"Input: {instance['input']}\n"
        return instance
    dataset = dataset.map(prompt_formatting)
    return dataset

def evaluate_model(
    model_id: str,
    chkpt: str,
    task_name: str,
    output_path: str = None,
    use_vllm: bool = True,
    max_new_tokens: int = 100,
    preprocess_fn: callable = preprocess_5shot,
    num_shots: int = 5,
    spaced: bool = False,
):
    # Load the dataset with optional spaced mode
    task = get_task(task_name, spaced=spaced)
    # pdb.set_trace()
    test_data = list(task.get_split("test"))
    
    # Build prompts using task's build_prompt method
    prompts = []

    for instance in test_data:
        # Try to use task's build_prompt method if available
        if hasattr(task, 'build_prompt'):
            prompts.append(task.build_prompt(instance, num_shots=num_shots))
        elif "prompt" in instance:
            prompts.append(instance["prompt"])
        elif "input" in instance:
            prompts.append(instance["input"])
        else:
            # Fallback: try to find any text-like column
            text_cols = [k for k, v in instance.items() if isinstance(v, str)]
            if text_cols:
                prompts.append(instance[text_cols[0]])
            else:
                raise ValueError(f"Cannot determine prompt for instance: {instance.keys()}")
    
    # Create dataset with prompts
    dataset = Dataset.from_list([{**item, "prompt": prompt} for item, prompt in zip(test_data, prompts)])

    # Don't apply preprocess_fn if we're already using task.build_prompt with num_shots
    # (to avoid adding ICL examples twice)
    if preprocess_fn and (not hasattr(task, 'build_prompt') or num_shots == 0):
        dataset = preprocess_fn(dataset)

    # Load the model
    if use_vllm:
        model = vllm.LLM(
            model=model_id,
            tokenizer=model_id,
            revision=chkpt,
            tokenizer_mode="auto",
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=True,
        )
        
        sampling_params = vllm.SamplingParams(
            temperature=0,  # greedy decoding
            max_tokens=max_new_tokens,
        )
                
        outputs = model.generate(dataset["prompt"], sampling_params)
        outputs = [it.outputs[0].text for it in outputs]
        breakpoint()
    else:
        model, tokenizer = load_model_revision(model_id, chkpt)
        generated_texts = []
        for prompt in dataset["prompt"]:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
            # del inputs["token_type_ids"]
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate output
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
            
            # Extract only the newly generated tokens (excluding the input prompt)
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            generated_texts.append(tokenizer.decode(generated_tokens, skip_special_tokens=True))
    dataset = dataset.add_column("predictions", generated_texts)
    # Save the predictions if output_path is provided
    if output_path:
        # Sanitize task name for file path
        task_name_safe = task_name.replace(':', '_').replace(',', '_')
        if spaced:
            task_name_safe += "_spaced"
        
        os.makedirs(output_path, exist_ok=True)
        
        # Get ground truth for computing correctness
        ground_truth = task.get_ground_truth("test")
        
        # Group by category if present, and add correct field
        category_items = {}  # category -> list of items
        
        for i, item in enumerate(dataset):
            pred = item.get('predictions', '')
            gt = ground_truth[i] if i < len(ground_truth) else ''
            
            # Compute correctness
            pred_clean = pred.split('\n')[0].strip().lower() if pred else ""
            gt_clean = gt.strip().lower() if gt else ""
            is_correct = (pred_clean == gt_clean)
            
            # Create detailed item
            detailed_item = {
                "index": i,
                "input": item.get('input', item.get('question', '')),
                "prompt": item.get('prompt', ''),
                "prediction": pred,
                "target": gt,
                "correct": is_correct,
                "metadata": {
                    "category_name": item.get('category_name', ''),
                    "question": item.get('question', item.get('input', '')),
                    "answer": item.get('answer', item.get('output', gt)),
                }
            }
            
            # Group by category
            category = item.get('category_name', '')
            if category:
                if category not in category_items:
                    category_items[category] = []
                category_items[category].append(detailed_item)
            else:
                if '_default' not in category_items:
                    category_items['_default'] = []
                category_items['_default'].append(detailed_item)
        
        # Save files - one per category or single file if no categories
        import json
        
        if len(category_items) == 1 and '_default' in category_items:
            # No categories - save single file
            file_name = os.path.join(output_path, f"{model_id.replace('/', '_')}_{chkpt}_{task_name_safe}_detailed.jsonl")
            with open(file_name, 'w', encoding='utf-8') as f:
                for item in category_items['_default']:
                    f.write(json.dumps(item, default=str) + '\n')
            print(f"Saved {len(category_items['_default'])} predictions to {file_name}")
        else:
            # Multiple categories - save separate files
            for category, items in category_items.items():
                if category == '_default':
                    continue
                category_safe = category.replace(':', '_').replace(',', '_').replace(' ', '_')
                file_name = os.path.join(output_path, f"{model_id.replace('/', '_')}_{chkpt}_{task_name_safe}_{category_safe}_detailed.jsonl")
                with open(file_name, 'w', encoding='utf-8') as f:
                    for item in items:
                        f.write(json.dumps(item, default=str) + '\n')
                
                # Count correct
                num_correct = sum(1 for item in items if item['correct'])
                print(f"  Saved {category}: {num_correct}/{len(items)} correct -> {os.path.basename(file_name)}")
        
        print(f"Predictions saved to {output_path}")
    # Evaluate the model
    metrics = task.evaluate(dataset["predictions"], split="test", updated_dataset=dataset.to_list())
    print(f"Metrics for {model_id} at {chkpt}: {metrics}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset.")
    parser.add_argument("--model_id", type=str, default="allenai/OLMo-1B-hf", help="Model identifier from Hugging Face.")
    parser.add_argument("--chkpt", type=str, default="step101000-tokens423B", help="Checkpoint identifier for the model.")
    parser.add_argument("--task_name", type=str, default="FRCT_FA1_ControlledAssociations", help="Path to the dataset for evaluation.")
    parser.add_argument("--output_path", default="output/", type=str, help="Path to save the evaluation results.")
    parser.add_argument("--load_vllm", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max tokens for generation.")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_id=args.model_id,
        chkpt=args.chkpt,
        task_name=args.task_name,
        output_path=args.output_path,
        use_vllm=args.load_vllm,
        max_new_tokens=args.max_new_tokens
    )
    
    # print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
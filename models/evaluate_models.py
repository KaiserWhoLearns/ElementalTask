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
from tasks.base_task import get_task

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
):
    # Load the dataset
    task = get_task(task_name)
    # pdb.set_trace()
    dataset = Dataset.from_list(list(task.get_split("test")))

    if preprocess_fn:
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
                
        outputs = model.generate(dataset["prompt"] if "prompt" in dataset else dataset["input"], sampling_params)
        outputs = [it.outputs[0].text for it in outputs]
    else:
        model, tokenizer = load_model_revision(model_id, chkpt)
        generated_texts = []
        prompts = dataset["prompt"] if "prompt" in dataset else dataset["input"]
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
            # del inputs["token_type_ids"]
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate output
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_texts.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    dataset = dataset.add_column("predictions", generated_texts)
    # Save the predictions if output_path is provided
    if output_path:
        file_name = os.path.join(output_path, f"{model_id.replace('/', '_')}_{chkpt}.jsonl")
        os.makedirs(output_path, exist_ok=True)
        dataset.to_json(file_name, orient="records", lines=True)
        print(f"Predictions saved to {output_path}")
    # Evaluate the model
    metrics = task.evaluate(dataset["predictions"], split="test", updated_dataset=dataset.to_list())
    print(f"Metrics for {model_id} at {chkpt}: {metrics}")

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
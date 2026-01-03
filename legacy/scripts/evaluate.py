import os
import sys
import argparse
import tqdm
import torch
import vllm
sys.path.append(os.getcwd())
from datasets import load_dataset, load_from_disk
from scripts.inference import load_model_revision

def evaluate_model(
    model_id: str,
    chkpt: str,
    dataset_path: str,
    output_path: str,
    local_dataset: bool = False,
    subset: str = "test",
    use_vllm: bool = True,
    max_new_tokens: int = 100,
):    
    if local_dataset:
        # Load local dataset
        data = load_from_disk(dataset_path)
    else:
        data = load_dataset(dataset_path, subset, split="validation")
    
    # # TODO: this is the format from kilt that probably needs to be generalized
    prompts = [item["input"] for item in data]
    answers = [item["output"][0]["answer"] for item in data]

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
                
        outputs = model.generate(prompts, sampling_params)
        outputs = [it.outputs[0].text for it in outputs]
    else:
        model, tokenizer = load_model_revision(model_id, chkpt)
        generated_texts = []
        for prompt in prompts[:10]:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
            del inputs["token_type_ids"]
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate output
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100)
            generated_texts.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        print(generated_texts)

        raise NotImplementedError

    acc = sum(
        1 if output.strip() == answer.strip() else 0 for output, answer in zip(outputs, answers)
    ) / len(answers)
    print(f"Accuracy: {acc:.4f}")



def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset.")
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier from Hugging Face.")
    parser.add_argument("--chkpt", type=str, help="Checkpoint identifier for the model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset for evaluation.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the evaluation results.")
    parser.add_argument("--subset", type=str, required=True, help="The split from the datasets to use.")
    parser.add_argument("--local_dataset", action="store_true", help="Use a local dataset instead of downloading from Hugging Face.")
    parser.add_argument("--load_vllm", action="store_true", help="True if using vllm for inference, else False; defaults to False.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max tokens for generation.")
    
    args = parser.parse_args()

    # Placeholder for evaluation logic
    print(f"Evaluating model {args.model_id} on dataset {args.dataset_path}...")

    # Need to enable multiprocessing for VLLM, disabled by default
    if args.load_vllm:
        os.environ["LLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    evaluate_model(
        args.model_id,
        args.chkpt,
        args.dataset_path,
        args.output_path,
        args.local_dataset,
        args.subset,
        args.load_vllm,
        args.max_new_tokens,
    )
    
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
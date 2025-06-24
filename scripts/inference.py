import os
import sys
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple
import argparse
sys.path.append(os.getcwd())
from datasets import load_dataset
#### OLMO
# Base model path

def load_model_revision(model_id: str, ckpt: Optional[str], use_vllm=False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    if "LLM360" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(
            "LLM360/CrystalCoder",
            revision=ckpt,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            "LLM360/CrystalCoder",
            revision=ckpt,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=ckpt, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=ckpt, device_map="auto", trust_remote_code=True)
    
    return model, tokenizer

def predict(model, tokenizer, dataset, output_path):
    generated_texts = []
    for example in tqdm(dataset):
        input_text = example["lm_input"]

        # Tokenize input and move to device
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        del inputs["token_type_ids"]
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate output
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        generated_texts.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    dataset.add_column("model_output", generated_texts)
    dataset.to_json(output_path, lines=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset.")
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier from Hugging Face.")
    parser.add_argument("--chkpt", type=str, help="Checkpoint identifier for the model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset for evaluation.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the evaluation results.")
    parser.add_argument("--subset", type=str, required=True, help="The split from the datasets to use")
    parser.add_argument("--local_dataset", action="store_true", help="Use a local dataset instead of downloading from Hugging Face.")
    
    args = parser.parse_args()

    model_id = args.model_id
    # https://huggingface.co/allenai/OLMo-1B-hf/tree/main
    # https://huggingface.co/allenai/OLMo-2-1124-7B/tree/main
    # model_id = "LLM360/CrystalCoder"
    # https://huggingface.co/LLM360/Crystal

    # Load a specific revision (replace with actual revision string, e.g., a commit hash or tag)
    ckpt = args.chkpt  # This should match the revision name in the repo
    # ckpt = "CrystalCoder_phase1_checkpoint_055500"

    # Load the model and tokenizer at the specific revision
    if "LLM360" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(
            "LLM360/CrystalCoder",
            revision=ckpt,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            "LLM360/CrystalCoder",
            revision=ckpt,
            trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=ckpt)
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=ckpt, device_map="auto")

    dataset = load_dataset("your_dataset_name", split="train")

    # Generate sample output
    prompt = "The universe began with"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


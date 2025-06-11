import os
import sys
sys.path.append(os.getcwd())
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

def prepare_kilt(subset: str = "nq"):
    """
    Download the KILT dataset and save it to the disk.
    """
    dataset = load_dataset("facebook/kilt_tasks", split="validation", name=subset)
    # Convert to universal format
    def convert_to_universal_format(example):
        example["lm_input"] = example["input"]
        example["reference"] = example["output"][0]["answer"]
        return example
    # Add prompt if needed
    dataset = dataset.map(convert_to_universal_format)
    # Save to disk as jsonl
    dataset.to_json(f"{os.environ['base_dir']}/data/kilt/{subset}.jsonl", lines=True)

if __name__ == "__main__":
    prepare_kilt(subset="nq")
from tasks.registry import discover_tasks, get_task, list_tasks, get_task_info
from tasks.base_task import TaskConfig, BaseTask

from dataclasses import dataclass, field
from typing import List, Dict, Any


class ExtractConfig:
    # function vector related arguments
    model_name: str = "EleutherAI/gpt-j-6B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    seed: int = 42
    layers: Optional[List[int]] = None  # If None, use all layers

    num_samples_per_task: int = 20
    num_shuffled_controls_per_task: int = 10
    head_selection: Literal["topk", "soft"] = "topk"
    topk_heads: int = 10
    cached_headset_path: Optional[str] = None # use a cached set of heads to save computation time

    # basis related arguments
    basis_method: Literal["svd", "pca"] = "svd"
    basis_dim: int = 20
    eps: float = 0.01 # for eps-rank, see notes


    
def discover_all_tasks():
    """Discover and list all available tasks."""
    print("Listing available tasks...")
    tasks = discover_tasks()
    print(f"Found {len(tasks)} tasks:")
    
    task_info = get_task_info()
    for task_name, info in task_info.items():
        print(f"  • {task_name}: {info['class']} - {info['docstring'][:100]}...")
    
    return list(tasks.keys())

def get_shuffled_prompts():
    raise NotImplementedError("This function is not yet implemented.")

def get_contribution_of_attn_head():
    raise NotImplementedError("This function is not yet implemented.")


def extract_informative_heads(config: ExtractConfig, tasks: List[BaseTask]) -> Dict:
    """Extract a fixed set of heads and weights for a set of seed tasks. {(l, i): weight}"""
    raise NotImplementedError("This function is not yet implemented.")

def extract_task_function_vec(task: BaseTask, config: ExtractConfig, head_set: Dict) -> Dict:
    """Extract function vector for a specific task."""
    raise NotImplementedError("This function is not yet implemented.")

def build_skill_basis(function_vecs: Dict[str, np.ndarray], method="svd", k=-1) -> Any:
    """Build a skill basis from a set of function vectors."""
    raise NotImplementedError("This function is not yet implemented.")

if __name__ == "__main__":
    # Discover and list all tasks
    discover_all_tasks()
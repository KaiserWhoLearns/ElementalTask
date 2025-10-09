from tasks.registry import discover_tasks, get_task, list_tasks, get_task_info
from tasks.base_task import TaskConfig, BaseTask

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Literal

import numpy as np
import torch

@dataclass
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

@dataclass
class Headset:
    mode: Literal["topk", "soft"]
    heads: List[Tuple[int, int]] = field(default_factory=list)  # list of (layer, head) tuples
    weights: Optional[np.ndarray] = None  # Optional weights for each head

@dataclass
class TaskHeadMeans:
    task_name: str
    residual_means: np.ndarray

@dataclass
class TaskFunctionVec:
    task_name: str
    function_vec: np.ndarray
    normalization: Literal["l2", "none"] = "l2"

@dataclass
class TaskMatrix:
    V: np.ndarray
    task_names: List[str]

@dataclass
class SkillBasis:
    method: Literal["svd", "pca"]
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    task_names: List[str]

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


def build_function_vec_from_means(
    head_means: TaskHeadMeans,
    head_set: Headset,
    normalization: Literal["l2", "none"] = "l2"
) -> TaskFunctionVec:
    """
    Combine the per-head residual stream means into a single function vector representing the task.
    """
    means = np.asarray(head_means.residual_means)
    assert means.ndim == 2, "Residual means should be a 2D array"
    d_model, H = means.shape

    if head_set.mode == "topk":
        vec_d = means.sum(axis=1)
    elif head_set.mode == "soft":
        weights = head_set.weights
        assert weights is not None, "Weights must be provided for soft head selection"
        assert len(weights) == H, "Weights length must match number of heads"
        vec_d = means @ weights
    else:
        raise ValueError(f"Unknown head selection mode: {head_set.mode}")
    
    if normalization == "l2":
        vec_d /= np.linalg.norm(vec_d) + 1e-10  # avoid division by zero

    return TaskFunctionVec(task_name=head_means.task_name, function_vec=vec_d, normalization=normalization)

def stack_function_vecs(task_vecs: List[TaskFunctionVec]) -> TaskMatrix:
    assert len(task_vecs) > 0, "No task vectors provided"
    vecs = [np.asarray(tv.function_vec) for tv in task_vecs]
    v_space = np.column_stack(vecs)
    return TaskMatrix(V=v_space, task_names=[tv.task_name for tv in task_vecs])

def build_skill_basis(task_vec_matrix: TaskMatrix, method="svd", k=-1) -> SkillBasis:
    """Build a skill basis from a set of function vectors."""
    # NOTE: just svd for now
    V = np.asarray(task_vec_matrix.V, dtype=np.float64)
    mean = V.mean(axis=1, keepdims=True)
    V_centered = V - mean

    U, S, Vt = np.linalg.svd(V_centered, full_matrices=False)

    if k == -1: # select based on energy
        energy = np.cumsum(S**2) / np.sum(S**2)
        k = int(np.searchsorted(energy, 0.95) + 1)

    U = U[:, :k].astype(np.float32, copy=False)
    S = S[:k].astype(np.float32, copy=False)
    Vt = Vt[:k, :].astype(np.float32, copy=False)

    return SkillBasis(method=method, U=U, S=S, Vt=Vt, task_names=task_vec_matrix.task_names)


if __name__ == "__main__":
    # Discover and list all tasks
    discover_all_tasks()
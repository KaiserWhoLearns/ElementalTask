from tasks.registry import discover_tasks, get_task, list_tasks, get_task_info
from tasks.base_task import TaskConfig, BaseTask
from function_vecs.model_internal_getters import ResidualCapture, get_blocks, get_attn, get_o_proj, infer_head_dims
from function_vecs.activation_patching import HeadSwap
 
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Literal
from itertools import islice

import numpy as np
import torch
import torch.nn as nn

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
    mean: Optional[np.ndarray] = None  # Mean vector used during centering
    
    def reconstruct(
        self,
        task_vec: TaskFunctionVec,
        k: Optional[int] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Reconstruct a task vector using top-k basis vectors.
        
        Args:
            task_vec: Vector to reconstruct
            k: Number of basis vectors to use (None = use all available)
            normalize: Whether to normalize the reconstruction (for cosine-based analysis)
            
        Returns:
            Reconstructed vector (d_model,)
        """
        v = np.asarray(task_vec.function_vec, dtype=np.float64)
        
        # Center the vector if we have a mean (for centered PCA)
        if self.mean is not None:
            v_work = v - self.mean.flatten()
        else:
            v_work = v
        
        # Use top-k basis vectors
        if k is None:
            k = self.U.shape[1]
        k = min(k, self.U.shape[1])
        
        U_k = self.U[:, :k]
        
        # Project onto basis and reconstruct
        alpha = U_k.T @ v_work  # (k,)
        v_reconstructed = U_k @ alpha  # (d_model,)
        
        # Add mean back if centered
        if self.mean is not None:
            v_reconstructed = v_reconstructed + self.mean.flatten()
        
        # Normalize if requested (for cosine-based comparison)
        if normalize and self.mean is None:  # Only normalize for non-centered bases
            norm = np.linalg.norm(v_reconstructed)
            if norm > 1e-10:
                v_reconstructed = v_reconstructed / norm
        
        return v_reconstructed.astype(np.float32)
    
    def epsilon_rank(
        self, 
        task_vec: TaskFunctionVec, 
        epsilon: float = 0.01,
        metric: Literal["cosine", "relative", "absolute"] = "cosine",
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        Find minimum k basis vectors needed to reconstruct task_vec within epsilon.
        
        Args:
            task_vec: The task function vector to reconstruct
            epsilon: Error threshold (default 0.01 = 1%)
            metric: Error metric to use:
                - "cosine": 1 - cosine_similarity (best for normalized vectors, range [0,2])
                - "relative": ||v - v̂|| / ||v|| (L2 relative error, range [0,1])  
                - "absolute": ||v - v̂|| (raw L2 error)
            return_details: If True, return dict with errors, projections, etc.
            
        Returns:
            If return_details=False: just the epsilon-rank (int)
            If return_details=True: dict with {
                "epsilon_rank": int,
                "errors": np.ndarray,  # error for k=0, 1, 2, ..., max_k
                "cosine_errors": np.ndarray,  # cosine distance
                "projections": np.ndarray,  # coordinates in basis
                "reconstruction": np.ndarray  # reconstructed vector at epsilon_rank
            }
        """
        v = np.asarray(task_vec.function_vec, dtype=np.float64)
        max_k = self.U.shape[1]
        
        # Center the vector if we have a mean
        if self.mean is not None:
            v_work = v - self.mean.flatten()
        else:
            v_work = v
        
        v_norm = np.linalg.norm(v_work)
        
        # Compute all projections once
        projections = self.U.T @ v_work  # (max_k,)
        
        # Compute reconstruction errors for each k
        l2_errors = np.zeros(max_k + 1, dtype=np.float64)
        cosine_errors = np.zeros(max_k + 1, dtype=np.float64)
        
        # k=0: no reconstruction
        l2_errors[0] = v_norm
        cosine_errors[0] = 1.0  # Worst case: orthogonal
        
        epsilon_rank = max_k  # default if threshold never met
        
        for k in range(1, max_k + 1):
            # Reconstruct with top k vectors
            v_recon = self.U[:, :k] @ projections[:k]
            
            # L2 error
            l2_error = np.linalg.norm(v_work - v_recon)
            l2_errors[k] = l2_error
            
            # Cosine error (for normalized vectors or if no centering)
            if self.mean is None:
                # Normalize both for cosine comparison
                v_norm_val = np.linalg.norm(v_work)
                v_recon_norm = np.linalg.norm(v_recon)
                if v_norm_val > 1e-10 and v_recon_norm > 1e-10:
                    cosine_sim = np.dot(v_work, v_recon) / (v_norm_val * v_recon_norm)
                    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)  # Numerical stability
                    cosine_errors[k] = 1.0 - cosine_sim
                else:
                    cosine_errors[k] = 1.0
            else:
                # For centered data, cosine doesn't make as much sense
                cosine_errors[k] = l2_errors[k] / (v_norm + 1e-10)
            
            # Check if we've met the threshold
            if metric == "cosine":
                error_val = cosine_errors[k]
            elif metric == "relative":
                error_val = l2_errors[k] / (v_norm + 1e-10)
            else:  # absolute
                error_val = l2_errors[k]
            
            if error_val <= epsilon and epsilon_rank == max_k:
                epsilon_rank = k
        
        if return_details:
            relative_errors = l2_errors / (v_norm + 1e-10)
            final_reconstruction = self.reconstruct(task_vec, k=epsilon_rank, normalize=(self.mean is None))
            
            return {
                "epsilon_rank": epsilon_rank,
                "errors": l2_errors.astype(np.float32),
                "relative_errors": relative_errors.astype(np.float32),
                "cosine_errors": cosine_errors.astype(np.float32),
                "projections": projections.astype(np.float32),
                "reconstruction": final_reconstruction,
                "original_norm": float(v_norm),
                "metric": metric
            }
        else:
            return epsilon_rank
    
    def batch_epsilon_rank(
        self,
        task_vecs: List[TaskFunctionVec],
        epsilon: float = 0.01,
        metric: Literal["cosine", "relative", "absolute"] = "cosine"
    ) -> Dict[str, int]:
        """
        Compute epsilon-rank for multiple task vectors efficiently.
        
        Args:
            task_vecs: List of task function vectors to analyze
            epsilon: Error threshold
            metric: Error metric to use ("cosine", "relative", or "absolute")
            
        Returns:
            Dictionary mapping task_name -> epsilon_rank
        """
        results = {}
        for task_vec in task_vecs:
            eps_rank = self.epsilon_rank(task_vec, epsilon=epsilon, metric=metric, return_details=False)
            results[task_vec.task_name] = eps_rank
        
        return results
    
    def explained_variance_ratio(self) -> np.ndarray:
        """
        Return the cumulative fraction of variance explained by first k components.
        Useful for scree plots.
        
        Returns:
            Array of shape (k,) with cumulative variance ratios
        """
        variance_explained = self.S**2 / np.sum(self.S**2)
        return np.cumsum(variance_explained)

def save_function_vec(vec: TaskFunctionVec, filepath: str):
    """Save a TaskFunctionVec to .npz file."""
    np.savez_compressed(
        filepath,
        task_name=vec.task_name,
        function_vec=vec.function_vec,
        normalization=vec.normalization
    )

def load_function_vec(filepath: str) -> TaskFunctionVec:
    """Load a TaskFunctionVec from .npz file."""
    data = np.load(filepath, allow_pickle=True)
    return TaskFunctionVec(
        task_name=str(data['task_name']),
        function_vec=data['function_vec'],
        normalization=str(data['normalization'])
    )

def save_skill_basis(basis: SkillBasis, filepath: str):
    """Save a SkillBasis to .npz file."""
    np.savez_compressed(
        filepath,
        method=basis.method,
        U=basis.U,
        S=basis.S,
        Vt=basis.Vt,
        task_names=np.array(basis.task_names, dtype=object),
        mean=basis.mean if basis.mean is not None else np.array([])
    )

def load_skill_basis(filepath: str) -> SkillBasis:
    """Load a SkillBasis from .npz file."""
    data = np.load(filepath, allow_pickle=True)
    mean = data['mean'] if data['mean'].size > 0 else None
    return SkillBasis(
        method=str(data['method']),
        U=data['U'],
        S=data['S'],
        Vt=data['Vt'],
        task_names=data['task_names'].tolist(),
        mean=mean
    )

def _batch_iter(iterable, n):
    iterable = iter(iterable)
    while True:
        chunk = list(islice(iterable, n))
        if not chunk:
            break
        yield chunk

@torch.no_grad()
def _score_batch(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    score_metric: str,
    gold_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Score the model on each prompt at the decision token (last non-pad).
    If gold_ids is provided (B,), return log p(gold) or margin at that token.
    """
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=False).to(device)
    out = model(**batch, use_cache=False, return_dict=True)

    # decision index = last non-pad
    t_star = (batch["attention_mask"].sum(dim=1) - 1).clamp(min=0)  # (B,)
    B = t_star.shape[0]
    logits = out.logits[torch.arange(B, device=device), t_star, :]   # (B,V)
    logp = torch.log_softmax(logits, dim=-1)

    if gold_ids is None:
        # Fallback (not recommended for AIE): next token in the input sequence
        ids = batch["input_ids"]
        gold = ids[torch.arange(B, device=device), (t_star + 1).clamp(max=ids.shape[1]-1)]
    else:
        gold = gold_ids.to(device)

    if score_metric == "logprob":
        return logp[torch.arange(B, device=device), gold]            # (B,)
    else:
        top2 = torch.topk(logp, k=2, dim=-1).values                  # (B,2)
        return top2[:, 0] - top2[:, 1]                               # (B,)


def _cache_attn_preproj(model, tokenizer, texts, layer_idx, device):
    with ResidualCapture(model, layer_idx) as cap:
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=False).to(device)
        _ = model(**batch, use_cache=False, return_dict=True)
    return cap.cache["attn_pre_proj"], batch["attention_mask"]  # (B,T,D), (B,T)


@torch.no_grad()
def _batch_per_head_contribs(
    model: nn.Module,
    tokenizer,
    batch_texts: List[str],
    layer_idx: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Returns per-head contributions to the residual at the decision token
    using your hook-only path. Shape: (B, H, D)
    """
    # tokenize
    batch = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=False).to(device)
    attn_mask = batch["attention_mask"]
    t_star = (attn_mask.sum(dim=1) - 1).clamp(min=0)  # (B,)

    # run once with ResidualCapture to cache pre-proj & o_proj outputs
    with ResidualCapture(model, layer_idx) as cap:
        _ = model(**batch, use_cache=False, return_dict=True)

    contrib = per_head_contributions_from_hooks(
        model=model,
        layer_idx=layer_idx,
        cap=cap,
        t_star=t_star,
        check_sum=False,        # Temporarily disable for debugging
        atol=1e-5,
        rtol=1e-4,
    )  # (B, H, D)

    return contrib

def first_answer_token_ids(tokenizer, answers: List[str]) -> torch.Tensor:
    ids = []
    for a in answers:
        enc = tokenizer(a, add_special_tokens=False, return_tensors="pt")
        # guard against empty strings
        if enc.input_ids.numel() == 0:
            # fall back to EOS if empty
            tok = tokenizer.eos_token_id
        else:
            tok = enc.input_ids[0, 0].item()
        ids.append(tok)
    return torch.tensor(ids, dtype=torch.long)

def discover_all_tasks():
    """Discover and list all available tasks."""
    print("Listing available tasks...")
    tasks = discover_tasks()
    print(f"Found {len(tasks)} tasks:")
    
    task_info = get_task_info()
    for task_name, info in task_info.items():
        print(f"  • {task_name}: {info['class']} - {info['docstring'][:100]}...")
    
    return list(tasks.keys())

# --- token index to use (last non-pad) ---
def _decision_index(attn_mask: torch.Tensor) -> torch.Tensor:
    # attn_mask: (B, T) with 1 for real tokens
    return (attn_mask.sum(dim=1) - 1).clamp(min=0)

# --- prompt sampling from your BaseTask ---
def _sample_task_prompts(task: BaseTask, n: int) -> List[str]:
    rows = task.get_split("test")
    if len(rows) == 0:
        return []
    n = min(n, len(rows))
    # simple: take first n; you can randomize if you like
    prompts = []
    for r in rows[:n]:
        prompts.append(task.build_prompt(r))
    return prompts

def _sample_prompts_and_answers(task, n):
    rows = task.get_split("test")[:n]
    texts = [task.build_prompt(r) for r in rows]
    answers = [r[task.config.output_column] for r in rows]
    return texts, answers
    
# --- simple shuffled controls (permute targets) ---
def get_shuffled_prompts(task: BaseTask, n: int) -> List[str]:
    rows = task.get_split("test")
    n = min(n, len(rows))
    rows = rows[:n]
    
    # Get original inputs and outputs
    inputs = [r[task.config.input_column] for r in rows]
    outputs = [r[task.config.output_column] for r in rows]
    
    # Shuffle the outputs to break input->output mapping
    import random
    shuffled_outputs = outputs.copy()
    random.Random(0).shuffle(shuffled_outputs)
    
    # Build new prompts with shuffled mappings
    ctrl_prompts = []
    for i, row in enumerate(rows):
        # Create varied broken demonstration examples for each prompt
        prompt = ""
        
        # Use different demonstration pairs for each prompt to add variety
        num_demos = min(2, len(inputs))
        demo_start_idx = i % max(1, len(inputs) - 1)  # Rotate starting position
        
        for j in range(num_demos):
            demo_idx = (demo_start_idx + j) % len(inputs)
            shuffled_idx = (demo_idx + 1) % len(shuffled_outputs)  # Offset for more randomness
            
            # Pair input[demo_idx] with shuffled_output[shuffled_idx] to break the pattern
            broken_demo = f"{inputs[demo_idx]} -> {shuffled_outputs[shuffled_idx]}"
            prompt += f"{broken_demo}\n"
        
        # Add the test input
        prompt += f"{row[task.config.input_column]} ->"
        ctrl_prompts.append(prompt)
    
    return ctrl_prompts

@torch.no_grad()
def _first_answer_token_ids(tokenizer, answers: List[str]) -> torch.Tensor:
    """Return a (B,) tensor with the first non-special token id of each answer (EOS if empty)."""
    ids = []
    for a in answers:
        enc = tokenizer(a, add_special_tokens=False, return_tensors="pt")
        if enc.input_ids.numel() == 0:
            tok = tokenizer.eos_token_id
        else:
            tok = enc.input_ids[0, 0].item()
        ids.append(tok)
    return torch.tensor(ids, dtype=torch.long)

@torch.no_grad()
def compute_aie_for_layer(
    model: nn.Module,
    tokenizer,
    icl_texts: List[str],
    icl_answers: List[str],     # NEW: true answers aligned with icl_texts
    ctrl_texts: List[str],
    layer_idx: int,
    device: str = "cuda",
    score_metric: str = "logprob",
) -> torch.Tensor:
    """
    Attention-importance estimate (AIE) per head at a given layer.

    Args:
        model, tokenizer: HF CausalLM + tokenizer (eval mode).
        icl_texts: list of ICL prompts (B items).
        icl_answers: list of gold answers (B items), same order as icl_texts.
        ctrl_texts: control prompts (B items), same length as icl_texts.
        layer_idx: which transformer block to analyze.
        device: torch device.
        score_metric: "logprob" or "margin".

    Returns:
        aie: (H,) tensor with mean score drop when head h is swapped.
    """
    # ---- 0) Prep and sanity checks
    assert len(icl_texts) == len(ctrl_texts) == len(icl_answers), "Batch sizes must match"
    gold_ids = _first_answer_token_ids(tokenizer, icl_answers)               # (B,)

    # We'll need decision indices t* from the ICL batch (for the swapper)
    batch_icl = tokenizer(icl_texts, return_tensors="pt", padding=True, truncation=False).to(device)
    t_star = (batch_icl["attention_mask"].sum(dim=1) - 1).clamp(min=0)       # (B,)

    # ---- 1) Base scores on ICL using the true gold token
    s_icl = _score_batch(model, tokenizer, icl_texts, device, score_metric, gold_ids=gold_ids)  # (B,)

    # ---- 2) Cache pre-projection tensors for ICL and control (B,T,D)
    # We hook the o_proj pre-hook to capture its input ("attn_pre_proj")
    with ResidualCapture(model, layer_idx) as cap_icl:
        _ = model(**batch_icl, use_cache=False, return_dict=True)
    attn_pre_icl = cap_icl.cache["attn_pre_proj"]                             # (B,T,D)
    mask_icl = batch_icl["attention_mask"]                                    # (B,T)

    batch_ctrl = tokenizer(ctrl_texts, return_tensors="pt", padding=True, truncation=False).to(device)
    with ResidualCapture(model, layer_idx) as cap_ctrl:
        _ = model(**batch_ctrl, use_cache=False, return_dict=True)
    attn_pre_ctrl = cap_ctrl.cache["attn_pre_proj"]                           # (B,T,D)
    mask_ctrl = batch_ctrl["attention_mask"]                                  # (B,T)

    # ---- 3) Infer head geometry
    blocks = get_blocks(model)
    block = blocks[layer_idx]
    attn = get_attn(block)
    hidden_size, H, Hd = infer_head_dims(model, block, attn)

    # ---- 4) Loop heads: swap in control head-slice at t* and re-score
    aie = torch.zeros(H, device=device)
    o_proj = get_o_proj(attn)
    if o_proj is None:
        raise RuntimeError("Attention module has no explicit output projection")

    for h in range(H):
        # Build a pre-hook that replaces the h-th head slice (at decision token) with control
        swapper = HeadSwap(attn_pre_ctrl, mask_ctrl, H, Hd).make_hook(h, t_star)
        handle = o_proj.register_forward_pre_hook(swapper)
        try:
            s_swapped = _score_batch(model, tokenizer, icl_texts, device, score_metric, gold_ids=gold_ids)  # (B,)
        finally:
            handle.remove()

        # Mean drop = base - swapped
        aie[h] = (s_icl - s_swapped).mean()

    return aie  # (H,)


def extract_informative_heads(config: ExtractConfig, tasks: List[BaseTask]) -> Headset:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(config.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model_name).to(config.device).eval()

    blocks = get_blocks(model)
    layers = config.layers if config.layers is not None else [len(blocks)-1]

    # screen on a few tasks
    screen_tasks = tasks[: min(8, len(tasks))]
    aie_accum = None
    for t in screen_tasks:
        icl = _sample_task_prompts(t, config.num_samples_per_task)
        ctrl = get_shuffled_prompts(t, config.num_samples_per_task)
        for li in layers:
            aie = compute_aie_for_layer(model, tok, icl, ctrl, li, config.device, config.score_metric)  # (H,)
            aie_np = aie.detach().cpu().numpy()
            if aie_accum is None:
                aie_accum = {(li): aie_np.copy()}
            else:
                aie_accum[li] = aie_accum.get(li, 0) + aie_np

    # pick top-k across selected layer(s)
    topk = config.topk_heads
    heads: List[Tuple[int,int]] = []
    for li in layers:
        scores = aie_accum[li]
        idx = np.argsort(-scores)[:topk].tolist()
        heads += [(li, h) for h in idx]

    return Headset(mode="topk", heads=heads, weights=None)

def extract_task_function_vec(
    task: BaseTask,
    config: ExtractConfig,
    head_set: Headset,
    model: Optional[nn.Module] = None,
    tokenizer: Optional[Any] = None,
) -> TaskFunctionVec:
    torch.manual_seed(config.seed); np.random.seed(config.seed)

    # Load once if not provided (lets you reuse across tasks)
    if model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(config.model_name).to(config.device).eval()
    else:
        assert tokenizer is not None

    # 1) sample prompts (+ optional shuffled)
    prompts = _sample_task_prompts(task, config.num_samples_per_task)
    if len(prompts) == 0:
        raise ValueError(f"No data for task {task.config.name}")

    # 2) collect per-head contributions and average across batch
    # By default: only one layer; if config.layers is None, take the last layer.
    blocks = get_blocks(model)
    layers = config.layers if config.layers is not None else [len(blocks) - 1]

    means_sum = None
    count = 0
    for batch_texts in _batch_iter(prompts, config.batch_size):
        # sum across chosen layers to stay in (B, H, d_model)
        contribs_accum = None
        for li in layers:
            c_li = _batch_per_head_contribs(model, tokenizer, batch_texts, li, config.device)  # (B, H, d)
            contribs_accum = c_li if contribs_accum is None else (contribs_accum + c_li)

        # average over batch → (H, d), then transpose to (d, H)
        # but we want (d, H) means; take mean over B dimension
        B = contribs_accum.shape[0]
        batch_means = contribs_accum.mean(dim=0).transpose(1, 0).contiguous()  # (d, H)

        # accumulate
        m_np = batch_means.detach().cpu().numpy()
        if means_sum is None:
            means_sum = np.zeros_like(m_np, dtype=np.float64)
        means_sum += m_np
        count += 1

    residual_means = (means_sum / count).astype(np.float32)   # (d, H)
    head_means = TaskHeadMeans(task_name=task.config.name, residual_means=residual_means)

    # 3) collapse to function vector via Headset
    return build_function_vec_from_means(head_means, head_set, normalization="l2")

def get_task_head_means(
    task: BaseTask,
    model: Any,
    tokenizer: Any,
    config: ExtractConfig,
    head_set: Headset
) -> TaskHeadMeans:

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    num_prompts = config.num_samples_per_task
    prompts = _sample_task_prompts(task, num_prompts)

    means_sum = None
    count = 0
    for batch in _batch_iter(prompts, config.batch_size):
        contribs = get_contribution_of_attn_head(
            model, tokenizer, batch, head_set, device=config.device)
        if isinstance(contribs, torch.Tensor):
            contribs = contribs.detach().cpu().numpy()
        
        assert contribs.ndim == 3
        B, d_model, _ = contribs.shape

        if means_sum is None:
            means_sum = np.zeros((d_model, head_set.num_heads), dtype=np.float64)
        means_sum += contribs.mean(axis=0)
        count += B

    res_means = (means_sum / count).astype(np.float32)
    return TaskHeadMeans(task_name=task.config.name, residual_means=res_means)

@torch.no_grad()
def per_head_contributions_from_hooks(
    *,
    model: nn.Module,
    layer_idx: int,
    cap: "ResidualCapture",
    t_star: torch.Tensor,           # (B,) last non-pad (or other decision token per sample)
    check_sum: bool = True,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> torch.Tensor:
    """
    Compute per-head contributions to the residual at the decision token using only hooked tensors.
    Returns: (B, H, D)
    """
    assert "attn_pre_proj" in cap.cache, "Enable o_proj pre-hook to capture attn_pre_proj"
    assert "attn_out_proj" in cap.cache, "Enable o_proj fwd-hook to capture attn_out_proj"

    blocks = get_blocks(model)
    block = blocks[layer_idx]
    attn = get_attn(block)
    o_proj = get_o_proj(attn)
    if o_proj is None:
        raise RuntimeError("This attention module has no explicit output projection")

    D, D2 = o_proj.weight.shape
    assert D == D2, "Expected square output projection (D, D)"

    hidden_size, H, Hd = infer_head_dims(model, block, attn)
    assert hidden_size == D, "Projection width must match hidden_size"
    assert D == H * Hd, f"Expected D == H*Hd, got {D} vs {H}*{Hd}"

    # 1) pick decision token and split into heads
    attn_pre = cap.cache["attn_pre_proj"]              # (B, T, D)
    B, T, _ = attn_pre.shape
    device = attn_pre.device
    b_idx = torch.arange(B, device=device)
    O_last = attn_pre[b_idx, t_star, :]                # (B, D)
    O_heads = O_last.view(B, H, Hd)                    # (B, H, Hd)

    # 2) slice o_proj columns per head and project
    W = o_proj.weight                                  # (D, D)
    # Stack per-head column blocks (D, Hd) along H
    Wo_heads = torch.stack([W[:, h*Hd:(h+1)*Hd] for h in range(H)], dim=0)  # (H, D, Hd)

    # batched per-head projection: (B,H,Hd) · (H,D,Hd) -> (B,H,D)
    contrib = torch.einsum("bhd,hDd->bhD", O_heads, Wo_heads)  # (B, H, D)

    if check_sum:
        # Sum over heads should equal the hooked attn_out_proj at t_star
        y_true = cap.cache["attn_out_proj"][b_idx, t_star, :]              # (B, D)
        y_hat = contrib.sum(dim=1)                                         # (B, D)
        if not torch.allclose(y_hat, y_true, atol=atol, rtol=rtol):
            max_err = (y_hat - y_true).abs().max().item()
            raise AssertionError(f"Sum of per-head contributions != attn_out_proj at decision token (max|Δ|={max_err:.3e})")

    return contrib

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

def build_skill_basis(task_vec_matrix: TaskMatrix, method="svd", k=-1, center=False) -> SkillBasis:
    """Build a skill basis from a set of function vectors.
    
    Args:
        task_vec_matrix: Matrix of task function vectors
        method: Method to use ("svd" or "pca")
        k: Number of components to keep (-1 for 95% variance)
        center: Whether to center the data before SVD. 
               Set to False (default) for L2-normalized vectors to use cosine-based metrics.
               Set to True for standard PCA on unnormalized vectors.
    
    Returns:
        SkillBasis with the computed basis
    """
    # NOTE: just svd for now
    V = np.asarray(task_vec_matrix.V, dtype=np.float64)
    
    if center:
        mean = V.mean(axis=1, keepdims=True)
        V_centered = V - mean
    else:
        # For normalized vectors, don't center (cosine-based analysis)
        mean = None
        V_centered = V

    U, S, Vt = np.linalg.svd(V_centered, full_matrices=False)

    if k == -1: # select based on energy
        energy = np.cumsum(S**2) / np.sum(S**2)
        k = int(np.searchsorted(energy, 0.95) + 1)

    U = U[:, :k].astype(np.float32, copy=False)
    S = S[:k].astype(np.float32, copy=False)
    Vt = Vt[:k, :].astype(np.float32, copy=False)
    
    if mean is not None:
        mean = mean.astype(np.float32)

    return SkillBasis(
        method=method, 
        U=U, 
        S=S, 
        Vt=Vt, 
        task_names=task_vec_matrix.task_names,
        mean=mean
    )


def extract_function_vector_simple(
    task_name: str,
    task_config: Optional[TaskConfig] = None,
    model_name: str = "gpt2",
    num_samples: int = 10,
    device: str = "auto"
) -> TaskFunctionVec:
    """
    Simplified one-stop interface for extracting function vectors.
    
    Args:
        task_name: Name of task (e.g., "simple_icl", "math")
        task_config: Optional custom config, will use defaults if None
        model_name: Model to use for extraction
        num_samples: Number of examples to use
        device: Device to run on ("auto", "cuda", "cpu")
        
    Returns:
        TaskFunctionVec with the extracted function vector
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create task
    if task_config is None:
        task_config = TaskConfig(name=task_name)
    task = get_task(task_name, task_config)
    
    # Load model and tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    
    # Simple config
    config = ExtractConfig(
        model_name=model_name,
        device=device,
        num_samples_per_task=num_samples,
        batch_size=min(8, num_samples)
    )
    
    # Auto-select heads from layer 0 (simple approach)
    num_heads = model.config.num_attention_heads
    head_set = Headset(
        mode="topk", 
        heads=[(0, h) for h in range(min(num_heads, 5))]  # Use first 5 heads from layer 0
    )
    
    # Extract function vector
    return extract_task_function_vec(task, config, head_set, model, tokenizer)


if __name__ == "__main__":
    # Discover and list all tasks
    discover_all_tasks()
# fv/resid_capture.py
from typing import Dict, List, Optional
import torch
import torch.nn as nn

def get_blocks(model) -> List[nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)            # LLaMA/Qwen/OPT/OLMo-2
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)         # NeoX/Pythia
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)           # GPT-2/J
    raise RuntimeError("Unsupported model layout")

def get_attn(block: nn.Module) -> nn.Module:
    for name in ("self_attn", "attention", "attn"):
        if hasattr(block, name):
            return getattr(block, name)
    raise RuntimeError(f"No attention module in {type(block)}")

def get_mlp(block: nn.Module) -> nn.Module:
    for name in ("mlp", "feed_forward", "ffn"):
        if hasattr(block, name):
            return getattr(block, name)
    raise RuntimeError(f"No MLP/FFN module in {type(block)}")

def infer_head_dims(model: nn.Module, block: nn.Module, attn: nn.Module):
    # Try attention module first (present for GPT-2, LLaMA, NeoX families)
    if hasattr(attn, "num_heads"):
        num_heads = int(attn.num_heads)
    elif hasattr(model.config, "num_attention_heads"):
        num_heads = int(model.config.num_attention_heads)
    else:
        raise RuntimeError("Cannot infer num_heads")

    if hasattr(model.config, "hidden_size"):
        hidden_size = int(model.config.hidden_size)
    elif hasattr(model.config, "n_embd"):  # GPT-2 / GPT-J
        hidden_size = int(model.config.n_embd)
    else:
        # Fallback: read from a known linear in the block
        # e.g., post-attn norm weight/affine size
        if hasattr(block, "ln_2"):
            hidden_size = block.ln_2.weight.numel()
        elif hasattr(block, "post_attention_layernorm"):
            hidden_size = block.post_attention_layernorm.weight.numel()
        else:
            raise RuntimeError("Cannot infer hidden_size")

    assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
    head_dim = hidden_size // num_heads
    return hidden_size, num_heads, head_dim

def get_post_attn_norm(block: nn.Module) -> nn.Module:
    # GPT-2
    if hasattr(block, "ln_2"):
        return block.ln_2
    # LLaMA/Qwen/OPT/OLMo-2 style
    if hasattr(block, "post_attention_layernorm"):
        return block.post_attention_layernorm
    # Some NeoX variants
    if hasattr(block, "post_attn_layer_norm"):
        return block.post_attn_layer_norm
    raise RuntimeError("Cannot find post-attention layernorm in block")

def get_o_proj(attn: nn.Module) -> Optional[nn.Module]:
    # Optional hook on the attention output projection
    if hasattr(attn, "o_proj"):
        return attn.o_proj          # LLaMA/Qwen/OPT/OLMo-2
    if hasattr(attn, "c_proj"):
        return attn.c_proj          # GPT-2
    if hasattr(attn, "dense"):
        return attn.dense           # NeoX
    return None

class ResidualCapture:
    """Capture residual stream tensors around the attention sublayer for one block."""
    def __init__(self, model: nn.Module, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        blocks = get_blocks(model)
        assert 0 <= layer_idx < len(blocks)
        self.block = blocks[layer_idx]
        self.attn = get_attn(self.block)
        self.mlp = get_mlp(self.block)
        self.o_proj = get_o_proj(self.attn)
        self.post_attn_norm = get_post_attn_norm(self.block)

        self.cache: Dict[str, torch.Tensor] = {}
        self._handles = []

    def _block_pre(self, _m, args):
        self.cache["resid_pre_attn"] = args[0].detach()  # input to block

    def _post_attn_norm_pre(self, _m, args):
        # input to the post-attn layernorm == residual AFTER attention add
        self.cache["resid_after_attn_add"] = args[0].detach()

    def _oproj_fwd(self, _m, _inp, out):
        y = out[0] if isinstance(out, (tuple, list)) else out
        self.cache["attn_out_proj"] = y.detach()

    def _oproj_pre(self, _m, inputs):
        # inputs[0] is (B, T, D) in most HF impls; some collapse to (B*T, D)
        x = inputs[0]
        if x.dim() == 2 and "resid_pre_attn" in self.cache:
            B, T, D = self.cache["resid_pre_attn"].shape
            x = x.view(B, T, D)
        self.cache["attn_pre_proj"] = x.detach()

    def enable(self):
        self._handles.append(self.block.register_forward_pre_hook(self._block_pre))
        self._handles.append(self.post_attn_norm.register_forward_pre_hook(self._post_attn_norm_pre))
        if self.o_proj is not None:
            self._handles.append(self.o_proj.register_forward_pre_hook(self._oproj_pre))   # <-- add this
            self._handles.append(self.o_proj.register_forward_hook(self._oproj_fwd))
        return self

    def disable(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self): return self.enable()
    def __exit__(self, *exc): self.disable()

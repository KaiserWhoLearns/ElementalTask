import torch
import torch.nn as nn

class AttnGetter:
    """Helper class that exposes q_proj, v_proj, o_proj from hf models.
    Currently supports:
    - GPT-2 style models
    - Llama/Qwen/GPT-J style models
    Note: does not support gqa/mqa yet
    """
    def __init__(self, attn_module: nn.Module, hidden_size: int, num_heads: int):
        self.attn_module = attn_module
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_size = hidden_size // num_heads

        self.arch_type = self._detect_arch_type()
        if self.arch_type is None:
            raise ValueError("Unsupported attention module architecture")

    @property
    def device(self):
        return next(self.attn_module.parameters()).device
    
    def _detect_arch_type(self) -> Optional[str]:
        """Detect the architecture type based on the presence of projection layers."""
        if hasattr(self.attn_module, "q_proj") and hasattr(self.attn_module, "v_proj") and hasattr(self.attn_module, "o_proj"):
            self.arch_type = "separate"
        elif hasattr(self.attn_module, "c_attn") and hasattr(self.attn_module, "c_proj"):
            self.arch_type = "gpt2"
        elif hasattr(self.attn_module, "query_key_value") and (hasattr(self.attn_module, "dense") or hasattr(self.attn_module, "o_proj")):
            self.arch_type = "neox"
        else:
            raise RuntimeError("Unsupported attention module type")
    
    def get_v_proj_weight(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.arch_type == "separate":
            W = self.attn_module.v_proj.weight
            b = self.attn_module.v_proj.bias if self.attn_module.v_proj.bias is not None else None
            return W, b
        elif self.flavor == "gpt2":
            # c_attn: (hidden, 3*hidden) -> last third is V
            W_full = self.m.c_attn.weight
            b_full = getattr(self.m.c_attn, "bias", None)
            h = self.hidden_size
            W = W_full[:, 2*h:3*h]
            b = b_full[2*h:3*h] if b_full is not None else None
            return W, b
        else:  # neox
            # query_key_value: (hidden, 3*hidden) -> last third is V
            W_full = self.m.query_key_value.weight
            b_full = getattr(self.m.query_key_value, "bias", None)
            h = self.hidden_size
            W = W_full[:, 2*h:3*h]
            b = b_full[2*h:3*h] if b_full is not None else None
            return W, b

    def get_o_proj_weight(self) -> torch.Tensor:
        if self.arch_type == "separate":
            return self.attn_module.o_proj.weight
        elif self.arch_type == "gpt2":
            return self.attn_module.c_proj.weight
        else:  # neox
            if hasattr(self.attn_module, "o_proj"):
                return self.attn_module.o_proj.weight
            else:
                return self.attn_module.dense.weight

    def get_head_outproj_slice(self, head_index: int) -> torch.Tensor:
        W = self.get_o_proj_weight()
        start = head_index * self.head_size
        end = start + self.head_size
        return W[start:end, :]

    def get_V_states(self, x_in: torch.Tensor) -> torch.Tensor:
        """Get the value states for input x_in of shape (B, T, hidden_size).
        Returns V states of shape (B, T, hidden_size).
        """
        W_v, b_v = self.get_v_proj_weight()
        Vlin = x_in @ W_v.t()
        if b_v is not None:
            Vlin = Vlin + b_v
        B, T, _ = Vlin.shape
        V = Vlin.view(B, T, self.num_heads, self.head_size)
        return V
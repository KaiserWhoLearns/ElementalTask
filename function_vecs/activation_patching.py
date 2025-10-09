import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional

class HeadSwap:
    def __init__(self, attn_pre_proj_src: torch.Tensor, attn_mask_src: torch.Tensor,
                 num_heads: int, head_dim: int):
        self.src = attn_pre_proj_src        # (B,T,D) from the other condition
        self.src_mask = attn_mask_src
        self.H = num_heads
        self.Hd = head_dim

    def make_hook(self, head_index: int, t_star: torch.Tensor):
        # returns a callable used as a pre-hook on o_proj
        def _hook(_m, inputs):
            x = inputs[0]                   # (B,T,D) or (B*T,D)
            if x.dim() == 2:
                B, T, D = self.src.shape
                x = x.view(B, T, D)
            B, T, D = x.shape
            # gather Hd slice for head_index at t_star from src, and write into x
            start = head_index * self.Hd
            end = start + self.Hd
            b_idx = torch.arange(B, device=x.device)
            x[b_idx, t_star, start:end] = self.src[b_idx, t_star, start:end]
            return (x,)  # must return tuple for pre-hook
        return _hook
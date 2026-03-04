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
        # Compute t_star for the source (control) batch
        self.t_star_src = (attn_mask_src.sum(dim=1) - 1).clamp(min=0)  # (B,)

    def make_hook(self, head_index: int, t_star_dst: torch.Tensor):
        """
        Create a hook that swaps head activations.
        
        Args:
            head_index: Which head to swap
            t_star_dst: Decision token indices for the destination (ICL) batch
        
        The hook reads from self.src at t_star_src (control's last token)
        and writes to x at t_star_dst (ICL's last token).
        """
        t_star_src = self.t_star_src  # last token positions in control batch
        
        def _hook(_m, inputs):
            x = inputs[0]                   # (B,T,D) or (B*T,D)
            if x.dim() == 2:
                B, T, D = self.src.shape
                x = x.view(B, T, D)
            B, T, D = x.shape
            dev = x.device
            # gather Hd slice for head_index at t_star_src from src, 
            # and write into x at t_star_dst
            start = head_index * self.Hd
            end = start + self.Hd
            b_idx = torch.arange(B, device=dev)
            # Ensure index tensors and source are on the same device as x
            # (needed for device_map="auto" multi-GPU setups)
            _t_star_dst = t_star_dst.to(dev) if t_star_dst.device != dev else t_star_dst
            _t_star_src = t_star_src.to(dev) if t_star_src.device != dev else t_star_src
            _src = self.src.to(dev) if self.src.device != dev else self.src
            # Read from control at control's decision token, write to ICL at ICL's decision token
            x[b_idx, _t_star_dst, start:end] = _src[b_idx, _t_star_src, start:end]
            return (x,)  # must return tuple for pre-hook
        return _hook
# tests/test_function_vecs_minimal.py
import os
import pytest
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

# Import from your package
from function_vecs.model_internal_getters import ResidualCapture, get_blocks, get_attn, get_o_proj, infer_head_dims
from function_vecs.extract_function_vecs import compute_aie_for_layer

MODEL_NAME = "sshleifer/tiny-gpt2"  # tiny, downloads fast

@pytest.mark.integration
@pytest.mark.skipif(
    "CI" in os.environ and os.environ.get("CI") == "true",
    reason="Skip on CI by default to avoid network downloads."
)
@torch.no_grad()
def test_compute_aie_nonzero_on_tiny_gpt2():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).eval()

    # Sanity: model should look GPT-2-ish (has transformer.h blocks and c_proj inside attention)
    blocks = get_blocks(model)
    assert len(blocks) >= 1
    attn0 = get_attn(blocks[0])
    assert get_o_proj(attn0) is not None

    # Tiny 3-example ICL-style prompts + single-token answers.
    # Use leading spaces so GPT-2 BPE treats them as single tokens (e.g., " blue", " red", " green")
    icl_texts = [
        "Prompt: color sequence red then blue -> Next?\nAnswer:",
        "Prompt: color sequence blue then red -> Next?\nAnswer:",
        "Prompt: primary color starting with g -> Next?\nAnswer:",
    ]
    icl_answers = [" blue", " red", " green"]  # first tokens we’ll score

    # Controls: make prompts “distracting” but keep size identical.
    # Here we just append mismatched references (any strong perturbation is fine).
    ctrl_answers = [" red", " green", " blue"]
    ctrl_texts = [p + f" (Reference:{a})" for p, a in zip(icl_texts, ctrl_answers)]

    # Use last layer for effect; tiny-gpt2 has very few layers
    last_layer = len(blocks) - 1

    # Compute AIE per head at this layer, scoring the true first answer token
    aie = compute_aie_for_layer(
        model=model,
        tokenizer=tok,
        icl_texts=icl_texts,
        icl_answers=icl_answers,   # <- NEW: gold token is derived from these
        ctrl_texts=ctrl_texts,
        layer_idx=last_layer,
        device=device,
        score_metric="logprob",    # "margin" also works; "logprob" is simpler to reason about
    )

    # Shape: (H,)
    num_heads = model.config.num_attention_heads
    assert aie.shape == (num_heads,), f"Unexpected AIE shape {aie.shape}, expected {(num_heads,)}"

    # We expect at least one head to matter (non-zero within tolerance)
    assert torch.any(torch.abs(aie) > 0), "AIE appears to be all zeros—check hooks/controls."

    # Optional: ensure it’s not just numerical noise
    assert torch.any(torch.abs(aie) > 1e-5), f"AIE too tiny across heads: {aie.cpu().numpy()}"

# tests/test_resid_capture_tiny_gpt.py
import os, torch, pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from function_vecs.model_internal_getters import ResidualCapture

MODEL_NAME = "sshleifer/tiny-gpt2"

@pytest.mark.integration
@pytest.mark.skipif("CI" in os.environ and os.environ.get("CI") == "true",
                    reason="Skip network downloads on CI")
def test_residual_capture_matches_attention_add_tiny_gpt2():
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    prompts = [
        "The quick brown fox",
        "The quick brown fox jumps",
        "The quick brown fox jumps over the lazy dog",
    ]
    batch = tok(prompts, return_tensors="pt", padding=True).to(device)
    attn_mask = batch["attention_mask"]
    t_star = (attn_mask.sum(dim=1) - 1).clamp(min=0)

    layer_idx = 0
    with ResidualCapture(model, layer_idx) as cap:
        _ = model(**batch, output_attentions=False, use_cache=False, return_dict=True)

    assert "resid_pre_attn" in cap.cache
    assert "resid_after_attn_add" in cap.cache
    assert "attn_out_proj" in cap.cache

    resid_pre  = cap.cache["resid_pre_attn"]          # (B, T, d)
    resid_after= cap.cache["resid_after_attn_add"]    # (B, T, d) == after attention add (before ln_2)
    attn_proj  = cap.cache["attn_out_proj"]           # (B, T, d)

    # In eval mode (dropout off): after_attn_add = pre_attn + attn_proj
    diff = (resid_after - (resid_pre + attn_proj)).abs()
    assert diff.max().item() < 1e-5, f"Residual mismatch (tokenwise): {diff.max().item():.3e}"

    # Check last non-pad token specifically
    B = resid_pre.size(0)
    idx = torch.arange(B, device=resid_pre.device)
    pre_last   = resid_pre[idx, t_star, :]
    after_last = resid_after[idx, t_star, :]
    proj_last  = attn_proj[idx, t_star, :]
    diff_last = (after_last - (pre_last + proj_last)).abs()
    assert diff_last.max().item() < 1e-5, f"Residual mismatch (last token): {diff_last.max().item():.3e}"

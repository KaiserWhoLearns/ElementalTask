# tests/test_resid_capture_tiny_gpt.py
import os, torch, pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from function_vecs.model_internal_getters import (
    ResidualCapture,
    get_blocks,
    get_attn,
    get_mlp,
    get_o_proj,
    get_post_attn_norm,
    infer_head_dims
)

MODEL_NAME = "sshleifer/tiny-gpt2"
OLMO2_MODEL_NAME = "allenai/OLMo-2-1124-7B"

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


@pytest.mark.integration
@pytest.mark.skipif("CI" in os.environ and os.environ.get("CI") == "true",
                    reason="Skip network downloads on CI")
def test_olmo2_model_getters():
    """Test that all getter functions work with OLMo-2-1124-7B."""
    torch.set_grad_enabled(False)
    device = "cpu"  # Use CPU to avoid OOM on GPU

    print(f"\nLoading {OLMO2_MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        OLMO2_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    ).eval()

    print("Testing getter functions...")

    # Test get_blocks
    blocks = get_blocks(model)
    assert len(blocks) == 32, f"Expected 32 blocks, got {len(blocks)}"
    print(f"✓ get_blocks: Found {len(blocks)} blocks")

    # Test get_attn
    attn = get_attn(blocks[0])
    assert attn is not None, "Attention module should not be None"
    print(f"✓ get_attn: {type(attn).__name__}")

    # Test get_mlp
    mlp = get_mlp(blocks[0])
    assert mlp is not None, "MLP module should not be None"
    print(f"✓ get_mlp: {type(mlp).__name__}")

    # Test get_o_proj
    o_proj = get_o_proj(attn)
    assert o_proj is not None, "o_proj should not be None"
    print(f"✓ get_o_proj: {type(o_proj).__name__}")

    # Test get_post_attn_norm
    post_attn_norm = get_post_attn_norm(blocks[0])
    assert post_attn_norm is not None, "post_attn_norm should not be None"
    print(f"✓ get_post_attn_norm: {type(post_attn_norm).__name__}")

    # Test infer_head_dims
    hidden_size, num_heads, head_dim = infer_head_dims(model, blocks[0], attn)
    assert hidden_size == 4096, f"Expected hidden_size=4096, got {hidden_size}"
    assert num_heads == 32, f"Expected num_heads=32, got {num_heads}"
    assert head_dim == 128, f"Expected head_dim=128, got {head_dim}"
    print(f"✓ infer_head_dims: hidden={hidden_size}, heads={num_heads}, head_dim={head_dim}")

    print(f"✅ All getter functions work with {OLMO2_MODEL_NAME}")


@pytest.mark.integration
@pytest.mark.skipif("CI" in os.environ and os.environ.get("CI") == "true",
                    reason="Skip network downloads on CI")
def test_olmo2_residual_capture():
    """Test ResidualCapture with OLMo-2-1124-7B."""
    torch.set_grad_enabled(False)
    device = "cpu"  # Use CPU to avoid OOM on GPU

    print(f"\nLoading {OLMO2_MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        OLMO2_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    ).eval()

    tok = AutoTokenizer.from_pretrained(OLMO2_MODEL_NAME, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Simple test prompts
    prompts = [
        "The quick brown fox",
        "Hello, world!",
    ]
    batch = tok(prompts, return_tensors="pt", padding=True).to(device)

    layer_idx = 0
    print(f"Testing ResidualCapture on layer {layer_idx}...")

    with ResidualCapture(model, layer_idx) as cap:
        _ = model(**batch, output_attentions=False, use_cache=False, return_dict=True)

    # Check that all expected tensors were captured
    expected_keys = ["resid_pre_attn", "resid_after_attn_add", "attn_out_proj", "attn_pre_proj"]
    for key in expected_keys:
        assert key in cap.cache, f"Missing key: {key}"
        print(f"✓ Captured {key}: shape={cap.cache[key].shape}")

    # Verify residual connection math (with tolerance for float16)
    resid_pre = cap.cache["resid_pre_attn"]
    resid_after = cap.cache["resid_after_attn_add"]
    attn_proj = cap.cache["attn_out_proj"]

    # In eval mode: resid_after = resid_pre + attn_proj
    diff = (resid_after - (resid_pre + attn_proj)).abs()
    max_diff = diff.max().item()
    print(f"Max residual difference: {max_diff:.3e}")

    # More lenient threshold for float16
    assert max_diff < 1e-2, f"Residual mismatch too large: {max_diff:.3e}"

    print(f"✅ ResidualCapture works correctly with {OLMO2_MODEL_NAME}")

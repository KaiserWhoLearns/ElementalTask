# OLMo-2 Model Checkpoints

## Key Differences Between 1B and 7B Models

### Model Names
- **1B Model:** `allenai/OLMo-2-0425-1B` (released April 2025)
- **7B Model:** `allenai/OLMo-2-1124-7B` (released November 2024)

### Training Token Scaling
The two models have different tokens-per-step ratios:
- **1B:** ~2.1B tokens per 1000 steps
- **7B:** ~4.2B tokens per 1000 steps (exactly 2x)

### Checkpoint Comparison

| Step | 1B Tokens | 7B Tokens | Notes |
|------|-----------|-----------|-------|
| 10,000 | 21B | 42B | Early checkpoint |
| 100,000 | 210B | 419B | ~100x more steps |
| 200,000 | 420B | 839B | 2x tokens (1B) vs 2x tokens (7B) |
| 300,000 | 630B | 1,259B | 3x base tokens |
| 400,000 | 839B | 1,678B | 4x base tokens |
| 500,000 | 1,049B | 2,098B | Final checkpoint before main |
| main | ~1.1T | ~2.1T | Final trained model |

## Available Configurations

### 1. `olmo2_checkpoints_1b.json`
Only the 1B model with 12 checkpoints (10k, 50k, 100k, ..., 500k, main)

### 2. `olmo2_checkpoints.json`
Only the 7B model with 12 checkpoints (10k, 50k, 100k, ..., 500k, main)

### 3. `olmo2_1b_7b_checkpoints.json`
Both models with 7 checkpoints each (10k, 100k, 200k, 300k, 400k, 500k, main)
- Faster evaluation for comparing across model sizes
- Selected checkpoints at every 100k steps

## Notes

- All checkpoints use the `stage1-stepXXXXX-tokensYYYB` format
- The `main` branch always comes last and represents the final trained model
- 7B checkpoints are ~14GB each, 1B checkpoints are ~2GB each
- Make sure to set `HF_HOME` to a directory with sufficient space

---

# LLM360 K2-V2 Checkpoints

## Model Overview
- **Model:** `LLM360/K2-V2` ([HuggingFace](https://huggingface.co/LLM360/K2-V2))
- **Parameters:** 70B
- **Architecture:** Decoder-only transformer with grouped-query attention, RMSNorm, 80 layers
- **Vocab Size:** 250,000
- **Pre-training Tokens:** ~12T tokens
- **Pre-training Sequence Length:** 8,192
- **License:** Apache 2.0

## Checkpoint Format
Pretrain checkpoints are stored as branches/tags on the HuggingFace repo with format `base_XXXXXXX` (step number, zero-padded to 7 digits). The final pretrain checkpoint is `base_final`.

All checkpoints: https://huggingface.co/LLM360/K2-V2/tree/base_final

## Available Configuration

### `k2v2_checkpoints.json`
11 checkpoints sampled uniformly across training, from early (step 20k) to final:

| Checkpoint | Step |
|-----------|------|
| `base_0020000` | 20,000 |
| `base_0125000` | 125,000 |
| `base_0265000` | 265,000 |
| `base_0405000` | 405,000 |
| `base_0545000` | 545,000 |
| `base_0685000` | 685,000 |
| `base_0825000` | 825,000 |
| `base_0965000` | 965,000 |
| `base_1105000` | 1,105,000 |
| `base_1245000` | 1,245,000 |
| `base_final` | Final |

## Loading
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("LLM360/K2-V2", revision="base_0720000", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("LLM360/K2-V2")
```

## Notes
- 70B checkpoints are very large (~140GB each); ensure sufficient disk space and set `HF_HOME` appropriately
- The tokenizer is the same across all checkpoints; loading from `main` is sufficient
- This is a base (pretrained) model, not instruction-tuned. For instruction-tuned variant, see `LLM360/K2-V2-Instruct`
- K2-V2 also has mid-training checkpoints (`mid_1_*`, `mid_2_*`, `mid_3_*`, `mid_4_*`) which are not included in the default config

---

# LLM360 CrystalCoder Checkpoints

## Model Overview
- **Model:** `LLM360/CrystalCoder` ([HuggingFace](https://huggingface.co/LLM360/CrystalCoder))
- **Parameters:** 7B
- **Architecture:** GPT-like (LLaMA-7B equivalent) with Maximal Update Parameterization (muP), LayerNorm, Rotary position embeddings on first 25% of hidden dims
- **Vocab Size:** 32,032
- **Pre-training Tokens:** ~1.4T tokens across 3 phases
- **Pre-training Sequence Length:** 2,048
- **License:** Apache 2.0

## Training Phases
CrystalCoder was trained in 3 phases with different data mixes:

| Phase | Data | Tokens | Steps | Cumulative Tokens |
|-------|------|--------|-------|-------------------|
| 1 | SlimPajama (first half) | 345B | 79,721 | 345B |
| 2 | SlimPajama (second half) + StarCoder (2x) | 927B | 214,387 | 1,272B |
| 3 | Python/web data + SlimPajama sample | 110B | 27,728 | 1,382B |

Tokens per step: ~4.3M (phase 1-2), ~4.0M (phase 3)

## Checkpoint Format
Checkpoints are stored as branches on the HuggingFace repo with format `CrystalCoder_phase{N}_checkpoint_{XXXXXX}` (step number, zero-padded to 6 digits). The final checkpoint is `CrystalCoder_phase3_checkpoint_027728` (also available as `main`).

Total available checkpoints: ~120 across all 3 phases.

## Available Configurations

### `crystal_checkpoints.json`
11 checkpoints sampled uniformly across training (by token count):

| Checkpoint | Phase | Cumulative Tokens |
|-----------|-------|-------------------|
| `CrystalCoder_phase1_checkpoint_001500` | 1 | ~6.5B |
| `CrystalCoder_phase1_checkpoint_033000` | 1 | ~143B |
| `CrystalCoder_phase1_checkpoint_064500` | 1 | ~279B |
| `CrystalCoder_phase2_checkpoint_018000` | 2 | ~423B |
| `CrystalCoder_phase2_checkpoint_051000` | 2 | ~565B |
| `CrystalCoder_phase2_checkpoint_081000` | 2 | ~695B |
| `CrystalCoder_phase2_checkpoint_114000` | 2 | ~838B |
| `CrystalCoder_phase2_checkpoint_144000` | 2 | ~967B |
| `CrystalCoder_phase2_checkpoint_174000` | 2 | ~1,097B |
| `CrystalCoder_phase2_checkpoint_207000` | 2 | ~1,239B |
| `CrystalCoder_phase3_checkpoint_027728` | 3 | ~1,382B |

### `crystal_sanity_check.json`
Single final checkpoint for quick testing.

## Loading
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "LLM360/CrystalCoder",
    revision="CrystalCoder_phase1_checkpoint_055500",
    trust_remote_code=True  # REQUIRED for custom muP architecture
)
tokenizer = AutoTokenizer.from_pretrained("LLM360/CrystalCoder", trust_remote_code=True)
```

## Notes
- **`trust_remote_code=True` is mandatory** — CrystalCoder uses a custom architecture with muP modifications
- 7B checkpoints are ~13GB each (3 shards); ensure sufficient disk space and set `HF_HOME` appropriately
- The tokenizer is the same across all checkpoints
- 250 branches total exist on HuggingFace (including older `mdl_phase*_step_*` naming convention — use the `CrystalCoder_phase*_checkpoint_*` naming)

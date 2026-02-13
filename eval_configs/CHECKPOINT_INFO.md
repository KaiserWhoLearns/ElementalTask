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

* Models for analysis (ckpt required)

    - Olmo 1
    - Olmo 2
    - LLM360
    - Emmy's Adhoc ckpts

* Tasks (phase I - sanity check)

    - Exact copying/semantic copying
        - Copy the input:
            Input: xyzabc
    - Arithmetic
    - Synonyms/antonyms
    - Parallel structures/templates
    - Reversal/token ops
        - Reverse the word cat: tac
    - Factual recall
        - facebook/kilt_tasks

* Complex Tasks: How can we construct a complex task with the elemental tasks we proposed?
    - extractiveQA = Understanding + Reasoning + Exact Copy
    - Opendomain QA = Memorization + Understanding + Reasoning + Lingusitics
    - Natural Langauge Inference = Understanding + Reasoning


```
Example usage:
  # List all available checkpoints
  python scripts/measure_ckpt_interp_perf.py --model_id allenai/OLMo-1B-hf --list_checkpoints_only

  # Evaluate 10 uniformly sampled checkpoints
  python scripts/measure_ckpt_interp_perf.py --model_id allenai/OLMo-1B-hf --use_vllm

  # Evaluate specific checkpoints
  python scripts/measure_ckpt_interp_perf.py --model_id allenai/OLMo-1B-hf --checkpoints step10000-tokens42B step50000-tokens210B --use_vllm
```





Usage Examples (Generate images):
```
  # Generate all plot types
  python analysis/plotting.py --csv_path output/olmo2_ckpt_interp_results_firstckpts.csv

  # Generate only performance curves
  python analysis/plotting.py --csv_path
  output/olmo2_ckpt_interp_results.csv --plot_type curves

  # Generate grouped task curves
  python analysis/plotting.py --csv_path
  output/olmo2_ckpt_interp_results.csv --plot_type grouped

  # Custom output directory and figure size
  python analysis/plotting.py --csv_path
  output/olmo2_ckpt_interp_results.csv --output_dir my_plots --figsize
   16 10
```


## Developmental TODOs

* Data Preparation
  * Write everything to `data` dir, with the task name as the sub-directory
  * A universal data format
    * Locally as `HF dataset .jsonl`, with `lm_input` indicate the input to the langauge model, and `reference` be the expected output.
* Model Inference
  * Pass model name, checkpoint
  * Run inference (write to `outputs`) / evaluation
* Data Saving
  * Kaiser made in save directory somewhere

* minor TODO: current version of VLLM is incompatible, need to search backward until we find a good one? (speculative, not sure but seems like the issue is recent) - maybe fix VLLM later for faster generation... (Millicent)

#
Test
```
python models/evaluate_models.py \
  --model_id LLM360/Crystal \
  --max_new_tokens 5 \
  --chkpt main 

python models/evaluate_models.py \
  --task_name FRCT_CV1_ScrambledWords \
  --max_new_tokens 5 \
  --chkpt main 

```


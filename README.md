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


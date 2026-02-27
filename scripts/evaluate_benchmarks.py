"""Evaluate language models on standard NLP benchmarks.

Supported benchmarks:
  arc_easy      — ARC-Easy (allenai/ai2_arc, ARC-Easy)
  arc_challenge — ARC-Challenge (allenai/ai2_arc, ARC-Challenge)
  winogrande    — WinoGrande (allenai/winogrande, winogrande_xl)
  boolq         — BoolQ (google/boolq)
  gsm8k         — GSM8K (openai/gsm8k)

Evaluation modes:
  generative  — model generates text; first token / letter is the prediction
  logprob     — score each answer choice by log-prob; argmax wins (MC tasks only)

Example (single checkpoint):
  python scripts/evaluate_benchmarks.py \\
    --model_id allenai/OLMo-2-1124-7B \\
    --benchmarks arc_challenge boolq \\
    --eval_mode logprob \\
    --output_dir results/benchmarks/olmo2_7b

Example (sweep over checkpoints):
  python scripts/evaluate_benchmarks.py \\
    --model_id EleutherAI/pythia-1b \\
    --checkpoints step1000 step2000 step4000 \\
    --benchmarks arc_easy winogrande \\
    --eval_mode logprob \\
    --output_dir results/benchmarks/pythia_1b \\
    --max_samples 500
"""

import argparse
import csv
import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.getcwd())

import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY: Dict[str, dict] = {
    "arc_easy": dict(
        cls_name="ARCTask",
        kwargs={"config": "ARC-Easy"},
        default_shots=25,
        default_split="test",
    ),
    "arc_challenge": dict(
        cls_name="ARCTask",
        kwargs={"config": "ARC-Challenge"},
        default_shots=25,
        default_split="test",
    ),
    "winogrande": dict(
        cls_name="WinograndeTask",
        kwargs={},
        default_shots=5,
        default_split="validation",
    ),
    "boolq": dict(
        cls_name="BoolQTask",
        kwargs={},
        default_shots=5,
        default_split="validation",
    ),
    "gsm8k": dict(
        cls_name="GSM8KTask",
        kwargs={},
        default_shots=8,
        default_split="test",
    ),
}


def load_task(name: str, eval_mode: str, num_shots: Optional[int] = None):
    """Load a benchmark task by name."""
    from tasks.benchmarks.arc_task import ARCTask
    from tasks.benchmarks.winogrande_task import WinograndeTask
    from tasks.benchmarks.boolq_task import BoolQTask
    from tasks.benchmarks.gsm8k_task import GSM8KTask

    class_map = {
        "ARCTask": ARCTask,
        "WinograndeTask": WinograndeTask,
        "BoolQTask": BoolQTask,
        "GSM8KTask": GSM8KTask,
    }

    info = BENCHMARK_REGISTRY[name]
    cls = class_map[info["cls_name"]]
    k = num_shots if num_shots is not None else info["default_shots"]

    return cls(
        eval_mode=eval_mode,
        num_shots=k,
        split=info["default_split"],
        **info["kwargs"],
    )


# ---------------------------------------------------------------------------
# vLLM helpers
# ---------------------------------------------------------------------------

def load_vllm_model(
    model_id: str,
    checkpoint: Optional[str],
    local_path: Optional[str],
    quantization: Optional[str] = None,
    gpu_memory_utilization: float = 0.90,
    max_model_len: Optional[int] = None,
):
    """Load a vLLM model, optionally at a specific revision/checkpoint."""
    import vllm

    model_path = local_path or model_id
    tp = torch.cuda.device_count()

    kwargs = dict(
        model=model_path,
        tokenizer=model_path,
        revision=checkpoint,
        tensor_parallel_size=max(tp, 1),
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    if quantization:
        kwargs["quantization"] = quantization
        kwargs["dtype"] = "float16"  # quant methods require fp16

    model = vllm.LLM(**kwargs)
    return model


def eval_generative_vllm(model, prompts: List[str], max_new_tokens: int = 50) -> List[str]:
    """Generate completions for a list of prompts using vLLM."""
    import vllm

    params = vllm.SamplingParams(temperature=0, max_tokens=max_new_tokens)
    outputs = model.generate(prompts, params)
    return [o.outputs[0].text for o in outputs]


def eval_logprob_vllm(
    model,
    prompts: List[str],
    choices_per_prompt: List[List[str]],
) -> List[str]:
    """Score each answer choice by log-prob and return the best label.

    Uses vLLM's `logprobs` parameter to get the top-k log-probs at the first
    generated position (the position right after "Answer:"). The choice whose
    token has the highest log-prob wins.

    Args:
        model: vLLM LLM instance
        prompts: list of prompt strings (each ending with "Answer:")
        choices_per_prompt: list of choice-label lists, one per prompt
                            (e.g. [["A", "B", "C", "D"], ...])

    Returns:
        list of predicted labels (one per prompt)
    """
    import vllm

    # We need at least as many logprobs as there are choices
    max_choices = max(len(c) for c in choices_per_prompt)
    params = vllm.SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=max(max_choices + 2, 10),  # return top-10 log-probs
    )

    outputs = model.generate(prompts, params)
    predictions = []

    # Build a tokenizer reference from the model (needed for token lookup)
    try:
        tokenizer = model.get_tokenizer()
    except Exception:
        tokenizer = None

    for output, choices in zip(outputs, choices_per_prompt):
        top_logprobs = output.outputs[0].logprobs  # list of {token_id: logprob, ...}
        if not top_logprobs:
            predictions.append(choices[0])
            continue

        first_pos_lps = top_logprobs[0]  # log-probs dict at position 0 of generation

        best_label = choices[0]
        best_score = float("-inf")

        for label in choices:
            # Try to find the token for " {label}" or "{label}"
            token_str = f" {label}"
            score = float("-inf")

            if tokenizer is not None:
                try:
                    token_ids = tokenizer.encode(token_str, add_special_tokens=False)
                    if token_ids:
                        tid = token_ids[0]
                        # first_pos_lps is a dict of {token_id: Logprob}
                        if tid in first_pos_lps:
                            lp_obj = first_pos_lps[tid]
                            # vLLM Logprob is a named tuple with .logprob
                            score = lp_obj.logprob if hasattr(lp_obj, "logprob") else float(lp_obj)
                except Exception:
                    pass

            # Fallback: scan by token string
            if score == float("-inf"):
                for tid, lp_obj in first_pos_lps.items():
                    lp_str = getattr(lp_obj, "decoded_token", None)
                    if lp_str and lp_str.strip().lower() == label.lower():
                        score = lp_obj.logprob if hasattr(lp_obj, "logprob") else float(lp_obj)
                        break

            if score > best_score:
                best_score = score
                best_label = label

        predictions.append(best_label)

    return predictions


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def truncate_prompts(model, prompts: List[str], reserve_tokens: int = 1) -> Tuple[List[str], int]:
    """Truncate prompts that exceed the model's max context length.

    Keeps the *tail* of the token sequence so the test instance (which appears
    last) is always preserved. ICL shots at the front are silently dropped.

    Args:
        model: vLLM LLM instance (must have get_tokenizer())
        prompts: list of prompt strings
        reserve_tokens: tokens to reserve for generation (default 1 for logprob)

    Returns:
        (truncated_prompts, n_truncated)
    """
    try:
        tokenizer = model.get_tokenizer()
        max_len = model.llm_engine.model_config.max_model_len - reserve_tokens
    except Exception:
        return prompts, 0

    truncated = []
    n_truncated = 0
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        if len(ids) > max_len:
            ids = ids[-max_len:]
            prompt = tokenizer.decode(ids, skip_special_tokens=False)
            n_truncated += 1
        truncated.append(prompt)

    if n_truncated:
        print(f"  Truncated {n_truncated}/{len(prompts)} prompts to fit context ({max_len} tokens)")

    return truncated, n_truncated


def evaluate_benchmark(
    task,
    model,
    eval_mode: str,
    max_samples: Optional[int],
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[float, List[dict]]:
    """Run evaluation on a loaded benchmark task.

    Returns:
        (accuracy, list of per-instance result dicts)
    """
    instances = task.data or []
    if not instances:
        print("  Warning: task has no data — skipping.")
        return 0.0, []

    if max_samples is not None and max_samples < len(instances):
        rng = random.Random(seed)
        instances = rng.sample(instances, max_samples)

    # Build prompts and truncate any that exceed the model's context window
    print(f"  Building {len(instances)} prompts …")
    prompts = [task.build_prompt(inst) for inst in instances]
    prompts, _ = truncate_prompts(model, prompts)

    # Get choices per instance (for logprob mode)
    choices_labels = [
        [c.label for c in task.get_choices(inst)]
        for inst in instances
    ]

    if eval_mode == "logprob" and any(choices_labels):
        print(f"  Scoring choices by log-prob …")
        predictions = eval_logprob_vllm(model, prompts, choices_labels)
    else:
        max_tokens = 50 if eval_mode != "generative_long" else 256
        # GSM8K: allow longer generation for chain-of-thought
        if hasattr(task, "TASK_NAME") and task.TASK_NAME == "gsm8k":
            max_tokens = 256
        print(f"  Generating (max_tokens={max_tokens}) …")
        raw_outputs = eval_generative_vllm(model, prompts, max_new_tokens=max_tokens)
        predictions = [task.normalize_prediction(out) for out in raw_outputs]

    # Score
    results = []
    n_correct = 0
    for inst, pred, prompt in zip(instances, predictions, prompts):
        gold = task.get_correct_label(inst)
        correct = task.check_answer(pred, inst) if eval_mode == "generative" \
            else (pred.strip().upper() == gold.strip().upper())
        if correct:
            n_correct += 1
        results.append({
            "prompt": prompt,
            "prediction": pred,
            "gold": gold,
            "correct": correct,
            "question": task.format_question(inst),
        })

    accuracy = n_correct / len(results) if results else 0.0
    return accuracy, results


def save_results(
    results: List[dict],
    accuracy: float,
    output_dir: str,
    model_name: str,
    benchmark: str,
    checkpoint: Optional[str],
):
    """Save per-instance predictions and accuracy summary to CSV files."""
    ckpt_tag = checkpoint or "final"
    safe_ckpt = ckpt_tag.replace("/", "_").replace(":", "_")
    safe_model = model_name.replace("/", "_")

    bench_dir = Path(output_dir) / benchmark
    bench_dir.mkdir(parents=True, exist_ok=True)

    # Per-instance predictions
    pred_file = bench_dir / f"{safe_model}_{safe_ckpt}_predictions.csv"
    with open(pred_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "prediction", "gold", "correct"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "question": r.get("question", ""),
                "prediction": r.get("prediction", ""),
                "gold": r.get("gold", ""),
                "correct": r.get("correct", False),
            })

    # Accuracy summary (append mode to accumulate multiple checkpoints)
    acc_file = bench_dir / f"{safe_model}_accuracy.csv"
    write_header = not acc_file.exists()
    with open(acc_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "checkpoint", "benchmark", "accuracy", "n_samples"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "model": model_name,
            "checkpoint": ckpt_tag,
            "benchmark": benchmark,
            "accuracy": f"{accuracy:.4f}",
            "n_samples": len(results),
        })

    print(f"  Saved predictions → {pred_file}")
    print(f"  Accuracy appended → {acc_file}")
    return acc_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LMs on standard benchmarks (ARC, WinoGrande, BoolQ, GSM8K)"
    )
    parser.add_argument("--model_id", required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--local_model_path", default=None,
                        help="Optional override local path (e.g. for downloaded models)")
    parser.add_argument("--checkpoints", nargs="*", default=[None],
                        help="Checkpoint revisions to evaluate (default: latest)")
    parser.add_argument("--benchmarks", nargs="+",
                        default=["arc_challenge", "winogrande", "boolq", "gsm8k"],
                        choices=list(BENCHMARK_REGISTRY.keys()),
                        help="Benchmarks to evaluate")
    parser.add_argument("--eval_mode", default="logprob",
                        choices=["generative", "logprob"],
                        help="Evaluation mode (default: logprob)")
    parser.add_argument("--num_shots", type=int, default=None,
                        help="Number of ICL shots (default: benchmark-specific)")
    parser.add_argument("--output_dir", default="results/benchmarks",
                        help="Directory for result files")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Subsample N instances per benchmark (useful for quick testing)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (informational; vLLM batches automatically)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="Override vLLM's max_model_len (default: auto from model config).")
    parser.add_argument("--quantization", default=None,
                        choices=[None, "awq", "gptq", "bitsandbytes", "fp8"],
                        help="vLLM quantization method (e.g. bitsandbytes for 4-bit). "
                             "Requires the model to be pre-quantized for awq/gptq/fp8.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                        help="Fraction of GPU memory vLLM may use (default 0.90). "
                             "Lower this if you get OOM crashes.")

    args = parser.parse_args()

    model_name = os.path.basename(args.local_model_path or args.model_id)

    # Pre-load tasks (once, before model loop)
    print("Loading benchmark tasks …")
    tasks = {}
    for bench in args.benchmarks:
        print(f"  {bench} …")
        try:
            tasks[bench] = load_task(bench, args.eval_mode, args.num_shots)
            print(f"    {len(tasks[bench].data)} eval instances, "
                  f"{len(tasks[bench]._few_shot_pool)} few-shot pool")
        except Exception as e:
            print(f"  ERROR loading {bench}: {e}")

    if not tasks:
        print("No tasks loaded — exiting.")
        sys.exit(1)

    # Evaluate each checkpoint
    all_results = {}  # {benchmark: {checkpoint: accuracy}}

    for checkpoint in args.checkpoints:
        tag = checkpoint or "final"
        print(f"\n=== Checkpoint: {tag} ===")

        # Load model
        print(f"Loading model {args.model_id} @ {tag} …")
        try:
            model = load_vllm_model(
                args.model_id, checkpoint, args.local_model_path,
                quantization=args.quantization,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
            )
        except Exception as e:
            print(f"ERROR loading model: {e}")
            continue

        for bench, task in tasks.items():
            print(f"\n--- {bench} ---")
            try:
                accuracy, results = evaluate_benchmark(
                    task, model, args.eval_mode,
                    args.max_samples, args.batch_size, args.seed
                )
                print(f"  Accuracy: {accuracy:.4f} ({sum(r['correct'] for r in results)}/{len(results)})")
                save_results(results, accuracy, args.output_dir,
                             model_name, bench, checkpoint)
                all_results.setdefault(bench, {})[tag] = accuracy
            except Exception as e:
                import traceback
                print(f"  ERROR evaluating {bench}: {e}")
                traceback.print_exc()

        # Explicitly delete model to free GPU memory before next checkpoint
        del model
        torch.cuda.empty_cache()

    # Print final summary table
    if all_results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        benchmarks = list(all_results.keys())
        checkpoints = list({ckpt for v in all_results.values() for ckpt in v})
        header = f"{'Checkpoint':<20}" + "".join(f"{b:<18}" for b in benchmarks)
        print(header)
        print("-" * len(header))
        for ckpt in sorted(checkpoints):
            row = f"{ckpt:<20}"
            for bench in benchmarks:
                acc = all_results.get(bench, {}).get(ckpt, float("nan"))
                row += f"{acc:<18.4f}"
            print(row)

        # Save JSON summary
        summary_path = Path(args.output_dir) / "summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "model": args.model_id,
                "eval_mode": args.eval_mode,
                "results": all_results,
            }, f, indent=2)
        print(f"\nJSON summary saved to {summary_path}")


if __name__ == "__main__":
    main()

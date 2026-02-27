"""Real benchmark tasks (ARC, WinoGrande, BoolQ, GSM8K, etc.).

These tasks support two evaluation modes:
  - "generative": model generates text; extract answer letter / yes/no / number
  - "logprob":    score each choice by log-probability; pick argmax (more faithful to published numbers)

Choose via the --eval_mode flag in evaluate_benchmarks.py.
"""

from .base_benchmark import BaseBenchmarkTask, MCChoice
from .arc_task import ARCTask
from .winogrande_task import WinograndeTask
from .boolq_task import BoolQTask
from .gsm8k_task import GSM8KTask

__all__ = [
    "BaseBenchmarkTask",
    "MCChoice",
    "ARCTask",
    "WinograndeTask",
    "BoolQTask",
    "GSM8KTask",
]

#!/usr/bin/env python3
"""Check final-eval task completion status across models.

This script reads:
- scripts/slurm_scripts/final/eval_tasks_final_lists.sh
- scripts/slurm_scripts/final/eval_tasks_final_*.sh
- eval_configs/*_0b_1t_main.json

For each final model launcher, it reports which FINAL_TASKS are:
- done:    metrics files >= expected checkpoint count
- partial: 0 < metrics files < expected
- missing: metrics files == 0

It also prints missing task indices and a ready-to-run sbatch command:
  sbatch --array=<missing_indices> <launcher_script>

Optionally, it annotates tasks currently RUNNING/PENDING in SLURM queue.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import glob

ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "scripts" / "slurm_scripts" / "final"
LISTS_FILE = FINAL_DIR / "eval_tasks_final_lists.sh"


@dataclass
class LauncherInfo:
    script_path: Path
    job_name: str
    config_path: Path
    output_base: Path


@dataclass
class TaskStatus:
    task: str
    index: int
    found: int
    expected: int
    status: str  # done | partial | missing
    queue_state: str  # running | pending | none


def parse_bash_quoted_list(content: str, array_name: str) -> List[str]:
    # Matches: ARRAY_NAME=( ... ) and extracts all "..." entries.
    m = re.search(rf"{re.escape(array_name)}\s*=\s*\((.*?)\)", content, re.S)
    if not m:
        raise ValueError(f"Could not find array {array_name} in {LISTS_FILE}")
    block = m.group(1)
    return re.findall(r'"([^"]+)"', block)


def load_final_tasks(include_benchmarks: bool = False) -> List[str]:
    txt = LISTS_FILE.read_text(encoding="utf-8")
    elemental = parse_bash_quoted_list(txt, "ELEMENTAL_TASKS")
    compositional = parse_bash_quoted_list(txt, "COMPOSITIONAL_TASKS")
    if include_benchmarks:
        benchmarks = parse_bash_quoted_list(txt, "BENCHMARK_TASKS")
        return elemental + compositional + benchmarks
    return elemental + compositional


def parse_launcher(script_path: Path) -> LauncherInfo:
    txt = script_path.read_text(encoding="utf-8")

    job_m = re.search(r"^#SBATCH\s+--job-name=(.+)$", txt, re.M)
    cfg_m = re.search(r'^CONFIG="([^"]+)"', txt, re.M)
    out_m = re.search(r'^OUTPUT_BASE="([^"]+)"', txt, re.M)

    if not (job_m and cfg_m and out_m):
        raise ValueError(f"Missing job/config/output in {script_path}")

    job_name = job_m.group(1).strip()
    config_path = Path(cfg_m.group(1)).expanduser()
    output_base = Path(out_m.group(1))
    if not output_base.is_absolute():
        output_base = ROOT / output_base

    return LauncherInfo(
        script_path=script_path,
        job_name=job_name,
        config_path=config_path,
        output_base=output_base,
    )


def expected_checkpoints(config_path: Path) -> int:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return sum(len(v) for v in data.values())


def sanitize_task(task: str) -> str:
    return task.replace(":", "_")


def task_aliases(task: str) -> List[str]:
    """Return canonical and known on-disk aliases for a task name."""
    task_s = sanitize_task(task)
    aliases = [task_s]

    # Some compositional runs are saved under blended_compositions_*.
    if task_s.startswith("compositional_"):
        aliases.append("blended_compositions_" + task_s[len("compositional_"):])

    return aliases


def count_outputs(output_base: Path, task: str) -> int:
    """Count checkpoint outputs for a task across known filename variants.

    Prefer metrics files when available, but fall back to detailed outputs for
    tasks that only emit detailed JSONL files.
    """
    best = 0
    for alias in task_aliases(task):
        metrics_pat = str(output_base / "**" / f"*_{alias}_metrics.json")
        detailed_pat = str(output_base / "**" / f"*_{alias}*_detailed.jsonl")
        best = max(best, len(glob.glob(metrics_pat, recursive=True)))
        best = max(best, len(glob.glob(detailed_pat, recursive=True)))
    return best


def compress_indices(indices: List[int]) -> str:
    if not indices:
        return ""
    parts: List[str] = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = i
    parts.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(parts)


def queue_index_map() -> Dict[str, Dict[int, str]]:
    """Return {job_name: {task_idx: running|pending}} from squeue.

    Looks at JOBID in form <array_jobid>_<task_idx>.
    """
    result: Dict[str, Dict[int, str]] = {}
    try:
        out = subprocess.check_output(
            ["squeue", "-h", "-o", "%j|%i|%T"],
            text=True,
        )
    except Exception:
        return result

    for line in out.splitlines():
        if not line.strip():
            continue
        job_name, job_id, state = line.split("|", 2)

        job_id = job_id.strip()
        indices: List[int] = []

        # Single task form: 6596791_42
        m_single = re.match(r"\d+_(\d+)$", job_id)
        if m_single:
            indices = [int(m_single.group(1))]
        else:
            # Range/list form: 6596792_[0-1,4-6,8,10]
            m_multi = re.match(r"\d+_\[(.+)\]$", job_id)
            if m_multi:
                spec = m_multi.group(1)
                for part in spec.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if "-" in part:
                        a, b = part.split("-", 1)
                        try:
                            start = int(a)
                            end = int(b)
                        except ValueError:
                            continue
                        if start <= end:
                            indices.extend(range(start, end + 1))
                    else:
                        try:
                            indices.append(int(part))
                        except ValueError:
                            continue

        if not indices:
            continue

        state_norm = state.strip().lower()
        if "running" in state_norm:
            s = "running"
        elif "pending" in state_norm:
            s = "pending"
        else:
            continue
        m = result.setdefault(job_name.strip(), {})
        for idx in indices:
            m[idx] = s

    return result


def gather_status(tasks: List[str], launcher: LauncherInfo, queue_map: Dict[str, Dict[int, str]]) -> Tuple[List[TaskStatus], Dict[str, int]]:
    exp = expected_checkpoints(launcher.config_path)
    statuses: List[TaskStatus] = []

    q = queue_map.get(launcher.job_name, {})

    summary = {
        "done": 0,
        "partial": 0,
        "missing": 0,
        "running": 0,
        "pending": 0,
        "expected": exp,
    }

    for i, t in enumerate(tasks):
        found = count_outputs(launcher.output_base, t)
        if found >= exp and exp > 0:
            status = "done"
        elif found > 0:
            status = "partial"
        else:
            status = "missing"

        qstate = q.get(i, "none")

        if qstate == "running":
            summary["running"] += 1
        elif qstate == "pending":
            summary["pending"] += 1

        summary[status] += 1

        statuses.append(TaskStatus(task=t, index=i, found=found, expected=exp, status=status, queue_state=qstate))

    return statuses, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        nargs="*",
        default=None,
        help="Optional filter by launcher basename tokens (e.g., olmo2_7b amber)",
    )
    parser.add_argument(
        "--show-tasks",
        action="store_true",
        help="Print full per-task missing/partial lists for each model.",
    )
    parser.add_argument(
        "--include-benchmarks",
        action="store_true",
        help="Include benchmark tasks in completion accounting (default: off).",
    )
    args = parser.parse_args()

    tasks = load_final_tasks(include_benchmarks=args.include_benchmarks)

    launchers = [p for p in FINAL_DIR.glob("eval_tasks_final_*.sh") if p.name != "eval_tasks_final_lists.sh"]
    launchers = sorted(launchers)

    if args.model:
        needles = [n.lower() for n in args.model]
        launchers = [p for p in launchers if any(n in p.stem.lower() for n in needles)]

    if not launchers:
        print("No final launchers matched.")
        return

    qmap = queue_index_map()

    print(f"Final tasks: {len(tasks)} total")
    print()

    for lp in launchers:
        info = parse_launcher(lp)
        statuses, summary = gather_status(tasks, info, qmap)

        missing = [s for s in statuses if s.status == "missing"]
        partial = [s for s in statuses if s.status == "partial"]
        missing_idxs = [s.index for s in missing]

        print(f"=== {lp.name} ===")
        print(f"job_name: {info.job_name}")
        print(f"output:   {info.output_base}")
        print(f"config:   {info.config_path.name} (expected ckpts/task={summary['expected']})")
        print(
            "status:   "
            f"done={summary['done']}/{len(tasks)}  "
            f"partial={summary['partial']}  "
            f"missing={summary['missing']}  "
            f"running={summary['running']}  "
            f"pending={summary['pending']}"
        )

        if missing_idxs:
            arr = compress_indices(missing_idxs)
            print(f"missing_indices: {arr}")
            print(f"resubmit_cmd:    sbatch --array={arr} {lp}")
        else:
            print("missing_indices: (none)")

        if partial:
            pidx = compress_indices([s.index for s in partial])
            print(f"partial_indices: {pidx}")

        if args.show_tasks:
            if missing:
                print("missing_tasks:")
                for s in missing:
                    print(f"  - [{s.index}] {s.task}")
            if partial:
                print("partial_tasks:")
                for s in partial:
                    print(f"  - [{s.index}] {s.task} ({s.found}/{s.expected})")

        print()


if __name__ == "__main__":
    main()

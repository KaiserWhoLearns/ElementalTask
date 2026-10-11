---
license: cc-by-4.0
language:
- en
- fr
- es
pretty_name: ElementalTask paper artifacts
tags:
- arxiv:2604.08510
- language-model-analysis
- function-vectors
- learning-trajectories
configs:
- config_name: task_examples
  default: true
  data_files:
  - split: train
    path: tasks/examples/*.jsonl
- config_name: trajectories
  data_files:
  - split: train
    path: dataset_views/trajectories.jsonl
- config_name: function_vector_index
  data_files:
  - split: train
    path: dataset_views/function_vector_index.jsonl
dataset_info:
- config_name: task_examples
  features:
  - name: example_id
    dtype: string
  - name: input
    dtype: string
  - name: operations
    dtype: string
  - name: prompt
    dtype: string
  - name: prompt_sha256
    dtype: string
  - name: record_id
    dtype: string
  - name: source_record_index
    dtype: int64
  - name: source_record_position
    dtype: int64
  - name: target
    dtype: string
  - name: task_id
    dtype: string
- config_name: trajectories
  features:
  - name: model
    dtype: string
  - name: checkpoint
    dtype: string
  - name: task_id
    dtype: string
  - name: metric
    dtype: string
  - name: value
    dtype: float64
  - name: status
    dtype: string
  - name: source_files
    dtype: string
- config_name: function_vector_index
  features:
  - name: task_id
    dtype: string
  - name: task_version
    dtype: string
  - name: model
    dtype: string
  - name: model_id
    dtype: string
  - name: checkpoint
    dtype: string
  - name: layer
    dtype: int64
  - name: layer_index_base
    dtype: int64
  - name: representation_type
    dtype: string
  - name: correctness_split
    dtype: string
  - name: num_examples
    dtype: int64
  - name: extraction_seed
    dtype: int64
  - name: dtype
    dtype: string
  - name: dimensions
    dtype: int64
  - name: norm
    dtype: float64
  - name: path
    dtype: string
  - name: sha256
    dtype: string
  - name: source_file
    dtype: string
  - name: extraction_source_revision
    dtype: string
  - name: model_revision_sha
    dtype: string
---

# ElementalTask artifacts, v0.1

Task snapshots, observed evaluation trajectories, and function vectors for
[What Do Language Models Learn and When? The Implicit Curriculum Hypothesis](https://arxiv.org/abs/2604.08510).
Use the BibTeX entry in `CITATION.bib` when citing the paper.

Some third-party task materials are omitted from this release pending
redistribution permission. See the coverage manifest for included tasks.

This release contains **88 tasks, 19,863 reference examples, 38,889 observed
accuracy measurements across 10 models, and 806 function vectors**.

Download the [Hugging Face snapshot](https://huggingface.co/datasets/elemental-tasks/paper-artifacts/tree/v0.1)
or the [GitHub release](https://github.com/KaiserWhoLearns/ElementalTask/releases/tag/artifacts-v0.1).
Individual ZIPs are listed in [DOWNLOADS.md](DOWNLOADS.md).

## Files

- `tasks-v0.1.zip`: frozen reference inputs, complete prompts, saved targets,
  per-task provenance, and current scoring source for reference.
- `trajectories-v0.1.zip`: observed accuracy CSV and a checkpoint/token table.
- `fvs-<model>-v0.1.zip`: one archive per model, with unchanged NumPy arrays,
  original extraction metadata, and an index with checksums.
- `manifest.json`: artifact counts, archive hashes, and hashes of source inputs.
- `task_membership.csv`: appendix, fixed-26, and expanded FV-analysis membership,
  with release availability kept separate.
- `coverage.csv`: task/model coverage, unavailable vectors, and actual FV splits.
- `SHA256SUMS`: hashes of the ZIPs, metadata, and unpacked files.
- `olmo13b_replay.json`: verification of the expanded OLMo-2 13B experiment and
  comparisons with the older predictions under different loading variants.

The appendix suite has 86 explicitly listed tasks; the expanded analysis adds
21 compositions, giving 107 task IDs across the analysis scopes. The manifest
records the included and withheld subsets. This release does not redefine the
paper's task count or claim that every analysis used the same tasks.

## Loading and verification

Python 3.10+ and NumPy are sufficient; model weights and GPU access are unnecessary.

To download the complete versioned snapshot using `huggingface_hub`:

```python
from huggingface_hub import snapshot_download
snapshot_download("elemental-tasks/paper-artifacts", repo_type="dataset",
                  revision="v0.1", local_dir="elementaltask-v0.1")
```

Then run the verifier from the downloaded directory:

```bash
python load_artifacts.py --root . --verify
```

To load an archive directly:

```python
import csv
import io
import json
import zipfile
import numpy as np

with zipfile.ZipFile("tasks-v0.1.zip") as archive:
    examples = [json.loads(line) for line in
                archive.read("examples/basic_arithmetic.jsonl").splitlines()]
    prompt, target = examples[0]["prompt"], examples[0]["target"]

with zipfile.ZipFile("trajectories-v0.1.zip") as archive:
    rows = list(csv.DictReader(io.StringIO(archive.read("trajectories.csv").decode())))
    # Keep missing values missing; zero is a measured accuracy.
    observed = [r for r in rows if r["status"] == "observed"]

with zipfile.ZipFile("fvs-olmo2_13b-v0.1.zip") as archive:
    index = list(csv.DictReader(io.StringIO(archive.read("index.csv").decode())))
    vector = np.load(io.BytesIO(archive.read(index[0]["path"])), allow_pickle=False)
```

## Provenance and limits

Task examples come from one archived evaluation run per task, preferring a final
checkpoint. Complete prompts preserve the demonstrations actually used in that
run. This snapshot does not cover all prompt variants across all model/checkpoint
runs. Per-task metadata flags example counts that differ from the appendix.
Current scoring sources are included for inspection; their historical revisions
were not recorded, and they are not a standalone evaluation package.

Trajectories preserve saved pivot values, including missingness. They are not
smoothed, rounded into token buckets, or reconstructed from raw responses.
Pythia `main` aliases `step143000`, with 299,892,736,000 training tokens; preserve
the raw labels but avoid counting both as independent checkpoints. Other token
positions are labeled estimates or unknown. Immutable model revision hashes
were not recovered. Historical paper analyses also used metrics JSON and other
preprocessing; the pivot export alone does not reproduce that preprocessing.

FV arrays are copied without renormalizing or changing their split. In the
inspected extractor, `residual_stream` means full block output, not selected
attention-head outputs. OLMo-2 13B's saved metadata differs from the paper's
head-selection entry. Some OLMo-2 1B vectors use incorrect or unsplit examples.
These facts are recorded in the indexes. Withheld tasks also have their derived
artifacts excluded from this release, so the bundles do not contain the full
FV prediction basis used in the paper.

The local layer-10 residual vectors reproduce all 44 OLMo-2 13B task scores in
the expanded camera-ready baseline experiment to numerical precision. This check
uses the complete local prediction basis, including tasks withheld here. The
older 26-task predictions are not exactly reproduced: three later basis tasks
and changes to trajectory loading explain much of the difference. With the
95-task basis restored and pivot-only loading, 25 targets differ by 0.022
percentage points on average (maximum 0.141); the remaining target's pivot has
15 checkpoints versus 51 in its saved result. See `olmo13b_replay.json` for the
separate checks and their limits. This verifies the downstream FV set without
establishing why it differs from the paper's head-selection entry.
The `ignoring_context` reference snapshot contains one example, while the appendix
lists five; the release preserves the archived record and flags the difference.

## Licenses

Original project data and artifact documentation: CC BY 4.0; see
`LICENSE-DATA.txt`. Project code: MIT; see `LICENSE-CODE.txt`. IOI examples retain
the upstream MIT license; see `THIRD_PARTY_NOTICES.md`. Existing permissions for
previously distributed versions remain available.

## Rebuilding from the research checkout

The builder reads only the reviewed task allowlist in `release/spec.json` and
checks the inventory hashes before exporting. It requires the archived source
results referenced by that spec. Run from the repository root:

```bash
python scripts/build_release_artifacts.py --output artifacts/v0.1
python release/load_artifacts.py --root artifacts/v0.1 --verify
```

The output directory must not already exist. Use a new path for a rebuild.

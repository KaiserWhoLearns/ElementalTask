#!/usr/bin/env python3
"""Load and verify a local ElementalTask artifact release without model access."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def csv_rows(archive: zipfile.ZipFile, name: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8"))))


def verify(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    allowed = set(manifest["included_tasks"])
    withheld = set(manifest["withheld_tasks"])
    assert allowed.isdisjoint(withheld), "Overlapping included/withheld tasks"
    assert not any(t.startswith(("textfrct", "benchmark")) for t in allowed)
    for line in (root / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        path = (root / name).resolve()
        assert path.is_relative_to(root.resolve()), "Unsafe checksum path"
        assert path.is_file() and digest(path.read_bytes()) == checksum, f"Checksum failed: {name}"
    examples, vectors, observed = 0, 0, 0
    dimensions = {}
    for item in manifest["archives"]:
        path = root / item["path"]
        assert digest(path.read_bytes()) == item["sha256"], f"Archive hash failed: {path.name}"
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None, f"ZIP CRC failed: {path.name}"
            for name in archive.namelist():
                assert not name.startswith("/") and ".." not in Path(name).parts
                assert "textfrct" not in name.lower(), f"Excluded material in archive: {name}"
            if path.name.startswith("tasks-"):
                index = csv_rows(archive, "index.csv")
                assert {r["task_id"] for r in index} == allowed
                record_ids = set()
                for row in index:
                    task = row["task_id"]
                    records = [json.loads(line) for line in archive.read(
                        "examples/" + task.replace(":", "__") + ".jsonl").splitlines()]
                    assert len(records) == int(row["examples"])
                    for record in records:
                        assert record["task_id"] == task
                        assert record["record_id"] not in record_ids
                        record_ids.add(record["record_id"])
                        assert digest(record["prompt"].encode()) == record["prompt_sha256"]
                        assert record["target"] is not None
                        assert not {"prediction", "api_key", "metadata"} & set(record)
                    examples += len(records)
            elif path.name.startswith("trajectories-"):
                rows = csv_rows(archive, "trajectories.csv")
                checkpoints = {(r["model"], r["checkpoint"]): r for r in csv_rows(archive, "checkpoints.csv")}
                seen = set()
                for row in rows:
                    assert row["task_id"] in allowed
                    key = row["model"], row["checkpoint"], row["task_id"]
                    assert key not in seen, f"Duplicate trajectory: {key}"
                    seen.add(key)
                    assert key[:2] in checkpoints
                    if row["status"] == "observed":
                        assert np.isfinite(float(row["value"])) and 0 <= float(row["value"]) <= 1
                        observed += 1
                    else:
                        assert row["status"] == "not_available" and row["value"] == ""
                for checkpoint in checkpoints.values():
                    if checkpoint["model"].startswith("pythia_") and checkpoint["checkpoint"] == "main":
                        assert int(checkpoint["training_tokens"]) == 299_892_736_000
                        assert checkpoint["canonical_checkpoint"] == "step143000"
            elif path.name.startswith("fvs-"):
                seen = set()
                for row in csv_rows(archive, "index.csv"):
                    assert row["task_id"] in allowed and row["task_id"] not in seen
                    seen.add(row["task_id"])
                    data = archive.read(row["path"])
                    assert digest(data) == row["sha256"]
                    vector = np.load(io.BytesIO(data), allow_pickle=False)
                    assert vector.ndim == 1 and len(vector) == int(row["dimensions"])
                    assert str(vector.dtype) == row["dtype"] and np.isfinite(vector).all()
                    assert np.linalg.norm(vector) > 0
                    model = row["model"]
                    dimensions.setdefault(model, len(vector))
                    assert dimensions[model] == len(vector)
                    metadata = json.loads(archive.read("metadata/" + row["task_id"].replace(":", "__") + ".json"))
                    assert metadata["task"] == row["task_id"]
                    assert metadata["representation_type"] == row["representation_type"]
                    assert int(row["layer"]) in metadata["layers"]
                    vectors += 1
    assert examples == manifest["tasks"]["examples"]
    assert observed == manifest["trajectories"]["observed_rows"]
    assert vectors == sum(m["vectors"] for m in manifest["function_vectors"].values())
    return dict(tasks=len(allowed), withheld_tasks=len(withheld), examples=examples,
                observed_trajectory_rows=observed, vectors=vectors, models=len(dimensions),
                archives=len(manifest["archives"]), verification="passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--verify", action="store_true", help="Check all file hashes and load every archived array")
    args = parser.parse_args()
    if args.verify:
        if not __debug__:
            raise RuntimeError("Verification requires Python without -O (assertions enabled)")
        print(json.dumps(verify(args.root), indent=2))
    else:
        manifest = json.loads((args.root / "manifest.json").read_text())
        print(json.dumps({"version": manifest["version"], "tasks": manifest["tasks"],
                          "trajectories": manifest["trajectories"], "function_vectors": manifest["function_vectors"]}, indent=2))


if __name__ == "__main__":
    main()

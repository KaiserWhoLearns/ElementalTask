import pytest
import tasks.frct_fa2_opposites.frct_fa2
from tasks.base_task import get_task

CSV_PATH = "tasks/frct_fa2_opposites/FRCT-LLM_fa2.csv"

def test_get_split_and_fields():
    task = get_task("frct_fa2_opposites", csv_path=CSV_PATH)
    examples = list(task.get_split("test"))

    assert len(examples) > 0
    for ex in examples:
        assert ex["input"].startswith("Think of words with opposite meanings to this word:")
        refs = ex["references"]
        assert isinstance(refs, list) and len(refs) >= 1
        # the legacy output must be one of them
        assert ex["output"] in refs

def test_evaluate_perfect_accuracy():
    task = get_task("frct_fa2_opposites", csv_path=CSV_PATH)
    examples = list(task.get_split("test"))
    # model picks the first opposite every time
    dummy_outputs = [ex["references"][0] for ex in examples]
    metrics = task.evaluate(dummy_outputs, split="test")
    assert metrics["accuracy"] == 1.0

def test_evaluate_accepts_any_opposite():
    task = get_task("frct_fa2_opposites", csv_path=CSV_PATH)
    examples = list(task.get_split("test"))
    # if there's more than one, pick the last; else the first
    dummy_outputs = [
        ex["references"][-1] if len(ex["references"]) > 1 else ex["references"][0]
        for ex in examples
    ]
    metrics = task.evaluate(dummy_outputs, split="test")
    assert metrics["accuracy"] == 1.0

def test_invalid_split_raises():
    task = get_task("frct_fa2_opposites", csv_path=CSV_PATH)
    with pytest.raises(ValueError):
        list(task.get_split("train"))

def test_evaluate_mismatch_length_raises():
    task = get_task("frct_fa2_opposites", csv_path=CSV_PATH)
    examples = list(task.get_split("test"))
    with pytest.raises(ValueError):
        task.evaluate(["foo"] * (len(examples) - 1), split="test")

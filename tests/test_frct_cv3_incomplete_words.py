import pytest
import tasks.frct_cv3_word_completion.frct_cv3
from tasks.base_task import get_task

CSV_PATH = "tasks/frct_cv3_word_completion/FRCT-LLM_cv3.csv"

def test_get_split_and_fields():
    task = get_task("frct_cv3_word_completion", csv_path=CSV_PATH)
    examples = list(task.get_split("test"))

    # basic sanity
    assert len(examples) > 0
    for ex in examples:
        # prompt format
        assert ex["input"].startswith("Complete the English word:")
        # references must exist and output must be among them
        refs = ex["references"]
        assert isinstance(refs, list) and len(refs) >= 1
        assert ex["output"] in refs

def test_evaluate_perfect_accuracy():
    task = get_task("frct_cv3_word_completion", csv_path=CSV_PATH)
    examples = list(task.get_split("test"))
    # model picks the first correct answer
    dummy_outputs = [ex["references"][0] for ex in examples]
    metrics = task.evaluate(dummy_outputs, split="test")
    assert metrics["accuracy"] == 1.0

def test_evaluate_accepts_any_reference():
    task = get_task("frct_cv3_word_completion", csv_path=CSV_PATH)
    examples = list(task.get_split("test"))
    # for entries with multiple refs, pick the second; else use first
    dummy_outputs = [
        ex["references"][1] if len(ex["references"]) > 1 else ex["references"][0]
        for ex in examples
    ]
    metrics = task.evaluate(dummy_outputs, split="test")
    assert metrics["accuracy"] == 1.0

def test_invalid_split_raises():
    task = get_task("frct_cv3_word_completion", csv_path=CSV_PATH)
    with pytest.raises(ValueError):
        list(task.get_split("train"))

def test_evaluate_mismatch_length_raises():
    task = get_task("frct_cv3_word_completion", csv_path=CSV_PATH)
    examples = list(task.get_split("test"))
    # one fewer output than examples
    with pytest.raises(ValueError):
        task.evaluate(["foo"] * (len(examples) - 1), split="test")

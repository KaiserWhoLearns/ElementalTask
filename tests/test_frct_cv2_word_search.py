import pytest
import tasks.frct_cv2_word_search.frct_cv2  # ensure registration
from tasks.base_task import get_task

CSV_PATH = "tasks/frct_cv2_word_search/FRCT-LLM_cv2.csv"

def test_get_split_and_fields():
    task = get_task("frct_cv2_word_search", csv_path=CSV_PATH)
    examples = list(task.get_split("test"))

    assert len(examples) > 0
    for ex in examples:
        assert ex["input"].startswith("Find all the four-letter words hidden in:")
        refs = ex["references"]
        assert isinstance(refs, list) and len(refs) >= 1
        assert ex["output"] in refs

def test_evaluate_perfect_accuracy():
    task = get_task("frct_cv2_word_search", csv_path=CSV_PATH)
    examples = list(task.get_split("test"))
    # simulate model that lists _all_ references, in correct order
    dummy_outputs = [",".join(ex["references"]) for ex in examples]
    metrics = task.evaluate(dummy_outputs, split="test")
    assert metrics["accuracy"] == 1.0

def test_evaluate_order_insensitive():
    task = get_task("frct_cv2_word_search", csv_path=CSV_PATH)
    examples = list(task.get_split("test"))
    # simulate reversed order, still should be 100%
    dummy_outputs = [",".join(reversed(ex["references"])) for ex in examples]
    metrics = task.evaluate(dummy_outputs, split="test")
    assert metrics["accuracy"] == 1.0

def test_invalid_split_raises():
    task = get_task("frct_cv2_word_search", csv_path=CSV_PATH)
    with pytest.raises(ValueError):
        list(task.get_split("train"))

def test_evaluate_partial_incorrect():
    task = get_task("frct_cv2_word_search", csv_path=CSV_PATH)
    examples = list(task.get_split("test"))
    # drop one word from the first example
    partial = examples[0]["references"][:-1]
    dummy_outputs = [",".join(partial)] + [
        ",".join(ex["references"]) for ex in examples[1:]
    ]
    metrics = task.evaluate(dummy_outputs, split="test")
    assert metrics["accuracy"] < 1.0

def test_evaluate_mismatch_length_raises():
    task = get_task("frct_cv2_word_search", csv_path=CSV_PATH)
    examples = list(task.get_split("test"))
    with pytest.raises(ValueError):
        task.evaluate(["foo"] * (len(examples) - 1), split="test")

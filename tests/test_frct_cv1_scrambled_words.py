import pytest
import tasks.frct_cv1_scrambled_words.frct_cv1 
from tasks.base_task import get_task

# adjust path if you moved the CSV somewhere else
CSV_PATH = "tasks/frct_cv1_scrambled_words/FRCT-LLM.csv"

def test_get_split_and_fields():
    task = get_task("frct_cv1_scrambled_words", csv_path=CSV_PATH)
    examples = list(task.get_split("test"))

    # basic sanity
    assert len(examples) > 0
    for ex in examples:
        # prompt format
        assert ex["input"].startswith("Unscramble the word:")
        # references must exist and output must be among them
        refs = ex["references"]
        assert isinstance(refs, list) and len(refs) >= 1
        assert ex["output"] in refs

def test_evaluate_perfect_accuracy():
    task = get_task("frct_cv1_scrambled_words", csv_path=CSV_PATH)
    examples = list(task.get_split("test"))
    # pretend model always picks the first correct answer
    dummy_outputs = [ex["references"][0] for ex in examples]
    metrics = task.evaluate(dummy_outputs, split="test")
    assert metrics["accuracy"] == 1.0

def test_invalid_split_raises():
    task = get_task("frct_cv1_scrambled_words", csv_path=CSV_PATH)
    with pytest.raises(ValueError):
        list(task.get_split("train"))

def test_evaluate_mismatch_length_raises():
    task = get_task("frct_cv1_scrambled_words", csv_path=CSV_PATH)
    examples = list(task.get_split("test"))
    # one fewer output than examples
    with pytest.raises(ValueError):
        task.evaluate(["foo"] * (len(examples) - 1), split="test")

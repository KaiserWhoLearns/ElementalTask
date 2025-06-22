from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Type
from datasets import Dataset

_TASK_REGISTRY: Dict[str, Type["Task"]] = {}

def register_task(cls: Type["Task"]) -> Type["Task"]:
    """
    Register a task class in the global task registry.
    """
    _TASK_REGISTRY[cls.__name__] = cls
    key = getattr(cls, "name", cls.__name__)
    _TASK_REGISTRY[key] = cls
    return cls

def get_task(task_name: str, **kwargs) -> "Task":
    """
    Retrieve a task class from the global task registry.
    """
    if task_name not in _TASK_REGISTRY:
        raise ValueError(f"Task '{task_name}' is not registered.")
    cls = _TASK_REGISTRY[task_name]
    return cls(**kwargs)

class Task(ABC):
    """
    Abstract base class for tasks.
    """
    name: str # unique identifier for the task

    def __init__(self, **config: Any):
        """
        Initialize the task with configuration parameters.
        Subclasses can define their own parameters.
        """
        for k, v in config.items():
            setattr(self, k, v)

    @abstractmethod
    def get_split(self, split: str) -> Dataset:
        """
        Get the dataset split for the task.
        This method should be implemented by subclasses.
        """
        pass

    def evaluate(
        self,
        outputs: List[str],
        split: str = "test",
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Default evaluation: compares `outputs` against
        either `ex["output"]` or membership in `ex.get("references")`.
        Returns a dict of metrics (e.g. {"accuracy": 0.85}).
        """
        examples = list(self.get_split(split))
        if len(outputs) != len(examples):
            raise ValueError("Number of outputs != number of examples")

        correct = 0
        for pred, ex in zip(outputs, examples):
            p = pred.strip()
            # if task provides multiple valid answers
            refs = ex.get("references", None)
            if refs:
                valid = {r.strip() for r in refs}
                if p in valid:
                    correct += 1
            else:
                if p == ex[self.output_key].strip():
                    correct += 1

        acc = correct / len(examples)
        return {"accuracy": acc}
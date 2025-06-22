from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Type
from datasets import Dataset

_TASK_REGISTRY: Dict[str, Type["Task"]] = {}

def register_task(cls: Type["Task"]) -> Type["Task"]:
    """
    Register a task class in the global task registry.
    """
    _TASK_REGISTRY[cls.__name__] = cls
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


    @abstractmethod
    def prepare_data(self) -> None:
        """
        Prepare the data for the task.
        This method should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_split(self, split: str) -> Dataset:
        """
        Get the dataset split for the task.
        This method should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def evaluate(self, model: Any) -> Dict[str, Any]:
        """
        Evaluate the model on the task.
        This method should be implemented by subclasses.
        """
        pass
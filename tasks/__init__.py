"""Task evaluation system with automatic task discovery."""

from .base_task import BaseTask, TaskConfig
from .evaluator import TaskEvaluator, ModelConfig, EvaluationConfig
from .registry import (
    discover_tasks, 
    register_task, 
    get_task_class, 
    get_task, 
    list_tasks, 
    get_task_info
)

# Automatically discover and register all tasks
print("Discovering tasks...")
discover_tasks()

__all__ = [
    "BaseTask", 
    "TaskConfig", 
    "TaskEvaluator", 
    "ModelConfig", 
    "EvaluationConfig",
    "discover_tasks",
    "register_task",
    "get_task_class",
    "get_task",
    "list_tasks",
    "get_task_info"
]

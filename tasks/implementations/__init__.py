"""
This directory contains all task implementations for automatic discovery.

Each task implementation should:
1. Inherit from BaseTask
2. Have a unique class name ending in 'Task'
3. Define TASK_NAME as a class attribute
4. Be importable from its module

Example:
```python
from tasks.base_task import BaseTask, TaskConfig

class MyCustomTask(BaseTask):
    TASK_NAME = "my_custom"  # Will be registered as "my_custom"
    
    def evaluate(self, predictions, split="test", **kwargs):
        # Implementation here
        pass
```

The task registry will automatically discover and register all tasks in this directory.
"""

from .copying_task import CopyingTask
from .ignoring_context_task import IgnoringContextTask
from .string_analogy_task import StringAnalogyTask

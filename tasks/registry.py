"""Automatic task discovery and registration system."""

import os
import importlib
import inspect
from pathlib import Path
from typing import Dict, Type, List
from .base_task import BaseTask, TaskConfig


class TaskRegistry:
    """Automatic task discovery and registration system."""
    
    def __init__(self):
        self._tasks: Dict[str, Type[BaseTask]] = {}
        self._discovered = False
    
    def discover_tasks(self, implementations_path: str = None) -> Dict[str, Type[BaseTask]]:
        """Automatically discover and register tasks from the implementations directory."""
        if self._discovered:
            return self._tasks
        
        if implementations_path is None:
            # Default to tasks/implementations directory
            current_dir = Path(__file__).parent
            implementations_path = current_dir / "implementations"
        else:
            implementations_path = Path(implementations_path)
        
        if not implementations_path.exists():
            print(f"Warning: Task implementations directory not found: {implementations_path}")
            self._discovered = True
            return self._tasks
        
        # Get all Python files in the implementations directory
        python_files = list(implementations_path.glob("*.py"))
        python_files = [f for f in python_files if f.name != "__init__.py"]
        
        discovered_count = 0
        failed_count = 0
        
        for py_file in python_files:
            try:
                # Convert file path to module name
                module_name = f"tasks.implementations.{py_file.stem}"
                
                # Import the module
                module = importlib.import_module(module_name)
                
                # Find all classes that inherit from BaseTask
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseTask) and 
                        obj != BaseTask and 
                        hasattr(obj, 'TASK_NAME')):
                        
                        task_name = obj.TASK_NAME
                        if task_name in self._tasks:
                            print(f"Warning: Task '{task_name}' already registered, skipping duplicate from {py_file.name}")
                        else:
                            self._tasks[task_name] = obj
                            discovered_count += 1
                            print(f"✅ Registered task: '{task_name}' from {py_file.name}")
            
            except ImportError as e:
                print(f"⚠️  Could not import {py_file.name}: {e}")
                failed_count += 1
            except Exception as e:
                print(f"❌ Error processing {py_file.name}: {e}")
                failed_count += 1
        
        self._discovered = True
        print(f"\n📊 Task Discovery Summary:")
        print(f"  ✅ Successfully registered: {discovered_count} tasks")
        if failed_count > 0:
            print(f"  ⚠️  Failed to load: {failed_count} files")
        print(f"  📋 Available tasks: {list(self._tasks.keys())}")
        return self._tasks
    
    def register_task(self, name: str, task_class: Type[BaseTask]) -> None:
        """Manually register a task class."""
        if not issubclass(task_class, BaseTask):
            raise ValueError(f"Task class must inherit from BaseTask")
        
        if name in self._tasks:
            print(f"Warning: Overriding existing task '{name}'")
        
        self._tasks[name] = task_class
        print(f"✅ Manually registered task: '{name}'")
    
    def get_task_class(self, name: str) -> Type[BaseTask]:
        """Get a task class by name."""
        if not self._discovered:
            self.discover_tasks()
        
        if name not in self._tasks:
            available = list(self._tasks.keys())
            raise ValueError(f"Task '{name}' not found. Available tasks: {available}")
        
        return self._tasks[name]
    
    def create_task(self, name: str, spaced: bool = False, *args, **kwargs) -> BaseTask:
        """Create a task instance by name.
        
        If no config/args are provided, tries to use factory functions for proper defaults.
        Falls back to creating a basic TaskConfig if no factory is available.
        
        Args:
            name: Task name, optionally with subtask filter (e.g., 'compositional:upper_reverse')
            spaced: If True, use spaced version of task (adds spaces between characters)
            *args, **kwargs: Additional arguments passed to task constructor
        """
        from .base_task import TaskConfig
        
        # Allow a simple subtask syntax: 'task:sub1,sub2' -> base task with category filters
        base_name = name
        subtask_specs = None
        if isinstance(name, str) and ':' in name:
            base_name, spec = name.split(':', 1)
            subtask_specs = spec  # Keep as string for factory functions

        # If arguments are provided, use them directly (honor explicit kwargs)
        if args or kwargs:
            task_class = self.get_task_class(base_name)
            if not args and 'config' not in kwargs:
                kwargs['config'] = TaskConfig(name=base_name)
            task_instance = task_class(*args, **kwargs)
            # If subtasks were requested, try to apply a simple post-filter
            if subtask_specs:
                try:
                    # Convert string to list for filtering
                    filter_list = [s.strip() for s in subtask_specs.split(',') if s.strip()]
                    import pandas as pd
                    if hasattr(task_instance, 'data') and isinstance(task_instance.data, pd.DataFrame):
                        if 'category_id' in task_instance.data.columns:
                            task_instance.data = task_instance.data[task_instance.data['category_id'].isin(filter_list)]
                        elif 'category_name' in task_instance.data.columns:
                            task_instance.data = task_instance.data[task_instance.data['category_name'].isin(filter_list)]
                    elif hasattr(task_instance, 'data') and isinstance(task_instance.data, list):
                        task_instance.data = [r for r in task_instance.data if r.get('category_id') in filter_list or r.get('category_name') in filter_list]
                except Exception:
                    pass
            return task_instance
        
        # Try to use factory functions for tasks that need special initialization
        factory_map = {
            'copying': lambda: self._import_and_call('tasks.implementations.copying_task', 'make_copying_task', use_generator=True),
            'ignoring_context': lambda: self._import_and_call('tasks.implementations.ignoring_context_task', 'make_ignoring_context_task', use_generator=True),
            'string_analogy': lambda: self._import_and_call('tasks.implementations.string_analogy_task', 'make_string_analogy_task', use_generator=True),
            'basic_arithmetic': lambda: self._import_and_call('tasks.implementations.basic_arithmetic', 'create_basic_arithmetic_task'),
            'ioi_task': lambda: self._import_and_call('tasks.implementations.ioi_task', 'create_ioi_task'),
            'token_reversal': lambda: self._import_and_call('tasks.implementations.token_reversal', 'create_token_reversal_task'),
            'part_of_speech': lambda: self._import_and_call('tasks.implementations.pos_id', 'create_pos_task'),
            'textfrct': lambda: self._create_textfrct_task(subtask_specs),
            'simple_icl': lambda: self._create_simple_icl_task(),
            'simple': lambda: self._create_simple_task(),
            'math': lambda: self._create_math_task(),
            'compositional': lambda: self._create_compositional_task(subtask_specs, spaced=spaced),
        }
        
        if base_name in factory_map:
            try:
                task_instance = factory_map[base_name]()
                # Note: compositional and textfrct handle their own filtering via factory args
                # For other tasks, apply post-filtering if subtasks were requested
                if subtask_specs and base_name not in ('compositional', 'textfrct'):
                    try:
                        filter_list = [s.strip() for s in subtask_specs.split(',') if s.strip()]
                        import pandas as pd
                        if hasattr(task_instance, 'data') and isinstance(task_instance.data, pd.DataFrame):
                            if 'category_id' in task_instance.data.columns:
                                task_instance.data = task_instance.data[task_instance.data['category_id'].isin(filter_list)]
                            elif 'category_name' in task_instance.data.columns:
                                task_instance.data = task_instance.data[task_instance.data['category_name'].isin(filter_list)]
                        elif hasattr(task_instance, 'data') and isinstance(task_instance.data, list):
                            task_instance.data = [r for r in task_instance.data if r.get('category_id') in filter_list or r.get('category_name') in filter_list]
                    except Exception:
                        pass
                return task_instance
            except Exception as e:
                print(f"Warning: Factory function failed for '{name}': {e}")
                print(f"Falling back to default config...")
        
        # Fallback: create with minimal config
        task_class = self.get_task_class(name)
        return task_class(config=TaskConfig(name=name))
    
    def _import_and_call(self, module_name: str, function_name: str, **kwargs):
        """Import a module and call a function from it."""
        import importlib
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        return func(**kwargs)
    
    def _create_simple_icl_task(self):
        """Create a SimpleICLTask with data from simple.csv."""
        config = TaskConfig(
            name="simple_icl",
            description="Simple ICL task with category-based demonstrations",
            data_path="dataset/simple.csv",
            data_format="csv",
            input_column="question",
            output_column="answer",
            evaluation_metrics=["accuracy"],
        )
        task_class = self.get_task_class("simple_icl")
        return task_class(config)
    
    def _create_simple_task(self):
        """Create a SimpleTask with data from simple.csv."""
        config = TaskConfig(
            name="simple",
            description="Simple task with exact match evaluation",
            data_path="dataset/simple.csv",
            data_format="csv",
            input_column="question",
            output_column="answer",
            evaluation_metrics=["accuracy"],
        )
        task_class = self.get_task_class("simple")
        return task_class(config)
    
    def _create_math_task(self):
        """Create a MathTask (generates synthetic data)."""
        config = TaskConfig(
            name="math",
            description="Math task with synthetic arithmetic problems",
            data_format="generator",  # Indicates synthetic/generated data
            input_column="input",
            output_column="output",
            evaluation_metrics=["accuracy"],
        )
        task_class = self.get_task_class("math")
        return task_class(config)
    
    def _create_compositional_task(self, category_filter=None, spaced=False):
        """Create a CompositionalTask with optional category filtering and spacing.
        
        Supports:
            - compositional (all categories)
            - compositional:upper_reverse (single category)
            - compositional:upper_reverse,lower_first (multiple categories)
            - spaced=True for character-spaced input/output
        """
        task_name = "compositional"
        if category_filter:
            task_name = f"compositional:{category_filter}"
        if spaced:
            task_name += "_spaced"
        
        config = TaskConfig(
            name=task_name,
            description="Compositional task with chained string operations",
            data_path="dataset/compositional_spaced.csv" if spaced else "dataset/compositional.csv",
            data_format="csv",
            input_column="input",
            output_column="output",
            evaluation_metrics=["accuracy"],
        )
        task_class = self.get_task_class("compositional")
        task = task_class(config, spaced=spaced)
        
        # Filter by category if specified
        if category_filter:
            categories = [c.strip() for c in category_filter.split(",")]
            if hasattr(task, 'data') and task.data:
                if isinstance(task.data, list):
                    task.data = [d for d in task.data if d.get('category_name') in categories]
        
        return task
    
    def _create_textfrct_task(self, category_filter=None):
        """Create a TextFRCT task with optional category filtering.
        
        Supports:
            - textfrct (all categories)
            - textfrct:CV1 (single category)
            - textfrct:CV1,CV2,FA1 (multiple categories)
        """
        from tasks.implementations.textfrct_task import TextFRCTTask
        
        # Parse category filter (can be single category or comma-separated list)
        categories = None
        if category_filter:
            if isinstance(category_filter, list):
                categories = category_filter
            else:
                categories = [c.strip() for c in str(category_filter).split(",")]
        
        task_name = f"textfrct:{','.join(categories)}" if categories else "textfrct"
        
        config = TaskConfig(
            name=task_name,
            description="TextFRCT evaluation dataset",
            data_path="dataset/TextFRCT.csv",
            data_format="csv",
            input_column="question",
            output_column="answer",
            evaluation_metrics=["accuracy"],
        )
        
        task = TextFRCTTask(config, skip_subjective=True, categories=categories)
        
        # Filter data by category_id if categories specified
        if categories and hasattr(task, 'data') and task.data:
            if isinstance(task.data, list):
                task.data = [d for d in task.data if d.get('category_id') in categories]
            
        return task
    
    def list_tasks(self) -> List[str]:
        """List all available task names."""
        if not self._discovered:
            self.discover_tasks()
        return list(self._tasks.keys())
    
    def get_task_info(self, name: str = None) -> Dict:
        """Get information about tasks."""
        if not self._discovered:
            self.discover_tasks()
        
        if name is None:
            # Return info for all tasks
            return {
                task_name: {
                    "class": task_class.__name__,
                    "module": task_class.__module__,
                    "docstring": task_class.__doc__ or "No description available"
                }
                for task_name, task_class in self._tasks.items()
            }
        else:
            # Return info for specific task
            if name not in self._tasks:
                raise ValueError(f"Task '{name}' not found")
            
            task_class = self._tasks[name]
            return {
                "name": name,
                "class": task_class.__name__,
                "module": task_class.__module__,
                "docstring": task_class.__doc__ or "No description available",
                "methods": [method for method in dir(task_class) if not method.startswith('_')]
            }


# Global registry instance
_task_registry = TaskRegistry()

# Public API functions
def discover_tasks(implementations_path: str = None) -> Dict[str, Type[BaseTask]]:
    """Discover and register all tasks from the implementations directory."""
    return _task_registry.discover_tasks(implementations_path)

def register_task(name: str, task_class: Type[BaseTask]) -> None:
    """Manually register a task class."""
    _task_registry.register_task(name, task_class)

def get_task_class(name: str) -> Type[BaseTask]:
    """Get a task class by name."""
    return _task_registry.get_task_class(name)

def get_task(name: str, spaced: bool = False, *args, **kwargs) -> BaseTask:
    """Create a task instance by name.
    
    Args:
        name: Task name, optionally with subtask filter (e.g., 'compositional:upper_reverse')
        spaced: If True, use spaced version (spaces between characters in input/output)
        *args, **kwargs: Additional arguments passed to task constructor
    
    Returns:
        Task instance
    
    Examples:
        >>> task = get_task("compositional:upper_reverse")
        >>> task = get_task("compositional:upper_reverse", spaced=True)
        >>> task = get_task("textfrct:CV1")
    """
    return _task_registry.create_task(name, spaced=spaced, *args, **kwargs)

def list_tasks() -> List[str]:
    """List all available task names."""
    return _task_registry.list_tasks()

def get_task_info(name: str = None) -> Dict:
    """Get information about tasks."""
    return _task_registry.get_task_info(name)

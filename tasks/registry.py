"""Automatic task discovery and registration system."""

import os
import importlib
import inspect
from pathlib import Path
from typing import Dict, Type, List
from .base_task import BaseTask


class TaskRegistry:
    """Automatic task discovery and registration system."""
    
    def __init__(self):
        self._tasks: Dict[str, Type[BaseTask]] = {}
        self._discovered = False
    
    def discover_tasks(self, implementations_path: str = None) -> None:
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
            return
        
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
    
    def create_task(self, name: str, *args, **kwargs) -> BaseTask:
        """Create a task instance by name."""
        task_class = self.get_task_class(name)
        return task_class(*args, **kwargs)
    
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
def discover_tasks(implementations_path: str = None) -> None:
    """Discover and register all tasks from the implementations directory."""
    _task_registry.discover_tasks(implementations_path)

def register_task(name: str, task_class: Type[BaseTask]) -> None:
    """Manually register a task class."""
    _task_registry.register_task(name, task_class)

def get_task_class(name: str) -> Type[BaseTask]:
    """Get a task class by name."""
    return _task_registry.get_task_class(name)

def get_task(name: str, *args, **kwargs) -> BaseTask:
    """Create a task instance by name."""
    return _task_registry.create_task(name, *args, **kwargs)

def list_tasks() -> List[str]:
    """List all available task names."""
    return _task_registry.list_tasks()

def get_task_info(name: str = None) -> Dict:
    """Get information about tasks."""
    return _task_registry.get_task_info(name)

"""
Function vector extraction package for ICL examples.

This package contains utilities for extracting function vectors from 
in-context learning examples across different tasks.
"""

from .extract_function_vecs import discover_all_tasks

__all__ = ['discover_all_tasks']
"""Demonstration slots; third-party examples are omitted pending permission."""

from .questions import VANILLA_QUESTIONS

# Supply your own authorized examples when running the optional external task.
# Empty defaults preserve the prompt builder's category lookup without sharing items.
VANILLA_DEMONSTRATIONS = {category: "" for category in VANILLA_QUESTIONS}

import os
import time

from typing import Callable

def get_project_root(project_name:str) -> str:
    """Returns project root folder."""
    path = os.path.dirname(os.path.abspath(__file__))
    while project_name != os.path.basename(path):
        path = os.path.dirname(path)
    print(f"Project root is: {path}")
    return path

def print_time(func: Callable) -> Callable:
    """Prints time taken by a function."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time taken by {func.__name__}: {end-start}")
        return result
    return wrapper
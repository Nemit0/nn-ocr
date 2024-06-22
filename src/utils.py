import os
import time
import torch

import pandas as pd
import numpy as np

from typing import Callable, Iterable
from torch import tensor
from rich import print

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

def set_deivce() -> None:
    if torch.cuda.is_available():
        print("[bold green]CUDA is available[/bold green]")
        torch.set_default_device("cuda:0")
    else:
        print("[bold red]CUDA is not available[/bold red]")
        torch.set_default_device("cpu")

def train_test_split(X:Iterable, Y,split:Iterable, random_state:int=1) -> tuple:
	torch.random.manual_seed(random_state)
	l = len(X)
	limit = int(np.floor(split*l))
	if isinstance(X, pd.DataFrame):
		X = tensor(X.values)
		Y = tensor(Y.values)
	
	indices = torch.randperm(l)
	X, Y = X[indices], Y[indices]
	x_train = X[0:limit]
	y_train = Y[0:limit]
	x_test = X[limit:l]
	y_test = Y[limit:l]
	return x_train,y_train,x_test,y_test
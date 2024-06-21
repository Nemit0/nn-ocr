import torch
import os

import pandas as pd
import numpy as np

from rich import print
from torch import tensor

from src.utils import get_project_root, print_time

@print_time
def main():
    project_root = get_project_root("nn-ocr")
    data_path = os.path.join(project_root, "data", "digit-recognizer")
    if torch.cuda.is_available():
        print("[bold green]CUDA is available[/bold green]")
        main_device = torch.device("cuda:0")
    else:
        print("[bold red]CUDA is not available[/bold red]")
        main_device = torch.device("cpu")

    # Load data
    df_train = pd.read_csv(os.path.join(data_path, "train.csv"))
    df_test = pd.read_csv(os.path.join(data_path, "test.csv"))

    print(df_train.shape)
    print(df_test.shape)
    print(df_train.head())

    x_train, y_train = tensor(df_train.iloc[:, 1:].values, device=main_device), tensor(df_train.iloc[:, 0].values, device=main_device)
    x_test, y_test = tensor(df_test.values, device=main_device), tensor(df_test.values, device=main_device)

    print(x_train.shape, y_train.shape)    

if __name__ == "__main__":
    main()
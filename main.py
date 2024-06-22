import torch
import os
import pandas as pd
import numpy as np
from rich import print
from torch import tensor

from src.neural_network import NeuralNetwork
from src.utils import get_project_root, print_time

@print_time
def main():
    project_root = get_project_root("nn-ocr")
    data_path = os.path.join(project_root, "data", "digit-recognizer")

    if torch.cuda.is_available():
        print("[bold green]CUDA is available[/bold green]")
        torch.set_default_device("cuda:0")
    else:
        print("[bold red]CUDA is not available[/bold red]")
        torch.set_default_device("cpu")

    # Load data
    df_train = pd.read_csv(os.path.join(data_path, "train.csv"))
    df_test = pd.read_csv(os.path.join(data_path, "test.csv"))

    print(df_train.shape)
    print(df_test.shape)
    print(df_train.head())

    # Normalize data (pixels are in range 0-255, we normalize to 0-1)
    x_train = df_train.iloc[:, 1:].values / 255.0
    y_train = df_train.iloc[:, 0].values

    # Test data (assuming test.csv has no labels)
    x_test = df_test.values / 255.0

    # Convert to torch tensors
    x_train = tensor(x_train).float()
    y_train = tensor(y_train).long()
    x_test = tensor(x_test).float()

    # Convert labels to one-hot encoding for training
    y_train_one_hot = torch.nn.functional.one_hot(y_train, num_classes=10).float()

    # Initialize neural network
    nn = NeuralNetwork(784, [128, 64], 10)

    # Train neural network
    nn.train(x_train, y_train_one_hot, epoch=10, learning_rate=0.01)

    # Predict using the trained model
    predictions = [torch.argmax(nn.feedforward(x)[-1]).item() for x in x_test]
    
    # For evaluation purposes, if you had a test set with labels:
    # Assuming y_test is the true labels, which you don't have in Kaggle test set, so this part would be for validation set
    # accuracy = nn.test(x_test, y_test)

    # Save predictions 
    submission_df = pd.DataFrame({'ImageId': np.arange(1, len(predictions) + 1), 'Label': predictions})
    submission_df.to_csv(os.path.join(data_path, "prediction.csv"), index=False)
    print("Dataframe Saved")

if __name__ == "__main__":
    main()

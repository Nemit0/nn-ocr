import torch
import pandas as pd
import numpy as np

from typing import List, Callable
from torch import tensor

from src.neural_network import NeuralNetwork

def main():
    input = torch.randn(10, 5)
    target = torch.randint(0, 5, (10,))
    target = torch.eye(5)[target]

    nn = NeuralNetwork(5, [10, 10], 5)
    nn.train(input, target, 1000, 0.01)
    print(nn.predict(input[0]))
    print(target[0])
    print(nn)
    

if __name__ == "__main__":
    main()
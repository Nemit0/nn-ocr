import torch
import time
import sys

import matplotlib.pyplot as plt

from torch import tensor
from typing import List, Callable

from .optimizers import *

class Layer(object):
    def __init__(self, input_dim: int, output_dim: int, activation: Callable) -> None:
        """
        Assuming input dimension is n, output dimension m, dataset size t
        Input: t x n
        Weights: n x m
        Biases : t x m
        Feedforward calculation: X * W + B
        activation: Activation function to be applied to the output of the layer
        """
        self.input_dim:int = input_dim
        self.output_dim:int = output_dim
        self.activation:Callable = activation
        self.weights:tensor = torch.randn(input_dim, output_dim) * (2 / output_dim)**0.5
        self.biases:tensor = torch.zeros(output_dim)

    def feedforward(self, X: tensor) -> tensor:
        z = torch.matmul(X, self.weights) + tensor([self.biases] * X.size(0))
        return self.activation(z)

    def __repr__(self) -> str:
        return f"Layer: {self.input_dim} -> {self.output_dim}"

def relu(x: tensor) -> tensor:
    return torch.maximum(x, tensor(0.0))

def softmax(x: tensor) -> tensor:
    _exp = torch.exp(x - torch.max(x))
    return _exp / torch.sum(_exp)

def cross_entropy_loss(output: tensor, target: tensor) -> tensor:
    return -torch.sum(target * torch.log(output + 1e-9)) / target.size(0)

def d_cross_entropy_loss(output: tensor, target: tensor) -> tensor:
    return -target / (output + 1e-9)

def backprop(network: List[tensor], y: tensor, layers: List[Layer], learning_rate: float):
    # Compute loss gradient
    loss_grad = network[-1] - y
    
    # Compute deltas for each layer
    deltas = [loss_grad]
    for i in reversed(range(len(layers) - 1)):
        delta = torch.matmul(layers[i + 1].weights.T, deltas[0])
        delta = delta * (network[i + 1] > 0).float()  # Derivative of ReLU
        deltas.insert(0, delta)

    # Update weights and biases
    for i, layer in enumerate(layers):
        layer.weights -= learning_rate * torch.outer(deltas[i], network[i])
        layer.biases -= learning_rate * deltas[i]

def mse_loss(pred: tensor, target: tensor) -> tensor:
    return torch.sum((pred - target)**2) / pred.size(0)

def rmse_loss(pred: tensor, target: tensor) -> tensor:
    return torch.sqrt(mse_loss(pred, target))

def analyse_net(layers: List[Layer], X: List[tensor], Y: List[int]) -> float:
    prediction = []
    correct_pred = 0
    for i in range(len(X)):
        y_pred = torch.argmax(feedforward(X[i], layers)[-1])
        prediction.append(y_pred)
        if y_pred == Y[i]:
            correct_pred += 1
    loss = rmse_loss(torch.tensor(prediction), torch.tensor(Y))
    acc = correct_pred / len(X)
    return acc, loss

def train(x_train: List[tensor], y_train: List[tensor], layers: List[Layer], epoch: int, learning_rate: float, verbose:bool=False) -> List[Layer]:
    for iteration in range(epoch):
        t0 = time.perf_counter()
        for i in range(len(x_train)):
            network = feedforward(x_train[i], layers)
            backprop(network, y_train[i], layers, learning_rate)
        acc, loss = analyse_net(layers, x_train, [torch.argmax(y) for y in y_train])
        if verbose:
            sys.stdout.flush()
            progress = int((iteration + 1) / epoch * 40)
            sys.stdout.write(f"\rEpoch {iteration + 1}, Loss {loss:.4f}, Time {time.perf_counter() - t0:.4f}, estimated remaining time: {(time.perf_counter() - t0) * (epoch - iteration - 1):.4f}\n[{'#' * progress}{'*' * (40 - progress)}]")
    if verbose:
        print(f"Training Data Accuracy {analyse_net(layers, x_train, [torch.argmax(y) for y in y_train])}")
    return layers

def test_accuracy(x_test: List[tensor], y_test: List[int], layers: List[Layer]):
    print(f"Test Data Accuracy {analyse_net(layers, x_test, y_test):.4f}")

class NeuralNetwork:
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int, optimizer:object=None, verbose:bool=False):
        # Initialize layers
        self.layers = [Layer(input_dim, hidden_layers[0], relu)]
        for i in range(len(hidden_layers) - 1):
            self.layers.append(Layer(hidden_layers[i], hidden_layers[i + 1], relu))
        self.layers.append(Layer(hidden_layers[-1], output_dim, softmax))
        self.verb = verbose
        self.optimizer = optimizer if optimizer else SGD()

    def feedforward(self, x: tensor) -> List[tensor]:
        for layer in self.layers:
            x = layer.feedforward(x)
        return x
    
    def predict(self, x: tensor) -> tensor:
        return torch.argmax(self.feedforward(x))
        
    def train(self, x_train: tensor, y_train: tensor, epoch: int, learning_rate: float):
        for iteration in range(epoch):
            progress = int((iteration + 1) / epoch * 40)
            if self.verb
            sys.stdout.write(f"[{'#' * progress}{'*' * (40 - progress)}] Epoch: {iteration + 1}\r")

    def test(self, x_test: List[tensor], y_test: List[int]):
        test_accuracy(x_test, y_test, self.layers)
    
    def predict(self, x: tensor) -> int:
        return torch.argmax(self.feedforward(x)[-1]).item()

    def __repr__(self):
        return f"Neural Network:\nInput Dimension: {self.layers[0].input_dim}\nOutput Dimension: {self.layers[-1].output_dim}\nHidden Layers: {[layer.output_dim for layer in self.layers[1:-1]]}"

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available")
        main_device = torch.device("cuda:0")
    else:
        print("CUDA is not available")
        main_device = torch.device("cpu")
    
    nn = NeuralNetwork(50, [128, 64], 10)
    
    # Generate synthetic training and testing data
    x_train = [torch.rand(50) for _ in range(1000)]
    y_train = [torch.eye(10)[torch.randint(0, 10, (1,)).item()] for _ in range(1000)]
    
    x_test = [torch.rand(50) for _ in range(200)]
    y_test = [torch.randint(0, 10, (1,)).item() for _ in range(200)]
    
    nn.train(x_train, y_train, 10, 0.01)
    nn.test(x_test, y_test)
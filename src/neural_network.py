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
    
    def backward(self, output_gradient: tensor) -> tensor:
        input_gradient = torch.matmul(output_gradient, self.weights.T)
        weights_gradient = torch.matmul(self.input.T, output_gradient)
        biases_gradient = output_gradient.sum(0)

        return input_gradient, weights_gradient, biases_gradient

    def update_params(self, delta_weights: tensor, delta_biases: tensor, learning_rate: float):
        self.weights -= learning_rate * delta_weights
        self.biases -= learning_rate * delta_biases

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

def mse_loss(pred: tensor, target: tensor) -> tensor:
    return torch.sum((pred - target)**2) / pred.size(0)

def rmse_loss(pred: tensor, target: tensor) -> tensor:
    return torch.sqrt(mse_loss(pred, target))

class NeuralNetwork:
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int, optimizer:object=None, verbose:bool=False):
        # Initialize layers
        self.layers = [Layer(input_dim, hidden_layers[0], relu)]
        for i in range(len(hidden_layers) - 1):
            self.layers.append(Layer(hidden_layers[i], hidden_layers[i + 1], relu))
        self.layers.append(Layer(hidden_layers[-1], output_dim, softmax))
        self.verb = verbose
        # self.optimizer = optimizer if optimizer else SGD()

    def feedforward(self, x: tensor) -> List[tensor]:
        for layer in self.layers:
            x = layer.feedforward(x)
        return x
    
    def predict(self, x: tensor) -> tensor:
        return torch.argmax(self.feedforward(x))
    
    def backpropagate(self, y: tensor, output_activations: List[tensor], learning_rate: float):
        # Start with the gradient from loss function
        output_gradient = 2 * (output_activations[-1] - y) / y.size(0)  # MSE derivative

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            current_activation = output_activations[i]
            if i > 0:
                prev_activation = output_activations[i-1]
            else:
                prev_activation = None  # For input layer
            
            output_gradient, delta_weights, delta_biases = layer.backward(output_gradient)
            layer.update_params(delta_weights, delta_biases, learning_rate)
        
    def train(self, x_train: tensor, y_train: tensor, epochs: int, learning_rate: float):
        for epoch in range(epochs):
            output_activations = self.feedforward(x_train)
            self.backpropagate(y_train, output_activations, learning_rate)
            if epoch % 100 == 0:
                loss = mse_loss(output_activations[-1], y_train)
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    def test(self, x_test:tensor, y_test:tensor):
        raise NotImplementedError
    
    def predict(self, x: tensor) -> int:
        return torch.argmax(self.feedforward(x)[-1]).item()

    def __repr__(self):
        return f"Neural Network:\nInput Dimension: {self.layers[0].input_dim}\nOutput Dimension: {self.layers[-1].output_dim}\nHidden Layers: {[layer.output_dim for layer in self.layers[1:-1]]}"
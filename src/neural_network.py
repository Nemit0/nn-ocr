import torch
import time
from torch import tensor
from typing import Iterable, Callable, List

class Layer:
    def __init__(self, input_dim: int, output_dim: int, activation: Callable):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.weights = torch.rand(output_dim, input_dim) - 0.5
        self.biases = torch.rand(output_dim) - 0.5

def relu(input_layer: tensor) -> tensor:
    return torch.maximum(input_layer, tensor(0.0))

def softmax(input_layer: tensor) -> tensor:
    _exp = torch.exp(input_layer - torch.max(input_layer))
    return _exp / torch.sum(_exp)

def feedforward(x_vector: tensor, layers: List[Layer]) -> List[tensor]:
    network = [x_vector]
    for layer in layers[:-1]:
        z = torch.matmul(layer.weights, network[-1]) + layer.biases
        activation = layer.activation(z)
        network.append(activation)
    final_layer = layers[-1]
    z = torch.matmul(final_layer.weights, network[-1]) + final_layer.biases
    out_layer = softmax(z)
    network.append(out_layer)
    return network

def backprop(network: List[tensor], y_vector: tensor, layers: List[Layer], learning_rate: float):
    deltas = [network[-1] - y_vector]
    for i in reversed(range(len(layers) - 1)):
        delta = torch.matmul(layers[i + 1].weights.T, deltas[0])
        delta = delta * (network[i + 1] > 0).float()  # Derivative of ReLU
        deltas.insert(0, delta)

    for i, layer in enumerate(layers):
        layer.weights -= learning_rate * torch.outer(deltas[i], network[i])
        layer.biases -= learning_rate * deltas[i]

def analyse_net(layers: List[Layer], X: List[tensor], Y: List[int]) -> float:
    correct_pred = 0
    for i in range(len(X)):
        y_pred = torch.argmax(feedforward(X[i], layers)[-1])
        if y_pred == Y[i]:
            correct_pred += 1
    return correct_pred / len(X)

def train(x_train: List[tensor], y_train: List[tensor], layers: List[Layer], epoch: int, learning_rate: float, x_test: List[tensor], y_test: List[int]) -> List[Layer]:
    for iteration in range(epoch):
        t0 = time.perf_counter()
        for i in range(len(x_train)):
            network = feedforward(x_train[i], layers)
            backprop(network, y_train[i], layers, learning_rate)
        acc = analyse_net(layers, x_train, [torch.argmax(y) for y in y_train])
        print(f"Epoch {iteration + 1}, Accuracy {acc:.4f}, Time {time.perf_counter() - t0:.4f}")
    print(f"Training Data Accuracy {analyse_net(layers, x_train, [torch.argmax(y) for y in y_train]):.4f}")
    return layers

def test_accuracy(x_test: List[tensor], y_test: List[int], layers: List[Layer]):
    print(f"Test Data Accuracy {analyse_net(layers, x_test, y_test):.4f}")

class NeuralNetwork:
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int):
        # Initialize layers
        self.layers = [Layer(input_dim, hidden_layers[0], relu)]
        for i in range(len(hidden_layers) - 1):
            self.layers.append(Layer(hidden_layers[i], hidden_layers[i + 1], relu))
        self.layers.append(Layer(hidden_layers[-1], output_dim, softmax))
        
    def train(self, x_train: List[tensor], y_train: List[tensor], epoch: int, learning_rate: float, x_test: List[tensor], y_test: List[int]):
        self.layers = train(x_train, y_train, self.layers, epoch, learning_rate, x_test, y_test)

    def test(self, x_test: List[tensor], y_test: List[int]):
        test_accuracy(x_test, y_test, self.layers)

if __name__ == "__main__":
	nn = NeuralNetwork(784, [128, 64], 10)
	X = tensor([[0+0.01*i for i in range(100)]])
	Y = tensor([torch.sin(x) for x in X])
	print(X, Y)

	nn.train(X, Y, 100, 0.01, X, Y)
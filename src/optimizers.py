from abc import ABC, abstractmethod

def gradiant_descent(weights, biases, gradients, learning_rate):
    raise NotImplementedError

class optimizers(object):
    @abstractmethod
    def update(self, weights, biases, gradients) -> tuple:
        pass

class Naive(optimizers):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def update(self, weights, biases, gradients):
        return weights - self.learning_rate * gradients, biases - self.learning_rate * gradients

class SGD(optimizers):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def update(self, weights, biases, gradients):
        return weights - self.learning_rate * gradients, biases - self.learning_rate * gradients

class Momentum(optimizers):
    def __init__(self, learning_rate: float, beta: float):
        self.learning_rate = learning_rate
        self.beta = beta
        self.v = 0

    def update(self, weights, gradients):
        self.v = self.beta * self.v + self.learning_rate * gradients
        return weights - self.v

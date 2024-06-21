import pandas as pd
import numpy as np
import time
import torch

from typing import Iterable, Callable
from torch import tensor

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

def relu(input_layer:Iterable) -> tensor:
	result = tensor(input_layer)
	result[result<0] = 0
	return result

def softmax(input_layer:Iterable) -> tensor:
	_exp = torch.exp(input_layer)
	return _exp/torch.sum(_exp)

def generate_weights(layers, seed:int=1) -> list:
	weights = []
	torch.manual_seed(seed)
	for i in range(len(layers) - 1):
		#Adding 1 for bias
		w = torch.rand(layers[i].output_dim, layers[i].input_dim + 1) - 1
		weights.append(w)
	return weights

def feedforward(x_vector,W):
	network = [torch.cat((tensor([1]), x_vector))]
	for weight in W[:-1]:
		next_layer = relu(torch.matmul(network[-1], weight))
		network.append(torch.cat((tensor([1]), next_layer)))
	out_layer = softmax(torch.matmul(network[-1], W[-1]))
	network.append(out_layer)
	return network

def backprop(network,y_vector,W,learning_rate):
	deltas = [network[-1]-y_vector]
	prev_layer = torch.matmul(deltas[0], W[-1].t())[1:]
	deltas.insert(0,prev_layer)
	for weight in list(reversed(W))[1:-1]:
		prev_layer = torch.matmul(deltas[0], weight.t())[1:]
		deltas.insert(0,prev_layer)
	#Weight Update
	for l in range(len(W)):
		for i in range(len(W[l])):
			for j in range(len(W[l][i])):
				W[l][i][j] -= learning_rate*deltas[l][j]*network[l][i]

def analyse_net(W,X,Y):
	correct_pred = 0
	for i in range(len(X)):
		y_pred = torch.argmax(feedforward(X[i],W)[-1])
		if(y_pred==Y[i]):
			correct_pred+=1
	return torch.true_divide(correct_pred,len(X))

def train(x_train,y_train,W,epoch,learning_rate,x_test,y_test):
    for iteration in range(epoch):
        t0 = time.clock()
        for i in range(len(x_train)):
            network = feedforward(x_train[i],W)
            backprop(network,y_train[i],W,learning_rate)
        print("Epoch",iteration+1,"Accuracy",analyse_net(W,x_train,y_train),"Time",time.clock()-t0)
    #Printing training data accuracy
    print("Training Data Accuracy",analyse_net(W,x_train,y_train))

    return W

def test_accuracy(x_test,y_test,W):
	print("Test Data Accuracy",analyse_net(W,x_test,y_test))

class Layer:
    def __init__(self, input_dim:int, output_dim:int, activation:Callable):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = torch.rand(output_dim, input_dim + 1) - 1

class NeuralNetwork:
    def __init__(self, layers:Iterable):
        self.layers = layers
        self.weights = generate_weights(layers)
		
    def train(self, x_train, y_train, epoch:int, learning_rate:float, x_test, y_test):
        train(x_train, y_train, self.weights, epoch, learning_rate, x_test, y_test)
	
    def test(self, x_test, y_test):
        test_accuracy(x_test, y_test, self.weights)
		
import numpy as np
from random import random
def step(x):
    return 1 if x>=0 else 0
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
def relu(x):
    return max(0,x)
def leakyrelu(x,alpha=0.2):
    if x<0:
        return alpha*x
    else:
        return x
def tanh(x):
    return 2*sigmoid(2*x)-1
class NeuralNetwork:
    def __init__(self):
        self.layers=[[3,[0.13436424411240122, 0.8474337369372327, 0.763774618976614]],[2,[0.2550690257394217, 0.49543508709194095]],[0.4494910647887381, 0.651592972722763]]
    def add(self,layer,neuron):
        self.layers.append([layer,neuron,[random() for i in range(neuron+1)]])
    def activate(self,weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
        return activation
    def predict(self,inputs):
        inputs=inputs
        for x in self.layers:
            new_inputs=[]
            for y in range(x[1]):
                act=self.activate(x[2],inputs)
                t=x[0](act)
                new_inputs.append(t)
            inputs = new_inputs
        return inputs
class NeuralNetwork1:
    def __init__(self):
        self.layers=[]
    def add(self,neuron):
        self.layers.append([neuron,[0.13436424411240122, 0.8474337369372327, 0.763774618976614]])
    def activate(self,weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
        print(activation)
        return activation
    def predict(self,inputs):
        inputs=inputs
        for x in self.layers:
            new_inputs=[]
            for y in range(x[0]):
                act=self.activate(x[1],inputs)
                print(act)
                t.append(act)
                tt=1.0 / (1.0 + np.exp(-act))
                new_inputs.append(tt)
            inputs = new_inputs
        return t
n=NeuralNetwork()
#n.add(sigmoid,2)
#n.add(sigmoid,2)
print(n.predict([1,0,None]))
print(n.layers)
from math import exp
# Calculate neuron activation for an input
# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = n.activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs
# test forward propagation
network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
row = [1, 0, None]
output = forward_propagate(network, row)
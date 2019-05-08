# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:26:58 2019

@author: CCF
"""

from numpy import *
import pickle
def make_dataset(dataset):
    X = []
    Y = []

    for video in dataset:
        X.append(video["features"])
        Y.append(video["category"])

    return X, Y


  
class NeuralNet(object): 
    def __init__(self): 
        # Generate random numbers 
        random.seed(1) 
  
        # Assign random weights to a 3 x 1 matrix, 
        self.synaptic_weights = 2 * random.random((3, 1)) - 1
  
    # The Sigmoid function 
    def __sigmoid(self, x): 
        return 1 / (1 + exp(-x)) 
  
    # The derivative of the Sigmoid function. 
    # This is the gradient of the Sigmoid curve. 
    def __sigmoid_derivative(self, x): 
        return x * (1 - x) 
  
    # Train the neural network and adjust the weights each time. 
    def train(self, inputs, outputs, training_iterations): 
        for iteration in xrange(training_iterations): 
  
            # Pass the training set through the network. 
            output = self.learn(inputs) 
  
            # Calculate the error 
            error = outputs - output 
  
            # Adjust the weights by a factor 
            factor = dot(inputs.T, error * self.__sigmoid_derivative(output)) 
            self.synaptic_weights += factor 
  
    # The neural network thinks. 
    def learn(self, inputs): 
        return self.__sigmoid(dot(inputs, self.synaptic_weights)) 
  
if __name__ == "__main__": 
  
    #Initialize 
    neural_network = NeuralNet() 
  
    # The training set. 
    dataset = pickle.load(open(dataset_bow, "rb"))
    X, Y = make_dataset(dataset)
  
    # Train the neural network 
    neural_network.train(X, Y, 10000) 
  
    # Test the neural network with a test example. 
    #print neural_network.learn() 
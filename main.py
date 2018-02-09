#imports numpy library as np("www.numpy.org")
import numpy as np

class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
    
    #sigmoid function("https://en.wikipedia.org/wiki/Sigmoid_function")
    def __nonlin(self, x,deriv=False):
        if deriv == True:
            return x * (1 - x)
        else:
            return 1 / (1+np.exp(-x))
    
    #trains neural network
    def train(self, training_set_inputs,training_set_outputs,number_of_training_iterations):
        #iterates a number of times 
        for iteration in xrange(number_of_training_iterations):
            #gets output based of training set inputs
            output = self.think(training_set_inputs,False)
            #gets the error when the output is wrong 
            error = training_set_outputs - output
            #adjusts training values depending on the error
            adjustment = np.dot(training_set_inputs.T, error * self.__nonlin(output,True))
            #synaptic weights change depending on adjustment
            self.synaptic_weights += adjustment
            
    #puts the dot matrix of the inputs and the synaptic weights in a segmoid function
    def think(self,inputs,rnd=False):
        if rnd == True:
            otpt = self.__nonlin(np.dot(inputs, self.synaptic_weights))
            return np.round(otpt,1)
        else:
            return self.__nonlin(np.dot(inputs, self.synaptic_weights))


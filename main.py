#imports methods from numpy library("www.numpy.org")
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
    def train(self, training_set_inputs,traing_set_outputs,number_of_training_iterations):
        #iterates a number of times 
        for iteration in xrange(number_of_training_iterations):
            #gets output based of training set inputs
            output = self.think(training_set_inputs)
            #gets the error when the output is wrong 
            error = training_set_outputs - output
            #adjusts training values depending on the error
            adjustment = np.dot(training_set_inputs.T, error * self.__nonlin(output,True))
            #synaptic weights chainge depending on adjustment
            self.synaptic_weights += adjustment
            
    #puts the dot matrix of the inputs and the synaptic weights in a segmoid function
    def think(self,inputs):
        return self.__nonlin(np.dot(inputs, self.synaptic_weights))
#main (runs when program is executed)    
if __name__ == "__main__":
    #creates instance of neural network class
    neural_network = NeuralNetwork()
    #Shows the starting synpatic weights
    print "Starting synaptic weights: "
    print neural_network.synaptic_weights
    #training data input
    training_set_inputs = np.array ([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    #training data output
    training_set_outputs = np.array([[0,1,1,0]]).T
    #trains neural network with training inputs and outputs 1000 times
    neural_network.train(training_set_inputs,training_set_outputs,1000)
    #Shows new synaptic weights after training
    print "New synaptic wheights after training: "
    print neural_network.synaptic_weights
    #Considers a new situation
    print "Consider new situation [1,0,0] -> ?: "
    print np.round(neural_network.think(np.array([1,0,0])),1)
    print "Consider new situation [0,1,0] -> ?: "
    print neural_network.think(np.array([0,1,0]))

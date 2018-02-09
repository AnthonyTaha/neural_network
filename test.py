#imports numpy library as np("www.numpy.org")
import numpy as np
#imports main neural network file
import main
#creates instance of neural network class
neural_network = main.NeuralNetwork()
#Shows the starting synpatic weights
print "Starting synaptic weights: "
print neural_network.synaptic_weights
#training data input
training_set_inputs = np.array ([[0,0,10],[10,10,10],[10,0,10],[0,10,10]])
#training data output
training_outputs = np.array([[0,1,1,0]]).T
#trains neural network with training inputs and outputs 1000 times
neural_network.train(training_set_inputs,training_outputs,1000)
#Shows new synaptic weights after training
print "New synaptic wheights after training: "
print neural_network.synaptic_weights
#Considers a new situation
print "Consider new situation [1,0,0] -> ?: "
print neural_network.think(np.array([10,0,0]),True)
print "Consider new situation [0,1,0] -> ?: "
print neural_network.think(np.array([0,10,0]),True)
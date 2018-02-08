import numpy as np

class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
        
    def __sigmoid(self, x):
        return 1 / (1+np.exp(-x))
    
    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def train(self, training_set_inputs,traing_set_outputs,number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            output = self.think(training_set_inputs)
            
            error = training_set_outputs - output
            
            adjustment = np.dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            
            self.synaptic_weights += adjustment
            
    
    def think(self,inputs):
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))
    
if __name__ == "__main__":
    neural_network = NeuralNetwork()
    
    print "Starting synaptic weights: "
    print neural_network.synaptic_weights
    
    training_set_inputs = np.array ([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    training_set_outputs = np.array([[0,1,1,0]]).T
    
    neural_network.train(training_set_inputs,training_set_outputs,1000)
    
    print "New synaptic wheights after training: "
    print neural_network.synaptic_weights
    
    print "Consider new situation [1,0,0] -> ?: "
    print neural_network.think(np.array([1,0,0]))
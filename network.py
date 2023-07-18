import numpy as np 

def sigmoid(z: np.array):
    return 1.0/(1.0+np.exp(-z))

class Network:

    def __init__(self, sizes: list):
        self.num_layers = len(sizes)
        # a list that contains the number of neurons in each layer
        self.sizes = sizes
        # the first layer has no biases, as it is the input layer, biases are 
        # create a bias for every neuron
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # create a weight for every edge (y by x array where y is the number of neurons in a layer, x is the number of neurons in the previous layer)
        # weights[0] represents the weights of the edges connecting the inputs to the second layer
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a




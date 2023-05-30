import numpy as np

class LayerDense:
    # Initialise layer with weights and biases
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn((n_inputs, n_neurons))
        self.biases = np.zeroes((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    # Backward pass
    def backward(self, dvalues):
        pass


class ActivationReLU:
    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        pass


class ActivationSoftmax:
    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs

        # Get unnormalised probalitites
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
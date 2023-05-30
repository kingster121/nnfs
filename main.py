from layer import Layer # Base layer
import numpy as np

class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.biases = np.random.rand(1, output_size) -0.5

    
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output
    

    # Computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # Update params
        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * output_error

        return input_error
    
class ActivationLayer(Layer):
    # activation = activation function
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime


    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error
    

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    
    def add(self, layer):
        self.layers.append(layer)

    
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime


    def predict(self, input_data):
        result = []

        # Run network 
        for i in range(len(input_data)):
            # Forward propagate
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagate(output)
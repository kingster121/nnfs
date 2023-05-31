from layer import Layer  # Base layer
import numpy as np
import time


class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.biases = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        # print(self)
        # print(f"input: {self.input}")
        # print(f"weights: {self.weights}")
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

        # Run the input_data through the network individually
        # AKA forward propagate the input all the way to the output layer
        for i in range(len(input_data)):
            # Forward propagate
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    # x_train = training_input
    # y_train = training_output
    def fit(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            err = 0
            for i in range(len(x_train)):
                # Forward propagation
                output = x_train[i]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                err += self.loss(y_train[i], output)

                # Backward propagation
                error = self.loss_prime(y_train[i], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

                # Check how the weights and biases in the first layer evolve
                if isinstance(self, FCLayer):
                    layer = self.layers[-1]
                    print(f"weights: {layer.weights}")
                    print(f"biases: {layer.biases}")
            # Calculate avg error of all samples
            err /= len(x_train)
            print(f"epoch = {epoch + 1}/{epochs} error = {err}")

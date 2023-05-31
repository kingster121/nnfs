# Neural Network From Scratch
Here, the purpose is to create a simple neural network without reliance on ML frameworks, using as much of plain vanilla Python as possible.
\
\
Source: [Medium - Neural Network from Scratch](https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65)\
TL;DR: Run either the mnist.py or xor.py

## Steps and Explanation
1. Forward propagation is about making propagations
2. However, the predictions might be bad and the model need to be further trained
3. Backward propagation is about improving the models' weights and biases for higher accuracy

## Model Structure
1. The algorithm is structured in layers whereas there could be any number of neurons in the layer, pre-defined by the user
2. Every neuron in the previous layer is mapped to the next layer's neurons through weights. And every neuron would have a bias linked to it.
\
E.g. In the second layer, there is 4 neurons and in the third layer, there is 8 neurons. The number of weights would be 4x8 whereas the biases would only be 4+8. 

## Activation function (sigmoid, ReLU, softmax). 
1. Prevent explosion and drowning out of the other weight/biases
2. Non-linear activation function is commonly used so that they can 'compound' on one another to fit any type of data
## Backward propagation
Very maths heavy. But the essence is just finding the dW/dC and dB/dC by applying chain rules to those further up the layers. 
\
Mathematical derivation: [Deriving backpropagation](https://www.youtube.com/watch?v=XE3krf3CQls&list=PLZbbT5o_s2xq7LwI2y8_QtvuXZedL6tQU&index=25)
\
Visual/Intuition: [3Blue1Brown backpropagation intuitions](https://www.youtube.com/watch?v=tIeHLnjs5U8&t=385s)
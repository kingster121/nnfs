# Neural Network From Scratch

A small fully connected neural network built with only **NumPy**, with no PyTorch or TensorFlow for the model itself. I wrote it to understand what actually happens inside a network: how forward propagation produces a prediction, and how backpropagation uses the chain rule to update every weight and bias.

The network is tested on two problems:
- **XOR**, a classic problem a single-layer model cannot solve because the data isn't linearly separable
- **MNIST handwritten digits**, 28×28 images classified into 10 classes

> Built by following Omar Aflak's [Math + Neural Network from Scratch in Python](https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65), then reworking and commenting the code to make sure I understood each step.

## Quick start

```bash
pip install numpy tensorflow   # TensorFlow/Keras is only used to download the MNIST dataset
python xor.py
python mnist.py
```

## Project structure

| File | What it does |
|---|---|
| `layer.py` | Base `Layer` class defining the forward/backward interface |
| `architecture.py` | `FCLayer` (fully connected), `ActivationLayer`, and the `Network` class that trains and predicts |
| `activation.py` | `tanh` and `ReLU`, with their derivatives |
| `loss.py` | Mean squared error and its derivative |
| `xor.py` | Trains a 2 → 3 → 1 network on XOR |
| `mnist.py` | Trains a 784 → 100 → 50 → 10 network on MNIST |

## How it works

### 1. Layers
Each fully connected layer holds a weight matrix and a bias vector. A layer with `n` inputs and `m` outputs has an `n × m` weight matrix and `m` biases, one per output neuron. For example, a layer going from 4 neurons to 8 neurons has 32 weights and 8 biases.

Activation functions are implemented as their own layers. This keeps each component small: a layer only needs to know how to compute its output, and how to pass the error backwards.

### 2. Forward propagation
The input passes through each layer in turn:

```
output = input · W + b      # fully connected layer
output = f(output)          # activation layer (tanh or ReLU)
```

Non-linear activations matter. Without them, stacking layers would collapse into one linear transformation, and the network could never learn XOR.

### 3. Loss
Mean squared error measures how far the prediction is from the target.

### 4. Backpropagation
Each layer receives the error with respect to its output (`∂E/∂Y`) and uses the chain rule to compute:

- `∂E/∂W = Xᵀ · ∂E/∂Y` to update the weights
- `∂E/∂B = ∂E/∂Y` to update the biases
- `∂E/∂X = ∂E/∂Y · Wᵀ`, which is passed back to the previous layer

Parameters are updated with stochastic gradient descent, one sample at a time:

```
W = W − learning_rate × ∂E/∂W
```

## Limitations

- **No mini-batching:** training is per sample, so `mnist.py` only trains on 1,000 of the 60,000 training images to keep run time reasonable.
- **MSE for classification:** softmax with cross-entropy would be the better fit for MNIST, but MSE kept the maths simple.
- **Basic initialisation:** weights are drawn uniformly from [−0.5, 0.5] rather than using Xavier or He initialisation.

## Resources
- [Deriving backpropagation step by step](https://www.youtube.com/watch?v=XE3krf3CQls&list=PLZbbT5o_s2xq7LwI2y8_QtvuXZedL6tQU&index=25)
- [3Blue1Brown: Backpropagation intuition](https://www.youtube.com/watch?v=tIeHLnjs5U8&t=385s)

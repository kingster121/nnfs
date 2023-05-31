import numpy as np

from architecture import Network, ActivationLayer, FCLayer
from activation import ReLU, ReLU_prime, tanh, tanh_prime
from loss import mse, mse_prime

from keras.datasets import mnist
from keras.utils import np_utils

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape and normalise input data
x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28)
x_train = x_train.astype("float32")
x_train /= 255
# Encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)

# Do the same for test data
x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28)
x_test = x_test.astype("float32")
x_test /= 255
y_test = np_utils.to_categorical(y_test)

# Setting up the model
net = Network()
net.add(FCLayer(28 * 28, 100))
net.add(ActivationLayer(tanh, tanh_prime))
# net.add(ActivationLayer(ReLU, ReLU_prime))
net.add(FCLayer(100, 50))
net.add(ActivationLayer(tanh, tanh_prime))
# net.add(ActivationLayer(ReLU, ReLU_prime))
net.add(FCLayer(50, 10))
net.add(ActivationLayer(tanh, tanh_prime))
# net.add(ActivationLayer(ReLU, ReLU_prime))

# Training the model on 1000 samples since mini-batch GD is not implemented, hard to train on all 60K data
net.use(mse, mse_prime)
net.fit(x_train[:1000],y_train[:1000], epochs=10, learning_rate=0.2)

# Test
out = net.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])

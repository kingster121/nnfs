import numpy as np

from architecture import Network, ActivationLayer, FCLayer
from activation import ReLU, ReLU_prime
from loss import mse, mse_prime

# Training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# Setting up the model
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(ReLU, ReLU_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(ReLU, ReLU_prime))

# Training the model
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# Test
out = net.predict(x_train)
print(out)

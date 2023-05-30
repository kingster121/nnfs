import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def ReLU(x):
    return np.maximum(0,x)

# Might have problems if x = 0
def ReLU_prime(x):
    if x>0:
        return 1
    elif x<0:
        return 0
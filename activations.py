import math

def _sigmoid(x):
    """Return the S-Curve activation of x."""
    return 1/(1+math.exp(-x))

def sigmoid(x, alpha):
    """ Scale the sigmoid function by alpha"""
    return _sigmoid(alpha * x)

def _tanh(x):
    """Wrapper function for hyperbolic tangent activation."""
    return math.tanh(x)

def tanh(x, alpha):
    """Scale the hyperbolic tangent function by alpha"""
    return alpha * _tanh(x)

def LReLU(x):
    """Leaky ReLU function for x"""
    if x >= 0:
        return x
    else:
        return 0.01 * x

import numpy as np

def sigmoid(x, derivative=False, x_precomp=False):
    """Sigmoid activation function. Works on scalars, vectors, matrices.
    Set derivative to true to compute derivative (wrt to x)"""
    s = 1.0/(1.0+np.exp(-x))
    if not derivative:
        return s
    else:
        if not x_precomp:
            return s*(1.0-s)
        else:
            return x*(1.0-x)

def column_vector(v):
    if v.shape[0] == 1:
        return v.T
    else:
        return v


def cross_entropy(output, target):
    """Get cross entropy between ouput and desired target"""
    output = np.array(output)
    target = np.array(target)
    s = -1.0/float(len(output))
    return s*np.sum(target*np.log(output)+(1-target)*np.log(1-output))


def sum_of_squares_error(output, target, derivative=False):
    """Sum of squares error function"""
    if not derivative:
        return .5*np.sum(np.power(np.subtract(output, target), 2))
    else:
        return output - target

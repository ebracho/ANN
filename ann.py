# Inspired by http://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np

def sigmoid(x, deriv=False):
    return x*(1-x) if deriv else 1/(1+np.exp(-x))

class ANN:

    # Fresh neural network
    def __init__(self, inp_dim, out_dim, hid_dim, syn0=None, syn1=None):
        if syn0 != None and syn1 != None:
            self.syn0, self.syn1 = syn0, syn1
        else:
            hid_dim = hid_dim if hid_dim else (inp_dim + out_dim) / 2
            self.syn0 = 2 * np.random.random((inp_dim, hid_dim)) - 1
            self.syn1 = 2 * np.random.random((hid_dim, out_dim)) - 1

    @classmethod
    def fromfile(cls, filename):
        if not filename.endswith('.npz'): filename += '.npz'
        with np.load(filename) as data:
            return cls(0, 0, syn0=data['syn0'], syn1=data['syn1'])

    def train(self, X, y, iterations=1):
        for _ in xrange(iterations):

            """ Forward Propagation """
            l0 = X
            l1 = sigmoid(np.dot(l0, self.syn0))
            l2 = sigmoid(np.dot(l1, self.syn1))

            """ Backpropagation """
            # How much did we miss by
            l2_error = y - l2

            # Did l2 miss by a lot? What direction is the target value?
            l2_delta = l2_error * sigmoid(l2, deriv=True)

            # How much did l1 contriute to the error?
            l1_error = np.dot(l2_delta, self.syn1.T)

            # Did l1 miss by a lot? What direction is the target value?
            l1_delta = l1_error * sigmoid(l1, deriv=True)

            self.syn1 += np.dot(l1.T, l2_delta)
            self.syn0 += np.dot(l0.T, l1_delta)

    def evaluate(self, X):
        l1 = sigmoid(np.dot(X, self.syn0))
        return sigmoid(np.dot(l1, self.syn1))
        
    def write_to_file(self, filename):
        np.savez(filename, syn0=self.syn0, syn1=self.syn1)
        

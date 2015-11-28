import numpy as np

def sigmoid(x, deriv=False):
    return x*(1-x) if deriv else 1/(1+np.exp(-x))

# Artificial Neural Network with one hidden layer
class ANN:
    def __init__(self, inp_dim, hid_dim, out_dim, alpha=1):
        self.syn0 = np.random.random((inp_dim+1, hid_dim))
        self.syn1 = np.random.random((hid_dim+1, out_dim))
        self.alpha = alpha
        self.l0_bias = np.ones(inp_dim)
        self.l1_bias = np.ones(hid_dim)

    # Propagate inp forward through the ann and return output
    def estimate(self, inp):
        l0 = np.concatenate((inp, [1]))
        l1 = sigmoid(np.dot(l0, self.syn0)) 
        l1 = np.concatenate((l1, [1]))
        l2 = sigmoid(np.dot(l1, self.syn1)) 
        return l2

    # Train synapses using Backpropagation Algorithm 
    def train(self, inp, out):
        # Compute output using forward propagation
        l0 = np.concatenate((inp, [1]))
        l1 = sigmoid(np.dot(l0, self.syn0)) 
        l1 = np.concatenate((l1, [1]))
        l2 = sigmoid(np.dot(l1, self.syn1)) 
        
        # Compute error using backpropagation
        l2_error = (out - l2) * sigmoid(l2, deriv=True)
        l1_error = np.dot(l2_error, self.syn1.T) * sigmoid(l1, deriv=True)
        l0_error = np.dot(l1_error[:-1], self.syn0.T) * sigmoid(l0, deriv=True)

        # Adjust synapses according to error and learning rate
        self.syn1 += self.alpha * np.outer(l1.T, l2_error)
        self.syn0 += self.alpha * np.outer(l0.T, l1_error[:-1])

        return l2_error

    # Trains neural network from a set. Yields error after each iteration
    def train_set(self, inp_set, out_set, iterations=1):
        for i in range(iterations):
            errors = []
            for inp, out in zip(inp_set, out_set):
                errors.append(abs(self.train(inp, out)))
            yield np.mean(errors)

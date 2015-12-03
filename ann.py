import numpy as np

@np.vectorize
def sigmoid(x, deriv=False):
    if deriv: 
        return x*(1-x)
    else:
        # Catch large x values to avoid overflow
        if x > 100:
            return 1
        if x < -100:
            return 0
        else:
            return 1/(1+np.exp(-x))

def euclidean_distance(a, b):
    return np.linalg.norm(a-b)

# Artificial Neural Network with one hidden layer
class ANN:
    def __init__(self, inp_dim, hid_dim, out_dim, alpha=0.01):
        self.syn0 = np.random.random((inp_dim+1, hid_dim)) - 0.5
        self.syn1 = np.random.random((hid_dim+1, out_dim)) - 0.5
        self.alpha = alpha
        self.total_iterations = 0

    @classmethod
    def from_file(cls, filename):
        obj = cls(1,1,1)
        with np.load(filename) as data:
            obj.syn0 = data['syn0']
            obj.syn1 = data['syn1']
            obj.alpha = data['alpha']
            obj.total_iterations = data['total_iterations']
        return obj
        
    # Propagate inp forward through the ann and return output
    def estimate(self, inp):
        l0 = np.concatenate((inp, [1]))
        l1 = sigmoid(np.dot(l0, self.syn0)) 
        l1 = np.concatenate((l1, [1]))
        l2 = sigmoid(np.dot(l1, self.syn1)) 
        return l2

    # Train synapses using Backpropagation Algorithm 
    def train(self, data):
        inp, out = data

        # Compute output using forward propagation
        l0 = np.concatenate((inp, [1])) # Append 1 to input layer for bias 
        l1 = sigmoid(np.dot(l0, self.syn0)) # Pass through syn0 and activate
        l1 = np.concatenate((l1, [1])) # Append 1 to hidden layer for bias
        l2 = sigmoid(np.dot(l1, self.syn1)) # Pass through syn1 and activate
        
        # Compute error using backpropagation. This is achieved by passing 
        # error vectors through the transpose of the synapse matrices. Note 
        # that we need to remove l1's bias node for the matrices to align.
        l2_error = (out - l2) * sigmoid(l2, deriv=True)
        l1_error = np.dot(l2_error, self.syn1.T) * sigmoid(l1, deriv=True)
        l1_error = l1_error[:-1]
        l0_error = np.dot(l1_error, self.syn0.T) * sigmoid(l0, deriv=True)

        # Adjust synapses according to error and learning rate
        self.syn1 += self.alpha * np.outer(l1.T, l2_error)
        self.syn0 += self.alpha * np.outer(l0.T, l1_error)

        return euclidean_distance(out, l2)

    # Trains neural network from a set. Yields error after each iteration
    def train_set(self, training_set):
        error = []
        for data in training_set:
            error.append(self.train(data))
        self.total_iterations += 1
        return error

    # Writes network to a file
    def write_to_file(self, filename):
        np.savez(filename, syn0=self.syn0, syn1=self.syn1, alpha=self.alpha, 
                 total_iterations=self.total_iterations)

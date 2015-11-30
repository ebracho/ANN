import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from ann import ANN

def plot_error(data):
    plt.xlabel('Iterations')
    plt.ylabel('Mean Error')
    plt.plot(data)
    plt.show()

# Training set for the logical operator XOR
training_set = [ ([0,0],[0]),
                 ([1,0],[1]),
                 ([0,1],[1]),
                 ([1,1],[0]) ]

# Construct a neural net with 2 input nodes, 3 hidden nodes, and 1 output node
ann = ANN(2,3,1,alpha=1)

# Run the network through 1000 iterations of the training set
error_data = [ann.train_set(training_set) for _ in range(1000)]

# Plot the mean error for each iteration
plot_error(error_data)


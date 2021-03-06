import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from ann import ANN

def plot_error(data):
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
error_data = [ann.train_set(training_set) for _ in range(2000)]

# Plot the mean error for each iteration
none, left, right, both = zip(*error_data)

none_line, = plt.plot(none)
left_line, = plt.plot(left)
right_line, = plt.plot(right)
both_line, = plt.plot(both)

plt.legend([none_line, left_line, right_line, both_line], ['[0,0]', '[1,0]', '[0,1]', '[1,1]'])

plt.xlabel('Iterations')
plt.ylabel('Error')

plt.show()


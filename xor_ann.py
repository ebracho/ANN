import numpy as np
from ann import ANN

# Training set input
X = np.array([[0,0,1],
              [1,0,1],
              [0,1,1],
              [1,1,1]])

# Training set output
y = np.array([[0,1,1,0]]).T

ann = ANN.fromfile('xor_ann')
# ann = ANN(3,1)
# ann.train(X,y,iterations=10000)
# ann.write_to_file('xor_ann')
ann.train(X,y,iterations=10000)
print ann.evaluate([1,0,1])
ann.write_to_file('xor_ann')

import numpy as np
import matplotlib.pyplot as plt
from ann import ANN

"""
def plot_training_progress(line, ann, inp_set, out_set, iterations):
    xdata = []
    ydata = []
    for i in xrange(iterations):
        errors = []
        for inp, out in zip(inp_set, out_set):
            errors.append(abs(ann.train(inp,out)))
        xdata.append(i)
        ydata.append(np.mean(errors))

    return xdata, ydata
"""

# Training set input
X = np.array([[0,0,1],
              [1,0,1],
              [0,1,1],
              [1,1,1]])

# Training set output
y = np.array([[0,1,1,0]]).T

ann = ANN(3,3,1,alpha=0.5)
error_data = list(ann.train_set(X,y,10000))
plt.plot(error_data)
plt.show()


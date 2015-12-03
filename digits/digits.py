import cPickle, gzip, sys
import numpy as np
from matplotlib import pyplot as plt
sys.path.append('../')
from ann import ANN

def plot_error(data):
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.plot(data)
    plt.show()

# Creates the output vector corresponding to digit
def classify(digit):
    v = np.zeros(10) + .1
    v[digit] = .9
    return v
    
# Convert dataset into format that ANN recognizes
def build_set(inpset, outpset):
    return [(inp, classify(outp)) for inp, outp in zip(inpset, outpset)]

def test_network(ann, test_set):
    passed = 0
    failed = 0
    for inp,outp in test_set:
        estimate = ann.estimate(inp)
        if np.argmax(estimate) == np.argmax(outp):
            passed += 1
        else:
            failed += 1
    accuracy = float(passed)/float(passed+failed)
    print 'Passed: %d' % passed
    print 'Failed: %d' % failed
    print 'Accuracy: %f' % accuracy
    

# Load the pickled dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

# Convert training and testing sets
train_set = build_set(train_set[0], train_set[1])
test_set = build_set(test_set[0], test_set[1])

digits_ann = ANN(784,30,10,alpha=0.1)

print '+---------------+'
print '|Before Training|'
print '+---------------+'
test_network(digits_ann, test_set)

digits_ann.train_set(train_set)

print '+--------------+'
print '|After Training|'
print '+--------------+'
test_network(digits_ann, test_set)

import Image, ImageOps, glob, sys
import numpy as np
from itertools import cycle, islice
sys.path.append('../') # Add ann.py to sys path
from ann import ANN

# The ideal input values from neural networks should be between +-0.5
# normalize_pixel() maps a pixel value (0:255) to a value in (-.5:.5)
def normalize_pixel(pixel):
    return (pixel/255.0)-0.5

# Returns a list of tuples that contain images raw pixels and desired output
def load_training_set(directory, output):
    training_set = []
    filepaths = glob.glob(directory + '/*.bmp')
    for fp in filepaths:
        # Open image and get data
        pixels = Image.open(fp).getdata()
        # Pixels are stored as tuples (red, green, blue). We need a flat list of bytes
        values = np.array([val for rgb in pixels for val in rgb]) 
        values = normalize_pixel(values) 
        training_set.append((values, output)) 
    return training_set
        
# Creating a master training set which cycles through images of each class
def build_training_sets():
    # Build training sets for each class
    violin_set = load_training_set('image_search/images/resized/violin', [1,0,0,0,0])
    trumpet_set = load_training_set('image_search/images/resized/trumpet', [0,1,0,0,0])
    tuba_set = load_training_set('image_search/images/resized/tuba', [0,0,1,0,0])
    flute_set = load_training_set('image_search/images/resized/flute', [0,0,0,1,0])
    clarinet_set = load_training_set('image_search/images/resized/clarinet', [0,0,0,0,1])

    # Create a list of tuples containing one of each time of instrument
    zipset = zip(violin_set, trumpet_set, tuba_set, flute_set, clarinet_set)
    return [inp for subset in zipset for inp in subset] # flatten zipset
    

# ann = ANN(40*40*3, 50, 5, 1)
ann = ANN.from_file('instrument_ann.npz')
ann.alpha=1
instrument_training_set = build_training_sets()
for _ in xrange(50):
    print ann.train_set(instrument_training_set)
    ann.write_to_file('instrument_ann.npz')

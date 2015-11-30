import Image, glob, sys
import numpy as np
from itertools import cycle, islice
sys.path.append('../') # Add ann.py to sys path
from ann import ANN

# The ideal input values from neural networks should be between +-0.5
# normalize_pixel() maps a pixel value (0:255) to a value in (-.5:.5)
def normalize_pixel(pixel):
    return (pixel/255.0)-0.5

# Returns a list of tuples that contain images raw pixels and desired output
def load_training_set(directory,output,width,height):
    training_set = []
    filepaths = glob.glob(directory + '/*.bmp')
    for fp in filepaths:
        # Load image, convert it to greyscale, and gets pixel data
        pixels = list(Image.open(fp).convert('RGB').getdata()) 
        pixels = [val for pixel in pixels for val in pixel]
        pixels = [normalize_pixel(pix) for pix in pixels]
        training_set.append((pixels,output))
    return training_set
        
def build_training_sets():
    # Build training sets for each class
    violin_set = load_training_set('image_search/images/resized/violin', [1,0,0,0,0], 50, 50)
    trumpet_set = load_training_set('image_search/images/resized/trumpet', [0,1,0,0,0], 50, 50)
    tuba_set = load_training_set('image_search/images/resized/tuba', [0,0,1,0,0], 50, 50)
    flute_set = load_training_set('image_search/images/resized/flute', [0,0,0,1,0], 50, 50)
    clarinet_set = load_training_set('image_search/images/resized/clarinet', [0,0,0,0,1], 50, 50)
    # Creating a master training set which cycles through images of each class
    zipset = zip(violin_set, trumpet_set, tuba_set, flute_set, clarinet_set)
    return [inp for subset in zipset for inp in subset]
    
# ann = ANN(7500,100,5)
ann = ANN.from_file('instrument_ann.npz')
ann.alpha=0.01
instrument_training_set = build_training_sets()
for _ in xrange(100):
    print ann.train_set(instrument_training_set)
ann.write_to_file('instrument_ann.npz')

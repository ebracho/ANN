import Image, ImageOps, urllib, sys
from cStringIO import StringIO
sys.path.append('../')
from ann import ANN

VIOLIN_CLASSIFIER = [1,0,0,0,0]
TRUMPET_CLASSIFIER = [0,1,0,0,0]
TUBA_CLASSIFIER = [0,0,1,0,0]
FLUTE_CLASSIFIER = [0,0,0,1,0]
CLARINET_CLASSIFIER = [0,0,0,0,1]

# Fits an Image object to 28x28 and grayscales
def format_image(im):
    pixels = ImageOps.fit(im,(40,40)).convert('L').getdata()
    return pixels
    # return [val for rgb in pixels for val in rgb]

def format_img_from_url(url):
    img_data = urllib.urlopen(url).read()
    im = Image.open(StringIO(img_data))
    return format_image(im)

def format_img_from_file(filepath):
    with open(filepath) as f:
        im = Image.open(StringIO(f.read()))
        return format_image(im)

def classify(estimate):
    total = sum(estimate)
    """
    print "Violin:   %s%%" % str(estimate[0]/total)
    print "Trumpet:  %s%%" % str(estimate[1]/total)
    print "Tuba:     %s%%" % str(estimate[2]/total)
    print "Flute:    %s%%" % str(estimate[3]/total)
    print "Clarinet: %s%%" % str(estimate[4]/total)
    """
    print "Violin:   %s%%" % str(100*estimate[0]/total)
    print "Trumpet:  %s%%" % str(100*estimate[1]/total)
    print "Tuba:     %s%%" % str(100*estimate[2]/total)
    print "Clarinet: %s%%" % str(100*estimate[3]/total)

method = sys.argv[1]
source = sys.argv[2]

if method == 'url':
    inp = format_img_from_url(source)

elif method == 'filepath':
    inp = format_img_from_file(source)

else:
    print "bad input"
    sys.exit(1)

ann = ANN.from_file('instrument_ann.npz')
classify(ann.estimate(inp))

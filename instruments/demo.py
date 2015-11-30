import Image, ImageOps, urllib, sys
from cStringIO import StringIO
sys.path.append('../')
from ann import ANN

VIOLIN_CLASSIFIER = [1,0,0,0,0]
TRUMPET_CLASSIFIER = [0,1,0,0,0]
TUBA_CLASSIFIER = [0,0,1,0,0]
FLUTE_CLASSIFIER = [0,0,0,1,0]
CLARINET_CLASSIFIER = [0,0,0,0,1]

# Fits an Image object to 50x50 and greyscales
def format_image(im):
    converted_im = ImageOps.fit(im, (50,50), Image.ANTIALIAS, centering=(0.5,0.5)).convert('RGB').getdata()
    return [val for pixel in converted_im for val in pixel]

def format_img_from_url(url):
    img_data = urllib.urlopen(url).read()
    im = Image.open(StringIO(img_data))
    return format_image(im)

def format_img_from_file(filepath):
    with open(filepath) as f:
        im = Image.open(StringIO(f.read()))
        return format_image(im)

# inp = format_img_from_url('http://www.amromusic.com/assets/1942/7_trumpet-1.jpg')
inp = format_img_from_file('image_search/images/original/violin0')

ann = ANN.from_file('instrument_ann.npz')
print ann.estimate(inp)

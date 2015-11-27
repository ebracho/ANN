import numpy as np
from ann import ANN

# Training set input
from PIL import Image
import os
dir = "images"
num = 0
for file in os.listdir(dir):
    im = Image.open(os.path.join(dir, file))
    pix = im.load()
    pix_count = 0
    white = 255
    for i in range(0,50):
        for j in range(0,25):
            RGB = pix[i, j]
            R, G, B = RGB
            if R < white or G < white or B < white:
                pix_count += 1
    a = float(pix_count) / 1250
    pix_count = 0
    for i in range(0,50):
        for j in range(25,50):
            RGB = pix[i, j]
            R, G, B = RGB
            if R < white or G < white or B < white:
                pix_count += 1
    b = float(pix_count)/ 1250
    d = [a,b]
    if (num == 0):
        arr = np.array([[a,b]])
    else:
        arr = np.vstack((arr,d))
    num+=1
ann = ANN(2,1,4)
y = np.array([[0,1]]).T
ann.train(arr,y,iterations=10000)
print ann.evaluate(arr[0])
print ann.evaluate(arr[1])

ann.write_to_file('xor_ann')

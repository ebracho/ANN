import numpy as np
from ann import ANN

# Training set input
from PIL import Image
import os
for dir_num in range(0, 2):
    #if dir_num == 0:
     #   dir = "images/clarinet_50"
    if dir_num == 0:
        dir = "images/trumpet_50"
    elif dir_num == 1:
        dir = "images/violin_50"
    num = 0
    for file in os.listdir(dir):
        im = Image.open(os.path.join(dir, file))
        pix = im.load()
        pix_count = 0
        white = 255
        for i in range(0, 50):
            for j in range(0, 25):
                RGB = pix[i, j]
                R, G, B = RGB
                if R <white or G <white or B <white:
                    pix_count += 1
        a = float(pix_count) / 1250
        pix_count = 0
        for i in range(0, 50):
            for j in range(25, 50):
                RGB = pix[i, j]
                R, G, B = RGB
                if R < white or G < white or B < white:
                    pix_count += 1
        b = float(pix_count) / 1250
        d = [a, b]
        if num == 0 and dir_num == 0:
            arr = np.array([[a, b]])
            y = np.array([[0]])
        else:
            arr = np.vstack((arr, d))
            if dir_num == 1 or dir_num == 0:
                y = np.append(y, [[dir_num]])
            elif dir_num == 2:
                y = np.append(y, [[0.5]])
        num += 1
ann = ANN(2,16,1, alpha=.5)
y = np.array(y, ndmin=2).T
ann.train(arr, y, iterations=60000)
for i in range(0, y.size):
    out=ann.estimate(arr[i])
    print(i, out, y[i])

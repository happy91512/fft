from cv2 import waitKey
from numba import jit
import numpy as np  
import math
import cv2
import time

#@jit(nopython = True)
def dft(img:np.array):
    pi = math.pi
    exp = np.exp
    size = img.shape
    dftimg_path = "./dftimg.txt"
    
    print(size)
    img_height = size[0]
    img_width = size[1]
    dft_img = (img_height*[img_width*[[0,0]]])
    start_time = time.time()
    for v in range(0, img_height):
        for u in range(0, img_width):
            for y in range(0, img_height):
                for x in range(0, img_width):
                    dft_img[v][u] = dft_img[v][u] + (img[y][x]*exp(2j*pi*((u*x/img_width)+(v*y/img_height))))
                    print(v,u) 
    end_time = time.time()
    file1 = open(dftimg_path, 'w')
    file1.write(str(dft_img))  
    print(dft_img.shape)
    print("spent", end_time-start_time, "seconds")

img_path = "./frog.png"
img = np.array(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
dft(img)
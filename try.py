from cv2 import imread, waitKey
from numba import jit
import numpy as np  
import math
import cv2

img_low = imread("./frog_low.jpg", cv2.IMREAD_GRAYSCALE)
dft_img = cv2.dft(np.float32(img_low), flags = cv2.DFT_COMPLEX_OUTPUT)
file1 = open("./dftimg2", 'w')
file1.write(str(dft_img))



from cv2 import waitKey
from numba import jit
import numpy as np  
import math
import cv2
import time

#@jit
def dft1(img:np.array):
    pi = math.pi
    exp = np.exp
    size = img.shape
    img_height = size[0]
    img_width = size[1]
    dft_arr = np.zeros((img_height, img_width, 2), dtype = complex)
    start_time = time.time()
    for u in range(0, img_height):
        for v in range(0, img_width):
            for x in range(0, img_height):
                for y in range(0, img_width):
                    dft_arr[u][v] = dft_arr[u][v] + (img[x][y]*exp(-1j*2*pi*((u*x/img_height)+(v*y/img_width))))
                    print(x,y)
    imag_part = np.imag(dft_arr[:, :, 0])
    real_part = np.real(dft_arr[:, :, 0])
    dft_arr[:, :, 1] = imag_part
    dft_arr[:, :, 0] = real_part 
    dft_arr = dft_arr.astype("float32")#change the format of des_arr
    end_time = time.time()
    print("spent", end_time-start_time, "seconds")
    return dft_arr

def dft2(img:np.array):
    dft_arr = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    return dft_arr

img_path = "./frog_low.jpg"
img = np.array(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))

dft_arr1 = dft1(img)
file1 = open("./dft_arr1", 'w')
file1.write(str(dft_arr1))

dft_arr2 = dft2(img)
file1 = open("./dft_arr2", 'w')
file1.write(str(dft_arr2))
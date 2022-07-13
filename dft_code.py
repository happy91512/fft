import numpy as np  
import math
import cv2
import time
from numba import jit

@jit(nopython = True)
def dft_2dim(img:np.array):
    pi = math.pi
    exp = np.exp
    size = img.shape
    img_height = size[0]
    img_width = size[1]
    dft_arr = np.zeros((img_height, img_width, 2), dtype = np.complex128)
    # start_time = time.time()
    for u in range(0, img_height):
        print(u)
        for v in range(0, img_width):
            for x in range(0, img_height):
                for y in range(0, img_width):
                    dft_arr[u][v] = dft_arr[u][v] + (img[x][y]*exp(-1j*2*pi*((u*x/img_height)+(v*y/img_width))))
    imag_part = np.imag(dft_arr[:, :, 0])
    real_part = np.real(dft_arr[:, :, 0])
    dft_arr[:, :, 1] = imag_part
    dft_arr[:, :, 0] = real_part
    return dft_arr

def dft_cv2(img:np.array):
    dft_arr = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    return dft_arr

if __name__ == "__main__":
    img_path = "./froglow.jpg"
    img = np.array(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))

    start_t = time.time()
    dft_arr1 = dft_2dim(img)
    dft_arr1 = dft_arr1.astype(np.float32)
    print(f"{time.time()- start_t} sec.")
    file1 = open("./dft_arr1", 'w')
    file1.write(str(dft_arr1))

    start_t = time.time()
    dft_arr2 = dft_cv2(img)
    print(f"{time.time()- start_t} sec.")
    file2 = open("./dft_arr2", 'w')
    file2.write(str(dft_arr2))

    arr_dif = dft_arr1 - dft_arr2
    file3 = open("./arr_dif", 'w')
    file3.write(str(arr_dif))
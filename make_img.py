import cv2
from cv2 import imread
from cv2 import imwrite

img = cv2.imread("./frogg.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (64,64), interpolation=cv2.INTER_AREA)
imwrite("./froglow.jpg", img)

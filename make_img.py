import cv2
from cv2 import imread
from cv2 import imwrite

img = cv2.imread("./frogg.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (32,32), interpolation=cv2.INTER_AREA)
imwrite("./frolow.jpg", img)

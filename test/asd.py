import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('pic.png',0)
surf = cv2.xfeatures2d.SURF(400)
# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img,None)
len(kp)
#plt.imshow(img)
#plt.show()

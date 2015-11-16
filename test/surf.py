import cv2
import numpy as np
import itertools

img = cv2.imread("pic_crop.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread("brick.png")
template_gray=cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
from matplotlib import pyplot as plt


sift = cv2.xfeatures2d.SIFT_create(1000)

# keypoints and descriptors for the images
ikp, ides = sift.detectAndCompute(img_gray, None)
tkp, tdes = sift.detectAndCompute(template_gray, None)

# create BFMatcher Object
matcher = cv2.BFMatcher()
# match descriptors
matches = matcher.match(ides, tdes)
# sort the matches in the order of their distance
matches = sorted(matches, key = lambda x:x.distance)

cv2.drawKeypoints(img_gray, ikp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('ikp.jpg', img)
cv2.drawKeypoints(template_gray, tkp, template, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('tkp.jpg', template)

# draw first 10 matches
img_match = cv2.drawMatches(img,ikp,template,tkp,matches[:4], img, flags=2)
cv2.imwrite('img_match.jpg', img_match) 

plt.imshow(img_match), plt.show()

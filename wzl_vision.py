#!/usr/bin/env python

'''
Skript to match two images based on the SURF Algorithm
of OpenCV, integrated in the WZL Smart Factory

The loop looks like this:
1. wait for an ID from the MES
2. get the pictures for this ID from the ERP server
3. extract Keypoints for those two images
4. take a picture
5. extract keypoints from the camera picture
5. compare both of the images with the camera picture
6. determine which orientation the brick will have
7. send the orientation back to the MES
8. goto step 1
'''

import numpy
import cv2
import os
import sys
import requests
import urllib
import json
import math
import cPickle as pickle
from image import *
import shutil
from mysocket import *
import matplotlib.pyplot as plt

# Globals
order_url_start = 'http://134.130.232.9:50000/AppParkServer/BrickServlet?orderID='
order_url_end = '&resourceType=json'
pic_url_start = 'http://134.130.232.9:50000/AppParkServer/BrickServlet?filename='
db_name = 'image_prop_database.pickle'
pairs_threshold = 5

# extract the keypoints and the descriptors out of a sample image
def extract_keypoints(img):
    # check opencv version because the syntax changed
    major, minor, _ = cv2.__version__.split(".")
    #print 'opencv version: major %s, minor %s' % (major,minor)
    if major == '2':
        # double hessianThreshold, int nOctaves, int nOctaveLayers, bool
        # extended, bool upright
        detector = cv2.SURF(400, 4, 2, 0, 1)
    else:
        detector = cv2.xfeatures2d.SURF_create(400, 4, 2, 0, 1)
    kp, desc = detector.detectAndCompute(img, None)
    # draw keypoints for debug purpose
    #imm=cv2.drawKeypoints(img, kp, None);
    #cv2.imshow("Image", imm);
    #cv2.waitKey(0)
    return kp, desc

# Image Matching
def match_images(kp1, desc1, kp2, desc2):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    if desc2 is not None:
        raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2)
    else:
        print('************ Warning, no descriptors found! **************')
        return None
    kp_pairs = filter_matches(kp1, kp2, raw_matches)
    return kp_pairs, raw_matches

# filter the matches by euclidean distance
def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    kp_pairs = zip(mkp1, mkp2)
    return kp_pairs
    
# Match Diplaying
def explore_match(img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = numpy.zeros((max(h1, h2), w1+w2), numpy.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = numpy.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = numpy.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        is_rect = is_rectangular(corners)
        if not is_rect:
            return None
        cv2.polylines(vis, [corners], True, (51, 255, 255))

    if status is None:
        status = numpy.ones(len(kp_pairs), numpy.bool_)
    p1 = numpy.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = numpy.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)
    return vis
  
def match_inliers(kp_pairs):
    """Draws the matches for """
    mkp1, mkp2 = zip(*kp_pairs)
    
    p1 = numpy.float32([kp.pt for kp in mkp1])
    p2 = numpy.float32([kp.pt for kp in mkp2])
   
    H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    if status is not None:
        inliers = numpy.sum(status)
        matched = len(status)
    return status, H
    
# check if the four points span a rectangle
def is_rectangular(corners):
    A = corners[0]
    B = corners[1]
    C = corners[2]
    D = corners[3]
    dist_AD = math.sqrt(math.pow(A[0] - D[0], 2) + math.pow(A[1] - D[1], 2))
    #print ('distance AD: %d' % dist_AD)
    dist_BC = math.sqrt(math.pow(B[0] - C[0], 2) + math.pow(B[1] - C[1], 2))
    #print ('distance BC: %d' % dist_BC)
    # check if the diagonals are almost equal and if the distances are valid
    angles = []
    if abs(dist_AD - dist_BC) < 50 and dist_AD > 80 and dist_BC > 80:
        angles.append(calc_angle(A-B, A-D))
        angles.append(calc_angle(B-A, B-C))
        angles.append(calc_angle(C-B, C-D))
        angles.append(calc_angle(D-C, D-A))
        angle_sum = sum(angles)
        #print('angle_sum = %f' % angle_sum)
        # first check if sum is in threshold
        if (angle_sum > 340 and angle_sum < 380):
            # check if every angle is around 90 degree
            angle_max = 110
            angle_min = 70
            threshold_check = []
            for angle in angles:
                threshold_check.append(is_angle_in_threshold(angle, angle_min, angle_max))
            if sum(threshold_check) == 4:
                return True
    else:
        return False

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

#Returns the angle in radians between vectors 'v1' and 'v2'
def calc_angle(v1, v2):
    return math.degrees(math.acos(dotproduct(v1, v2) / (length(v1) * length(v2))))

def is_angle_in_threshold(angle, angle_min, angle_max):
    if (angle < angle_max and angle > angle_min):
        return True
    else:
        return False

# saves the two images for a brick
# takes the orderID and the position of the brick, which is a number from 1-6
# from bottom to top
def save_brick_pics(order_id, brick_id, position):
    print 'Retrieving bricks pictures from the server...'
    order = requests.get(order_url_start + str(order_id) + order_url_end).json()
    for brick in order:
        if brick['brickID'] == int(brick_id):
            images = brick['images']
            for image in images:
                if not os.path.isdir('images'):
                    os.makedirs('images')
                image_folder = 'images'
                if (position <=2) or (position >= 5): #bottom or top row
                    print 'top or bottom row'
                    if 'left' in image:
                        urllib.urlretrieve(pic_url_start + image['left'],
                                image_folder + '/left.png')
                        print 'Saved left.png'
                    if 'right' in image:
                        urllib.urlretrieve(pic_url_start + image['right'],
                                image_folder + '/right.png')
                        print 'Saved right.png'
                else:
                    if 'front' in image:
                        urllib.urlretrieve(pic_url_start + image['front'],
                                image_folder + '/front.png')
                        print 'Saved front.png'
                    if 'back' in image:
                        urllib.urlretrieve(pic_url_start + image['back'],
                                image_folder + '/back.png')
                        print 'Saved back.png'

# compare an image from the server with the camera image
def extract_and_compare(cam_pic, brick_pic):
    brick_pic = cv2.imread(brick_pic,0)
    kp, desc = extract_keypoints(brick_pic)
    kp2, desc2 = extract_keypoints(cam_pic)

    kp_pairs, matches = match_images(kp, desc, kp2, desc2)

    if kp_pairs is not None and len(kp_pairs) >= pairs_threshold:
        status, H = match_inliers(kp_pairs)
        if status is not None:
            vis = explore_match(brick_pic, cam_pic, kp_pairs, status, H)
            if vis is not None:
                match = True
                print '################ ITS A MATCH (%s) ####################'
                imgplot = plt.imshow(vis)
                plt.show()
                plt.show()
                return True
    else:
        return False

def save_keypoints():
    print 'Extracting keypoints out of the sample images...'
    image_list = []
    filenames = ['/back.png', '/front.png', '/left.png', '/right.png']
    # os.walk returns a three tuple where index 1 is a list of containing folder
    # names
    for dirs in os.walk('../images'):
        #sort them, else they would be arbitrary ordered
        for brick_id in sorted(dirs[1], key=int):
            for filename in filenames:
                filepath = '../images/' + brick_id + filename
                if os.path.isfile(filepath):
                    sample = cv2.imread(filepath, 0)
                    kp, desc = extract_keypoints(sample)
                    # Store the keypoints
                    temp_image = Image(kp, desc, filename, brick_id)
                    image_list.append(temp_image)
    print 'Keypoints for %d images extraced' % len(image_list)
    print 'Dump keypoints database to \'%s\'...' % db_name
    pickle.dump(image_list, open(db_name, 'wb'))

# Main
if __name__ == '__main__':
    sock = mysocket() #create a new socket instance
    print 'Connecting to server...'
    sock.connect()
    print 'Connection successful' 

    try_number = 1
    while True:
        print 'Waiting to receive an ID...'
        reply = sock.receive()
        if reply is "":
            continue
        print 'Received: ' + reply
        #split the reply which is separated by semilkolons
        splitted_reply = reply.split(';')
        order_id = splitted_reply[0]
        brick_id = splitted_reply[1]
        position = splitted_reply[2]
        # 1. orderID, brickID, position von unten nach oben 1-6,
        save_brick_pics(order_id, brick_id, position)

        # saves a camera picture as single.bmp
        if sys.platform.startswith('linux'):
            cmd = '../bin/SingleCaptureStorage'
        elif sys.platform.startswith('win32'):
            cmd = 'SingleCaptureStorage.exe'
        os.system(cmd)
        cam_pic = cv2.imread('single.bmp',0)
        cam_pic = cv2.resize(cam_pic, None, fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)

        # It's the bottom or the the top row
        if position <= 2 or position >= 5:
            if extract_and_compare(cam_pic, 'images/left.png'):
                print 'Its left'
                sock.send('True')
                continue
            elif extract_and_compare(cam_pic, 'images/right.png'):
                print 'Its right'
                sock.send('False')
                continue
        else: # It's the middle row 
            if extract_and_compare(cam_pic, 'images/front.png'):
                print 'Its front'
                sock.send('True')
                continue
            elif extract_and_compare(cam_pic, 'images/back.png'):
                print 'Its back'
                sock.send('False')
                continue
        #All went wrong
        sock.send('False')

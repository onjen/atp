#!/usr/bin/env python

'''
Documentation
'''

import numpy
import cv2
import os
import sys
import requests
import urllib
import json
import math


# Globals
order_url_start = 'http://134.130.232.9:50000/AppParkServer/BrickServlet?orderID='
order_url_end = '&resourceType=json'
pic_url_start = 'http://134.130.232.9:50000/AppParkServer/BrickServlet?filename='
pairs_threshold = 5

# Image Matching
def match_images(img1, img2):
    """Given two images, returns the matches"""
    detector = cv2.xfeatures2d.SURF_create(400, 5, 5)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    print 'sample - %d features, cam_pic - %d features' % (len(kp1), len(kp2))

    if desc2 is not None:
            raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
    else:
        print('************ Warning, no descriptors extracted! **************')
        return None
    kp_pairs = filter_matches(kp1, kp2, raw_matches)
    return kp_pairs

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
def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = numpy.zeros((max(h1, h2), w1+w2), numpy.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = numpy.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = numpy.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        print corners
        is_rect = is_rectangular(corners)
        #TODO change the method calls
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

    cv2.imshow(win, vis)
    cv2.waitKey()
  
def draw_matches(window_name, kp_pairs, img1, img2):
    """Draws the matches for """
    mkp1, mkp2 = zip(*kp_pairs)
    
    p1 = numpy.float32([kp.pt for kp in mkp1])
    p2 = numpy.float32([kp.pt for kp in mkp2])
    
    if len(kp_pairs) >= 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    if status is not None:
        inliers = numpy.sum(status)
        matched = len(status)
        print( '%d / %d    inliers/matched quotient=%d' % (inliers, matched, matched/inliers))
        if (matched/inliers < 25):
           explore_match(window_name, img1, img2, kp_pairs, status, H)
    else:
        H, status = None, None
        #print '%d matches found, not enough for homography estimation' % len(p1)
    
# check if the four points span a rectangle
def is_rectangular(corners):
    A = corners[0]
    B = corners[1]
    C = corners[2]
    D = corners[3]
    dist_AD = math.sqrt(math.pow(A[0] - D[0], 2) + math.pow(A[1] - D[1], 2))
    print ('distance AD: %d' % dist_AD)
    dist_BC = math.sqrt(math.pow(B[0] - C[0], 2) + math.pow(B[1] - C[1], 2))
    print ('distance BC: %d' % dist_BC)
    # check if the diagonals are almost equal and if the distances are valid
    # TODO dependent on the resolution
    if abs(dist_AD - dist_BC) < 20 and dist_AD > 80 and dist_BC > 80:
        angle_A = py_ang(A-B, A-D)
        angle_B = py_ang(B-A, B-C)
        angle_C = py_ang(C-B, C-D)
        angle_D = py_ang(D-C, D-A)
        print ('angles A: %f, B: %f, C: %f, D: %f' % (angle_A, angle_B, angle_C,
            angle_D))
        angle_sum = angle_A + angle_B + angle_C + angle_D
        print('angle_sum = %f' % angle_sum)
        if (angle_sum > 340 and angle_sum < 380):
            return True
    else:
        return False

#Returns the angle in radians between vectors 'v1' and 'v2'
def py_ang(v1, v2):
    cosang = numpy.dot(v1, v2)
    sinang = numpy.linalg.norm(numpy.cross(v1, v2))
    return math.degrees(numpy.arctan2(sinang, cosang))

# get all brick pictures from active orders in the database
def get_picture_db():
    r = requests.get('http://134.130.232.9:50000/AppParkServer/OrderServlet?progress=99')
    json_request = r.json()
    orders = []
    bricks = []
    # get all bricks
    for i in json_request:
        # an order looks like this:
        # {"brickID":21914,"color":15,"hasStuds":1,"offsetX":3,"offsetY":0,"offsetZ":3,"sizeX":4,"sizeZ":2,
        # "images":[{"front":"brickID21914front.png"},{"back":"template303back.png"},
        # {"left":"template303left.png"},{"right":"template303right.png"}]}
        order = requests.get(order_url_start + str(i['orderID']) + order_url_end).json()
        for brick in order:
            bricks.append(brick)
        orders.append(order)

    if not os.path.isdir('../images'):
        os.makedirs('../images')

    for brick in bricks:
        brick_folder = '../images/' + str(brick['brickID'])
        if not os.path.isdir(brick_folder):
            os.makedirs(brick_folder)
        # TODO remove old folders
        images = brick['images']
        for image in images:
            if 'front' in image:
                urllib.urlretrieve(pic_url_start + image['front'], brick_folder + '/front.png')
            if 'back' in image:
                urllib.urlretrieve(pic_url_start + image['back'], brick_folder + '/back.png')
                if 'left' in image:
                    urllib.urlretrieve(pic_url_start + image['left'], brick_folder + '/left.png')
                if 'right' in image:
                    urllib.urlretrieve(pic_url_start + image['right'], brick_folder + '/right.png') 

# Main
if __name__ == '__main__':
    # TODO when adding an order calculate keypoints and so on
    # TODO kp_pairs threshold dynamically, dependant on the features?
    # TODO lego2 and lego3 aren't recognised

    cam_pic = cv2.imread('lego6.png', 0)

    # os.walk returns a three tuple where 0 is the folder name
    #sample = cv2.imread('images/20141/back.png', 0)
    #kp_pairs = match_images(sample, cam_pic)
    #print("Length of kp_pairs = %d" % len(kp_pairs))
    #draw_matches('find_obj', kp_pairs, sample , cam_pic)
    for i in os.walk('images'):
         if os.path.isfile(i[0] + '/back.png'):
             sample = cv2.imread(i[0] + '/back.png', 0)
             kp_pairs = match_images(sample, cam_pic)
             if kp_pairs is not None and len(kp_pairs) >= pairs_threshold:
                 draw_matches('find_obj', kp_pairs, sample , cam_pic)
         if os.path.isfile(i[0] + '/front.png'):
             sample = cv2.imread(i[0] + '/front.png', 0)
             kp_pairs = match_images(sample, cam_pic)
             if kp_pairs is not None and len(kp_pairs) >= pairs_threshold:
                 draw_matches('find_obj', kp_pairs, sample , cam_pic)
         if os.path.isfile(i[0] + '/left.png'):
             sample = cv2.imread(i[0] + '/left.png', 0)
             kp_pairs = match_images(sample, cam_pic)
             if kp_pairs is not None and len(kp_pairs) >= pairs_threshold:
                 draw_matches('find_obj', kp_pairs, sample , cam_pic)
         if os.path.isfile(i[0] + '/right.png'):
             sample = cv2.imread(i[0] + '/right.png', 0)
             kp_pairs = match_images(sample, cam_pic)
             if kp_pairs is not None and len(kp_pairs) >= pairs_threshold:
                 draw_matches('find_obj', kp_pairs, sample , cam_pic)


#!/usr/bin/env python

'''
Documentation
different filters (matches, quotient, geometrical and so on)
performance (serialize and store the keypoints, save to database)
TODO
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

# Globals
order_url_start = 'http://134.130.232.9:50000/AppParkServer/BrickServlet?orderID='
order_url_end = '&resourceType=json'
pic_url_start = 'http://134.130.232.9:50000/AppParkServer/BrickServlet?filename='
pairs_threshold = 5

# extract the keypoints and the descriptors out of a sample image
def extract_keypoints(img):
    # double hessianThreshold, int nOctaves, int nOctaveLayers, bool
    # extended, bool upright
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
        print( '%d / %d    inliers/matched quotient=%d' % (inliers, matched, matched/inliers))
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
    # TODO dependent on the resolution
    if abs(dist_AD - dist_BC) < 50 and dist_AD > 80 and dist_BC > 80:
        angle_A = angle(A-B, A-D)
        angle_B = angle(B-A, B-C)
        angle_C = angle(C-B, C-D)
        angle_D = angle(D-C, D-A)
        #print ('angles A: %f, B: %f, C: %f, D: %f' % (angle_A, angle_B, angle_C,
        #    angle_D))
        angle_sum = angle_A + angle_B + angle_C + angle_D
        #print('angle_sum = %f' % angle_sum)
        if (angle_sum > 340 and angle_sum < 380):
            return True
    else:
        return False

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

#Returns the angle in radians between vectors 'v1' and 'v2'
def angle(v1, v2):
  return math.degrees(math.acos(dotproduct(v1, v2) / (length(v1) * length(v2))))

# get a list of bricks from active orders in the database
# @return a list of JSON objects containing brick data
def get_brick_db():
    print 'Retrieving bricks from the server...'
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
    return bricks
'''
    if not os.path.isdir('../images'):
        os.makedirs('../images')

    for brick in bricks:
        brick_folder = '../images/' + str(brick['brickID'])
        if not os.path.isdir(brick_folder):
            os.makedirs(brick_folder)
        # TODO remove old folders
        images = brick['images']
        #TODO all as else if and if no image is in there remove folder
        for image in images:
            if 'front' in image:
                urllib.urlretrieve(pic_url_start + image['front'], brick_folder + '/front.png')
            if 'back' in image:
                urllib.urlretrieve(pic_url_start + image['back'], brick_folder + '/back.png')
            if 'left' in image:
                urllib.urlretrieve(pic_url_start + image['left'], brick_folder + '/left.png')
            if 'right' in image:
                urllib.urlretrieve(pic_url_start + image['right'], brick_folder + '/right.png') 
'''
def save_keypoints():
    # TODO update with new order
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
    db_name = 'image_prop_database.pickle'
    print 'Dump keypoints database to \'%s\'...' % db_name
    pickle.dump(image_list, open(db_name, 'wb'))

# reformat to de-serialize keypoints and descriptors
def unpickle_keypoints(image):
    keypoints = []
    descriptors = []
    for point in image.keypoints_array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        keypoints.append(temp_feature)
        temp_descriptor = point[6]
        descriptors.append(temp_descriptor)
    return keypoints, numpy.array(descriptors)

# the pickled database contains all images, so we have multiple brick_id's
# because on brick has several images (e.g. front.png, back.png...)
def remove_duplicate_bricks(image_list):
    checked = []
    for brick in image_list:
        brick_id = brick.brick_id
        if brick_id not in checked:
            checked.append(brick_id)
    # convert to int list
    checked = map(int, checked)
    return checked

# compare the brick id's from the server and in meomory
# bricks_from_database is a JSON object containing brick data
# image_list is the list of image_objects which is stored
def compare_brick_ids(bricks_from_database, image_list):
    bricks_from_database_formatted = []
    bricks_saved = remove_duplicate_bricks(image_list)
    for brick in bricks_from_database:
        bricks_from_database_formatted.append(brick['brickID'])

    for i in range(0, len(bricks_saved)):
        print( 'database: %d, saved: %d' %
            (bricks_from_database_formatted[i],bricks_saved[i]))

# Main
if __name__ == '__main__':
    # TODO when adding an order calculate keypoints and so on
    # TODO kp_pairs threshold dynamically, dependant on the features?
    # TODO lego2 and lego3 aren't recognised
    # TODO check if every angle is rougly around 90 degree
    # TODO remember to free memory
    # time when extracting all keypoints of the samples 14s real

    #save_keypoints()

    # saves a camera picture as single.bmp
    #cmd = '../bin/SingleCaptureStorage'
    #os.system(cmd)
    #cam_pic = cv2.imread('../bin/single.bmp',0)
    cam_pic = cv2.imread('lego5.png', 0)
    image_list = []
    # TODO make sure it wrote before trying to load
    print 'Loading saved pickle database...'
    image_list = pickle.load(open('image_prop_database.pickle', 'rb'))
    # get a list of JSON objects containing brick data
    bricks = get_brick_db()
    compare_brick_ids(bricks, image_list)

    kp2, desc2 = extract_keypoints(cam_pic)

    for image_obj in image_list:
        kp1, desc1 = unpickle_keypoints(image_obj)
        kp_pairs, matches = match_images(kp1, desc1, kp2, desc2)

        if kp_pairs is not None and len(kp_pairs) >= pairs_threshold:
            status, H = match_inliers(kp_pairs)
            if status is not None:
                sample = cv2.imread('../images/' + image_obj.brick_id + image_obj.filename, 0)
                vis = explore_match(sample, cam_pic, kp_pairs, status, H)
                if vis is not None:
                    print '################ ITS A MATCH ####################'
                    #cv2.imshow('wzl_vision', vis)
                    #cv2.waitKey()

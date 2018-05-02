import os
import sys
import numpy as np
import cv2
import numpy.matlib
import dlib

from scipy.io import loadmat
from scipy.ndimage import imread, affine_transform
import nudged

def get_facial_points(image, num_points):
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    detector = dlib.get_frontal_face_detector()
    dets = detector(image, 1)
    points = []
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = predictor(image, d)
        for i in range(num_points):
            pt = shape.part(i)
            points.append([int(pt.x), int(pt.y)])
    return np.array(points)

def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max # Divide all values by the largest possible value in the datatype

def normalize_image(imat):
    rgb = np.zeros((imat.shape[0], imat.shape[1], 3), dtype=np.float)
    rgb[:, :, 0] = (imat[:, :, 2] - 104.008)/255
    rgb[:, :, 1] = (imat[:, :, 1] - 116.669)/255
    rgb[:, :, 2] = (imat[:, :, 0] - 122.675)/255
    return rgb


tracker_path = 'data/images_tracker/00047'
reftracker = loadmat(tracker_path)['tracker']
reftracker = get_facial_points(imread('/Users/yu-chieh/Downloads/images_data_crop/00047.jpg'), 49)



refpos = np.floor(reftracker.mean(axis=0))

xxc, yyc = np.meshgrid(np.linspace(1,1800, 1800), np.linspace(1,2000, 2000))
xxc = (xxc-600-refpos[0])/600
yyc = (yyc-600-refpos[1])/800
mmask_png =  'meanmask.png'
maskc = im2double(imread(mmask_png))
maskc = np.pad(maskc, [600,600], 'constant')



def get_processed_image(img):
    image = imread(img)
    a = get_facial_points(image, 49)
    b = reftracker
    trans = nudged.estimate(a, b);
    h = np.array(trans.get_matrix())[0:2]
    # h, status = cv2.findHomography(a, b)
    warpedxx = np.array(cv2.warpAffine(xxc, h, xxc.shape))
    warpedyy = np.array(cv2.warpAffine(yyc, h, yyc.shape))
    warpedmask = np.array(cv2.warpAffine(maskc, h, maskc.shape))
    # warpedxx = np.array(cv2.warpPerspective(xxc, h, xxc.shape))
    # warpedyy = np.array(cv2.warpPerspective(yyc, h, yyc.shape))
    # warpedmask = np.array(cv2.warpPerspective(maskc, h, maskc.shape))
    warpedxx = warpedxx[600:1400, 600:1200]
    warpedyy = warpedyy[600:1400, 600:1200]
    warpedmask = warpedmask[600:1400, 600:1200]
    filtered_img = np.zeros((image.shape[0], image.shape[1], 6))
    filtered_img[:, :, 0:3] = normalize_image(image)
    filtered_img[:, :, 3] = warpedxx
    filtered_img[:, :, 4] = warpedyy
    filtered_img[:, :, 5] = warpedmask
    return filtered_img


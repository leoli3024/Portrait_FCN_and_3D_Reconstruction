import skimage.io as io
import numpy as np
import scipy
import cv2
import dlib



image = scipy.ndimage.imread('res/org0.jpg', mode='RGB')

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



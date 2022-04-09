import cv2
import numpy as np

from Utils.StereoUtils import splitMergedImage
from Utils.ImageUtils import drawEpiline
from Objects.BRIEF import DetectorBRIEF


def getCenterPair(image, F):
    image_left, image_right = splitMergedImage(image)
    height, width = image_left.shape

    detector = DetectorBRIEF()

    point_left = np.array([(int)(width / 2), (int)(height / 2)]).reshape(1, 2)

    points = []
    x_arr = range(0, width)

    line = cv2.computeCorrespondEpilines(point_left, 1, F)

    a = line[0][0][0]
    b = line[0][0][1]
    c = line[0][0][2]

    for x in x_arr:
        y = (int)(-(c + a * x) / b)
        points.append([x, y])

    vectors_a = detector.getParamVectors(image_left, point_left)
    vectors_b = detector.getParamVectors(image_right, points)

    pair = detector.calculatePairs(vectors_a, vectors_b)

    if len(pair) > 0:
        pair = pair[0]
        return [pair[0].getCoord(), pair[1].getCoord()]
    else:
        return None

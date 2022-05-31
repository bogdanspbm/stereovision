import cv2
import numpy as np
import sys

from Utils.ImageUtils import splitMergedImage, getScanline
from Objects.KeyPoint import getHausdorfDistance
from Objects.BRIEF import DetectorBRIEF
from Objects.Timer import Timer
import matplotlib.pyplot as plt
import Utils.MathUtils as MathUtils

'''
This method calculates epipolar line for a center line
using Fundamental matrix, image id (1 - if main image is left, other if main image is right)
and image shape
'''


def getEpilineCenter(F, image=1, shape=(1920, 1080)):
    width, height = shape
    point_a = np.array([0, 0]).reshape(1, 2)
    point_b = np.array([(int)(width / 2), (int)(height / 2)]).reshape(1, 2)

    flag = 1

    if image == 1:
        flag = 2
    else:
        flag = 1

    line_a = cv2.computeCorrespondEpilines(point_a, flag, F)
    line_b = cv2.computeCorrespondEpilines(point_b, flag, F)

    a_1 = line_a[0][0][0]
    b_1 = line_a[0][0][1]
    c_1 = line_a[0][0][2]

    a_2 = line_b[0][0][0]
    b_2 = line_b[0][0][1]
    c_2 = line_b[0][0][2]

    t_2 = (b_2 * c_1 - b_1 * c_2) / (b_1 - b_2 * a_1 / a_2)
    t_1 = a_1 * t_2 / a_2

    x_2 = t_2 / a_2
    y_2 = (t_2 + c_2) / (-b_2)

    x_1 = t_1 / a_1
    y_1 = (t_1 + c_1) / (-b_1)

    return [x_1, y_1]


'''
This method finds an epipolar lines pair on two images
for any point, using point coordinate, Fundamental matrix, image id and image shape
'''


def findBothEpilines(point, F, image=1, shape=(1920, 1080)):
    epiline_center = getEpilineCenter(F, image, shape)
    line_a = cv2.computeCorrespondEpilines(np.array(point).reshape(1, 2), image, F)

    a_1 = line_a[0][0][0]
    b_1 = line_a[0][0][1]
    c_1 = line_a[0][0][2]

    a_2 = epiline_center[1] - point[1]
    b_2 = point[0] - epiline_center[0]
    c_2 = point[1] * (epiline_center[0] - point[0]) - point[0] * (epiline_center[1] - point[1])

    return [a_1, b_1, c_1], [a_2, b_2, c_2]


'''
This method draws epipolar lines on image using image, poinr, Fundamental matrix and image id
'''


def drawBothEpilines(image, point, F, image_mode=1):
    image_left, image_right = splitMergedImage(image)
    height, width = image_left.shape
    line_a, line_b = findBothEpilines(point, F, image_mode, (width, height))

    x_a_1 = 0
    x_a_2 = width
    y_a_1 = (int)((-line_a[0] * x_a_1 - line_a[2]) / (line_a[1]))
    y_a_2 = (int)((-line_a[0] * x_a_2 - line_a[2]) / (line_a[1]))

    x_b_1 = 0
    x_b_2 = width
    y_b_1 = (int)((-line_b[0] * x_b_1 - line_b[2]) / (line_b[1]))
    y_b_2 = (int)((-line_b[0] * x_b_2 - line_b[2]) / (line_b[1]))

    cv2.line(image_left, (x_a_1, y_a_1), (x_a_2, y_a_2), (0, 0, 255))
    cv2.line(image_right, (x_b_1, y_b_1), (x_b_2, y_b_2), (0, 0, 255))

    frame = cv2.hconcat([image_left, image_right])

    return frame


'''
This method finds center stereo pair using BRIEF detector
Epipolar line calculates by Fundamental matrix
'''


def getCenterPair(image, F):
    image_left, image_right = splitMergedImage(image)
    height, width = image_left.shape

    detector = DetectorBRIEF()

    point_left = np.array([(int)(width / 2), (int)(height / 2)]).reshape(1, 2)

    points = []
    x_arr = range(0, (int)(width / 2))

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


'''
This method finds center stereo pair using BRIEF detector
Epipolar line is taken as horizontal
'''


def getCenterPair(image):
    image_left, image_right = splitMergedImage(image)
    height, width = image_left.shape

    detector = DetectorBRIEF()

    point_left = np.array([(int)(width / 2), (int)(height / 2)]).reshape(1, 2)

    points = []
    x_arr = range((int)(width / 2), width)

    for x in x_arr:
        y = (int)(height / 2)
        points.append([x, y])

    vectors_a = detector.getParamVectors(image_left, point_left)
    vectors_b = detector.getParamVectors(image_right, points)

    X = []
    Y = []

    for vec in vectors_b:
        X.append(vec.coord[0])
        Y.append(getHausdorfDistance(vec, vectors_a[0]))

    plt.plot(X, Y)
    plt.show()

    pair = detector.calculatePairs(vectors_a, vectors_b)

    if len(pair) > 0:
        pair = pair[0]
        return [pair[0].getCoord(), pair[1].getCoord()]
    else:
        return None


'''
This method finds any stereo pairs using STAR keypoint detection and BRIEF descriptor
'''


def getAnyPairs(image):
    timer = Timer()
    pairs = []
    image_left, image_right = splitMergedImage(image)

    star = cv2.xfeatures2d.StarDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    timer.refreshTimer()
    kp_left = star.detect(image_left, None)
    kp_left, des_left = brief.compute(image_left, kp_left)
    kp_right = star.detect(image_right, None)
    kp_right, des_right = brief.compute(image_right, kp_right)

    matches = bf.match(des_left, des_right)
    matches = sorted(matches, key=lambda x: x.distance)

    for match in matches:
        pair = [[kp_left[match.queryIdx].pt],
                [kp_right[match.trainIdx].pt]]
        pairs.append(pair)

    return pairs


'''
This method finds stereo pair for a center point using Block Matching algorythm
'''


def getCenterPairBlockMatching(image, block_radius=10, scale=2):
    height, width = image.shape

    image_small = cv2.resize(image.copy(), ((int)(width / scale), (int)(height / scale)),
                             interpolation=cv2.INTER_LINEAR)
    small_radius = int(block_radius / scale)

    blur_small = cv2.GaussianBlur(image_small,
                                  (small_radius + (small_radius + 1) % 2, small_radius + (small_radius + 1) % 2),
                                  small_radius * 2)

    image_left, image_right = splitMergedImage(blur_small)
    height, width = image_left.shape
    process_image = cv2.copyMakeBorder(
        image_right,
        0,
        0,
        small_radius,
        small_radius,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0))

    N = (int)(width / 2)
    H = (int)(height / 2)
    X = []
    Y = []

    point_left = np.array([N, (int)(height / 2)]).reshape(1, 2)
    main_block = image_left[int(H - small_radius):int(H + small_radius),
                 int(point_left[0][0] - small_radius):int(point_left[0][0] + small_radius)]

    min_SAD = -1
    sub_center = -1
    for i in range(N - small_radius, width):
        X.append(i * 2)
        sub_block = process_image[int(H - small_radius):int(H + small_radius),
                    int(i - small_radius):int(i + small_radius)]
        SAD = np.sum(np.absolute(main_block - sub_block) ** 2)
        Y.append(SAD)

        if min_SAD == -1 or min_SAD > SAD:
            min_SAD = SAD
            sub_center = i - small_radius

    return [(N * scale, H * scale), (sub_center * scale, H * scale)]


'''
This method finds stereo pair for a center point using epipolar line compare algorythm
'''


def getCenterPairEpipolar(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    image_left, image_right = splitMergedImage(blur)
    height, width = image_left.shape

    center_x = (int)(width / 2)

    epipolar_a = image_left[(int)(height / 2), :]
    epipolar_b = image_right[(int)(height / 2), :]

    ext_a = MathUtils.findExtremums(epipolar_a)
    ext_b = MathUtils.findExtremums(epipolar_b)

    der_a = MathUtils.getDerivative(epipolar_a)
    der_b = MathUtils.getDerivative(epipolar_b)

    fluct_zones_a = MathUtils.getFluctuateZone(der_a)
    fluct_zones_b = MathUtils.getFluctuateZone(der_b)

    left_border, right_border = MathUtils.getFlucluateZoneBorder(fluct_zones_b)

    left_ext_pair, right_ext_pair = MathUtils.findBorderExtremus(ext_b, left_border, right_border)

    left_border, right_border = MathUtils.getFlucluateZoneBorder(fluct_zones_a)

    left_ext, right_ext = MathUtils.findBorderExtremus(ext_a, left_border, right_border)

    left_disp = abs(left_ext[0] - left_ext_pair[0])
    right_disp = abs(right_ext[0] - right_ext_pair[0])

    center_disp = left_disp + (right_disp - left_disp) * (center_x - left_border) / (right_border - left_border)
    center_x_pair = center_x + center_disp

    return [(center_x, (int)(height / 2)), (center_x_pair, (int)(height / 2))]


'''
This method generates scanline map using two epilines
'''


def generateScanlineMap(vec_a, vec_b):
    N = len(vec_a)
    mat = np.abs(np.transpose(np.tile(vec_a, (N, 1))) - np.tile(vec_b, (N, 1)))
    return mat

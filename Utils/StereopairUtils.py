import cv2
import numpy as np

from Utils.StereoUtils import splitMergedImage
from Utils.ImageUtils import drawEpiline
from Objects.KeyPoint import convertToCVKeypoint
from Objects.BRIEF import DetectorBRIEF, DetectorHorizontalBRIEF
from Objects.Detector import HorizontalDetector
from Objects.Timer import Timer
import matplotlib.pyplot as plt


def getCenterPair(image, F):
    image_left, image_right = splitMergedImage(image)
    height, width = image_left.shape

    # detector = DetectorBRIEF()
    detector = DetectorBRIEF()
    timer = Timer()

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


def getCenterPair(image):
    image_left, image_right = splitMergedImage(image)
    height, width = image_left.shape

    # detector = DetectorBRIEF()
    detector = HorizontalDetector()
    timer = Timer()

    point_left = np.array([(int)(width / 2), (int)(height / 2)]).reshape(1, 2)

    points = []
    x_arr = range(0, (int)(width / 2))

    for x in x_arr:
        y = (int)(height / 2)
        points.append([x, y])

    vectors_a = detector.getParamVectors(image_left, point_left)
    vectors_b = detector.getParamVectors(image_right, points)

    pair = detector.calculatePairs(vectors_a, vectors_b)

    if len(pair) > 0:
        pair = pair[0]
        return [pair[0].getCoord(), pair[1].getCoord()]
    else:
        return None


def getAnyPairs(image):
    timer = Timer()
    pairs = []
    image_left, image_right = splitMergedImage(image)
    height, width = image_left.shape

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

    image = cv2.drawMatches(image_left, kp_left, image_right, kp_right, matches[:10], None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    for match in matches:
        pair = [[kp_left[match.queryIdx].pt],
                [kp_right[match.trainIdx].pt]]
        pairs.append(pair)

    return pairs


def getCenterPairBlockMatching(image, block_radius=10, scale=2):
    timer = Timer()

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
        sub_block = process_image[int(H - small_radius):int(H + small_radius),
                    int(i - small_radius):int(i + small_radius)]
        SAD = np.sum(np.absolute(main_block - sub_block) ** 2)

        if min_SAD == -1 or min_SAD > SAD:
            min_SAD = SAD
            sub_center = i - small_radius

    return [(N * scale, H * scale), (sub_center * scale, H * scale)]

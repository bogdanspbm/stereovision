import cv2
import numpy as np

from Utils.ImageUtils import splitMergedImage
from Utils.ImageUtils import drawEpiline
from Objects.CameraUndistorter import Undistorter
from Objects.KeyPoint import convertToCVKeypoint, getHausdorfDistance
from Objects.BRIEF import DetectorBRIEF, DetectorHorizontalBRIEF
from Objects.Detector import HorizontalDetector
from Objects.Timer import Timer
import matplotlib.pyplot as plt
import Utils.MathUtils as MathUtils


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


def generateDisparityMap(image, F):
    timer = Timer()
    height, width = image.shape
    resized_down = cv2.resize(image, ((int)(width / 8), (int)(height / 8)), interpolation=cv2.INTER_LINEAR)
    timer.printTime("Resize")
    # blur = cv2.GaussianBlur(image, (21, 21), 10)
    # image_left, image_right = splitMergedImage(blur)
    #undistorter = Undistorter()
    #stereo = cv2.StereoBM_create(numDisparities=16, blockSize=21)
    #image_left, image_right = undistorter.quickUndistort(image)
    #disparity = stereo.compute(image_left, image_right)
    # detected_edges = cv2.Canny(image, 100, 200)
    # detected_edges = cv2.Laplacian(image, cv2.CV_64F)

    # height, width = blur.shape
    # blur_offset = np.append(blur, np.zeros((height, 1)), axis=1)
    # blur_offset = blur[:,0:]

    return resized_down
    # return cv2.hconcat([image_left, image_right])


def generateScanline(image, point, F, image_mode=1):
    timer = Timer()
    blur = cv2.GaussianBlur(image, (21, 21), 10)
    image_left, image_right = splitMergedImage(blur)
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

    X_1 = []
    I_1 = []
    for x in range(x_a_1, x_a_2):
        # y = (int)((-line_a[0] * x - line_a[2]) / (line_a[1]))
        y = point[1]
        if y >= 0 and y < height:
            X_1.append(x)
            I_1.append(image_left[y, x])

    I_2 = []
    X_2 = []
    for x in range(x_b_1, x_b_2):
        # y = (int)((-line_b[0] * x - line_b[2]) / (line_b[1]))
        y = point[1]
        if y >= 0 and y < height:
            X_2.append(x)
            I_2.append(image_right[y, x])

    extremums_a = MathUtils.findExtremums(I_1, separate=False)
    extremums_b = MathUtils.findExtremums(I_2, separate=False)

    points_a = []

    for ext in extremums_a:
        x = ext[0]
        # y = (int)((-line_a[0] * x - line_a[2]) / (line_a[1]))
        y = point[1]
        points_a.append([x, y, ext[2]])

    points_b = []

    for ext in extremums_b:
        x = ext[0]
        # y = (int)((-line_b[0] * x - line_b[2]) / (line_b[1]))
        y = point[1]
        points_b.append([x, y, ext[2]])

    pairs = MathUtils.matchExtremus(image_left, image_right, points_a, points_b)

    disp_pairs = []

    if len(pairs) > 0:
        for x in range(x_a_1, x_a_2):
            y = (int)((-line_a[0] * x - line_a[2]) / (line_a[1]))
            pre_pair = None
            after_pair = None
            for i in range(len(pairs) - 1):
                pair_l = pairs[i]
                pair_r = pairs[i + 1]

                if pair_l[0][0] <= x and pair_r[0][0] >= x:
                    pre_pair = pair_l
                    after_pair = pair_r
                    break

            if pre_pair is None:
                if x < pairs[0][0][0]:
                    after_pair = pairs[0]
                if x > pairs[len(pairs) - 1][0][0]:
                    pre_pair = pairs[len(pairs) - 1]

            disp = 0

            if pre_pair is not None and after_pair is not None:
                start_disp = pre_pair[0][0] - pre_pair[1][0]
                end_disp = after_pair[0][0] - after_pair[1][0]
                disp = (x - pre_pair[0][0]) * end_disp / (after_pair[0][0] - pre_pair[0][0]) + (
                        x - after_pair[0][0]) * start_disp / (pre_pair[0][0] - after_pair[0][0])

            elif after_pair is not None:
                disp = after_pair[0][0] - after_pair[1][0]
            elif pre_pair is not None:
                disp = pre_pair[0][0] - pre_pair[1][0]

            disp_pairs.append(disp)

    return disp_pairs


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

    detector = DetectorBRIEF()
    # detector = HorizontalDetector()
    timer = Timer()

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
        X.append(i * 2)
        sub_block = process_image[int(H - small_radius):int(H + small_radius),
                    int(i - small_radius):int(i + small_radius)]
        SAD = np.sum(np.absolute(main_block - sub_block) ** 2)
        Y.append(SAD)

        if min_SAD == -1 or min_SAD > SAD:
            min_SAD = SAD
            sub_center = i - small_radius

    plt.plot(X, Y)
    # plt.show()

    return [(N * scale, H * scale), (sub_center * scale, H * scale)]

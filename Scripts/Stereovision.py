import numpy as np

import Utils.StereoUtils as StereoUtils
import Utils.FileUtils as FileUtils
import Objects.Camera as Camera
import Utils.ImageUtils as ImageUtils
import Utils.MathUtils as MathUtils
from Objects.Timer import Timer
import Utils.StereopairUtils as StereopairUtils
import cv2


def loadImagesAndDoEplines():
    P_1, P_2 = FileUtils.loadProjectionMatrix()
    timer = Timer()
    images = FileUtils.getFramesImages()
    counter = 1
    radius = 25

    F = FileUtils.loadFundamentalMatrix()

    # StereopairUtils.getEpilineCenter(F)

    for image in images:
        height, width = image.shape
        counter += 1
        timer.refreshTimer()
        frame = []
        scanline = StereopairUtils.generateDisparityMap(image, F)
        timer.printTime("Calc epilines")

        frame = scanline
        print(frame.shape)
        cv2.imwrite("../Result/both_epiline_" + str(counter) + ".png", frame)


def loadImagesAndDoBlockMatching():
    P_1, P_2 = FileUtils.loadProjectionMatrix()
    timer = Timer()
    images = FileUtils.getFramesImages()
    counter = 1
    radius = 25

    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=radius)

    for image in images:
        counter += 1
        height, width = image.shape
        timer.refreshTimer()
        im_left, im_right = StereoUtils.splitMergedImage(image)
        pair = StereopairUtils.getCenterPairBlockMatching(image, radius)
        # pair = StereopairUtils.getCenterPair(image)
        timer.printTime("BM Pair")

        a = np.array(pair[0]) - (radius, radius)
        b = np.array(pair[0]) + (radius, radius)
        cv2.rectangle(image, (a[0], a[1]), (b[0], b[1]), (0, 0, 255), 5)
        a = np.array(pair[1]) - (radius, radius) + (int(width / 2), 0)
        b = np.array(pair[1]) + (radius, radius) + (int(width / 2), 0)
        cv2.rectangle(image, (a[0], a[1]), (b[0], b[1]), (0, 0, 255), 5)

        depth = stereo.compute(im_left, im_right)

        print(depth)
        cv2.imwrite("../Result/stereo_pair_" + str(counter) + ".png", depth)

        point_left = np.array(pair[0]).astype(float)
        point_right = np.array(pair[1]).astype(float)


def openImageAndGetDistance():
    P_1, P_2 = FileUtils.loadProjectionMatrix()
    F = FileUtils.loadFundamentalMatrix()

    image = cv2.imread("../Images/Image_1.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    pair = StereopairUtils.getCenterPairBlockMatching(gray, 21)

    point_left = np.array(pair[0]).astype(float)
    point_right = np.array(pair[1]).astype(float)

    cv2.circle(image, ((int)(width / 4), (int)(height / 2)), 5, (0, 0, 255))
    cv2.circle(image, ((int)(width / 2) + (int)(point_right[0]), (int)(point_right[1])), 5, (0, 0, 255))
    coord = StereoUtils.getWorldCoordinates(P_1, P_2, point_left, point_right)
    dist = MathUtils.getVectorNorm(coord)
    ImageUtils.imageDrawDistance(image, dist)

    cv2.imwrite("../Result/Distance.jpg", image)


def openVideoAndGetDistanceBlockMatching():
    camera = Camera.VirtualCamera(0, 3840, 1080)
    P_1, P_2 = FileUtils.loadProjectionMatrix()
    F = FileUtils.loadFundamentalMatrix()
    counter = 0
    radius = 50
    while True:
        frame = camera.getFrame()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pair = StereopairUtils.getCenterPairBlockMatching(frame, radius)
            height, width = frame.shape
            if pair is not None:
                counter += 1
                point_left = np.array(pair[0]).astype(float)
                point_right = np.array(pair[1]).astype(float)
                cv2.circle(frame, ((int)(width / 4), (int)(height / 2)), 5, (0, 0, 255))
                cv2.circle(frame, ((int)(width / 2) + (int)(point_right[0]), (int)(point_right[1])), 5, (0, 0, 255))
                coord = StereoUtils.getWorldCoordinates(P_1, P_2, point_left, point_right)
                ImageUtils.imageDrawDistance(frame, coord[2])
                print(coord)
            resized = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            cv2.imshow("Virutal Camera", resized)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            return


# loadImagesAndDoBlockMatching()
# openVideoAndGetDistanceBlockMatching()
# loadImagesAndDoEplines()
openImageAndGetDistance()

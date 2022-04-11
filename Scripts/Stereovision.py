import numpy as np

import Utils.StereoUtils as StereoUtils
import Utils.FileUtils as FileUtils
import Objects.Camera as Camera
import Utils.ImageUtils as ImageUtils
from Objects.Timer import Timer
import Utils.StereopairUtils as StereopairUtils
import cv2


def loadImagesAndGetDistance():
    P_1, P_2 = FileUtils.loadProjectionMatrix()
    images = FileUtils.getFramesImages()
    for image in images:
        image_left, image_right = StereoUtils.splitMergedImage(image)
        ret, corners_left = cv2.findChessboardCornersSB(image_left, (9, 14))
        ret, corners_right = cv2.findChessboardCornersSB(image_right, (9, 14))
        point_left = corners_left[(int)(len(corners_left) / 2)][0]
        point_right = corners_right[(int)(len(corners_right) / 2)][0]
        print(point_left, point_right)
        coord = StereoUtils.getWorldCoordinates(P_1, P_2, point_left, point_right)
        print(coord)


def loadImagesAndGetDistanceToTheCenter():
    P_1, P_2 = FileUtils.loadProjectionMatrix()
    F = FileUtils.loadFundamentalMatrix()
    images = FileUtils.getFramesImages()
    counter = 1
    for image in images:
        counter += 1
        image_left, image_right = StereoUtils.splitMergedImage(image)
        height, width = image_left.shape
        pair = StereopairUtils.getAnyPairs(image)
        if pair is not None and len(pair) != 0:
            pair = pair[0]
            print("Image", counter)
            point_left = np.array(pair[0]).astype(float)[0]
            point_right = np.array(pair[1]).astype(float)[0]
            coord = StereoUtils.getWorldCoordinates(P_1, P_2, point_left, point_right)
            cv2.circle(image, ((int)(width / 2), (int)(height / 2)), 10, (0, 0, 255))
            cv2.circle(image, (width + (int)(point_right[0]), (int)(point_right[1])), 10, (0, 0, 255))
            cv2.imwrite("../Frames/stereo_pair_" + str(counter) + ".png", image)


def openVideoAndGetDistanceToTheCenter():
    camera = Camera.VirtualCamera(0, 3840, 1080)
    P_1, P_2 = FileUtils.loadProjectionMatrix()
    F = FileUtils.loadFundamentalMatrix()
    counter = 0
    while True:
        frame = camera.getLastFrame()
        left_frame, right_frame = StereoUtils.splitMergedImage(frame)
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width, alpha = left_frame.shape
            pair = StereopairUtils.getCenterPair(frame, F)
            if pair is not None:
                counter += 1
                point_left = np.array(pair[0]).astype(float)
                point_right = np.array(pair[1]).astype(float)
                cv2.circle(frame, ((int)(width / 2), (int)(height / 2)), 5, (0, 0, 255))
                cv2.circle(frame, (width + (int)(point_right[0]), (int)(point_right[1])), 5, (0, 0, 255))
                # cv2.imwrite("../Frames/stereo_pair_" + str(counter) + ".png", frame)
                coord = StereoUtils.getWorldCoordinates(P_1, P_2, point_left, point_right)
                ImageUtils.imageDrawDistance(left_frame, coord[2])
                # print(coord)
            else:
                ImageUtils.imageDrawDistance(left_frame, -1)
            resized = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            cv2.imshow("Virutal Camera", resized)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            return


def openVideoAndGetDistanceToAnyKeyPoint():
    camera = Camera.VirtualCamera(0, 3840, 1080)
    P_1, P_2 = FileUtils.loadProjectionMatrix()
    F = FileUtils.loadFundamentalMatrix()
    counter = 0
    stereo = cv2.StereoBM_create(0, 21)
    while True:
        frame = camera.getLastFrame()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            left_frame, right_frame = StereoUtils.splitMergedImage(frame)
            height, width = left_frame.shape
            depth = stereo.compute(left_frame, right_frame)
            cv2.imshow("Frame", depth)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            return


def loadImagesAndDoBlockMatching():
    P_1, P_2 = FileUtils.loadProjectionMatrix()
    timer = Timer()
    images = FileUtils.getFramesImages()
    counter = 1
    radius = 50
    for image in images:
        counter += 1
        height, width = image.shape
        timer.refreshTimer()
        pair = StereopairUtils.getCenterPairBlockMatching(image, radius)
        timer.printTime("BM Pair")

        a = np.array(pair[0]) - (radius, radius)
        b = np.array(pair[0]) + (radius, radius)
        cv2.rectangle(image, (a[0], a[1]), (b[0], b[1]), (0, 0, 255), 5)
        a = np.array(pair[1]) - (radius, radius) + (int(width / 2), 0)
        b = np.array(pair[1]) + (radius, radius) + (int(width / 2), 0)
        cv2.rectangle(image, (a[0], a[1]), (b[0], b[1]), (0, 0, 255), 5)

        cv2.imwrite("../Result/stereo_pair_" + str(counter) + ".png", image)

        point_left = np.array(pair[0]).astype(float)
        point_right = np.array(pair[1]).astype(float)


# loadImagesAndGetDistance()
# loadImagesAndGetDistanceToTheCenter()
# openVideoAndGetDistanceToTheCenter()
# openVideoAndGetDistanceToAnyKeyPoint()
loadImagesAndDoBlockMatching()

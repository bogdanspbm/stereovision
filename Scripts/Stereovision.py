import numpy as np

import Utils.StereoUtils as StereoUtils
import Utils.FileUtils as FileUtils
import Objects.Camera as Camera
import Utils.ImageUtils as ImageUtils
import Utils.MathUtils as MathUtils
from Objects.Timer import Timer
import Utils.StereopairUtils as StereopairUtils
import cv2

'''
This is a main script which has 4 methods
Two of them work with pre-taken pictures and two of them work with a video stream
All scripts focus on distance measuring
'''

'''
This method loads images from 'Frames' folder
measures distance using block matching algorythm
and saves a result
'''


def openImagesAndGetDistanceBM():
    timer = Timer()
    images = FileUtils.getFramesImages()
    counter = 1
    radius = 25

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

        cv2.imwrite("../Result/distance_" + str(counter) + ".jpg", image)


'''
This method loads images from 'Frames' folder
measures distance using epipolar line compare algorythm
and saves a result
'''


def openImagesAndGetDistanceEpipolar():
    P_1, P_2 = FileUtils.loadProjectionMatrix()

    timer = Timer()
    images = FileUtils.getFramesImages()
    counter = 1
    for gray in images:
        height, width = gray.shape

        timer.refreshTimer()
        pair = StereopairUtils.getCenterPairEpipolar(gray)
        timer.printTime("EPIPOLAR")

        point_left = np.array(pair[0]).astype(float)
        point_right = np.array(pair[1]).astype(float)

        cv2.circle(gray, ((int)(width / 4), (int)(height / 2)), 5, (0, 0, 255))
        cv2.circle(gray, ((int)(width / 2) + (int)(point_right[0]), (int)(point_right[1])), 5, (0, 0, 255))
        coord = StereoUtils.getWorldCoordinates(P_1, P_2, point_left, point_right)
        dist = MathUtils.getVectorNorm(coord)
        ImageUtils.imageDrawDistance(gray, dist)

        cv2.imwrite("../Result/distance_" + str(counter) + ".jpg", gray)
        counter += 1


'''
This method opens a video stream
measures distance using block matching algorythm
and saves a result
'''


def openVideoAndGetDistanceBlockMatching():
    camera = Camera.VirtualCamera(0, 3840, 1080)
    P_1, P_2 = FileUtils.loadProjectionMatrix()
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


'''
This method opens a video stream
measures distance using epipolar line compare algorythm
and saves a result
'''


def openVideoAndGetDistanceEpipolar():
    camera = Camera.VirtualCamera(0, 3840, 1080)
    P_1, P_2 = FileUtils.loadProjectionMatrix()
    counter = 0
    while True:
        frame = camera.getFrame()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pair = StereopairUtils.getCenterPairEpipolar(frame)
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

# openImagesAndGetDistanceEpipolar()

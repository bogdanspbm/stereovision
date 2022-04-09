import numpy as np

import Utils.StereoUtils as StereoUtils
import Utils.FileUtils as FileUtils
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
        print(point_left,point_right)
        coord = StereoUtils.getWorldCoordinates(P_1, P_2, point_left, point_right)
        print(coord)


def getDistanceToTheCenter():
    P_1, P_2 = FileUtils.loadProjectionMatrix()
    F = FileUtils.loadFundamentalMatrix()
    images = FileUtils.getFramesImages()
    counter = 1
    for image in images:
        counter += 1
        # image_left, image_right = StereoUtils.splitMergedImage(image)
        pair = StereopairUtils.getCenterPair(image, F)
        if pair is not None:
            print("Image", counter)
            point_left = np.array(pair[0]).astype(float)
            point_right = np.array(pair[1]).astype(float)
            coord = StereoUtils.getWorldCoordinates(P_1, P_2, point_left, point_right)
            print(point_left)
            print(point_right)
            print(coord)


#loadImagesAndGetDistance()
getDistanceToTheCenter()

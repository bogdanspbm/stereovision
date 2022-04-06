import cv2

from Utils.FileUtils import getFramesImages
from Utils.StereoUtils import splitMergedImage, getProjectionMatrixCalibrated
import numpy as np


class Calibrator():
    def __init__(self, images=None, rows=9, columns=14, square_size=6):
        if images is None:
            images = getFramesImages()
        self.images = images
        self.__splitImages()
        self.rows = rows
        self.columns = columns
        self.square_size = square_size

    def calibrate(self):
        self.__getCorners()
        self.__generateWorldCoordinates()
        self.__calibrateInternalParams()

    def __calibrateInternalParams(self):
        width, height = self.images_left[0].shape

        N = len(self.corners_left)
        obj_points = []
        for i in range(N):
            obj_points.append(self.object_points)

        retval, leftMatrix, leftCoeffs, leftR, leftT = cv2.calibrateCamera(obj_points, self.corners_left,
                                                                           (width, height), None, None)

        width, height = self.images_right[0].shape
        retval, rightMatrix, rightCoeffs, rightR, rightT = cv2.calibrateCamera(obj_points, self.corners_right,
                                                                               (width, height), None, None)

        flags = 0
        flags |= cv2.CALIB_SAME_FOCAL_LENGTH

        retval, leftMatrix, leftCoeffs, rightMatrix, rightCoeffs, R, T, E, F = \
            cv2.stereoCalibrate(obj_points, self.corners_left, self.corners_right,
                                leftMatrix, leftCoeffs, rightMatrix,
                                rightCoeffs, (width, height), flags=flags)

        self.internal_left = leftMatrix
        self.internal_right = rightMatrix

        self.P_1, self.P_2 = getProjectionMatrixCalibrated(leftMatrix, leftMatrix, R, T)


    def __splitImages(self):
        self.images_left = []
        self.images_right = []

        for image in self.images:
            image_l, image_r = splitMergedImage(image)
            self.images_left.append(image_l)
            self.images_right.append(image_r)

    def __generateWorldCoordinates(self):
        # Генерация мировых координат
        objp = np.zeros((1, self.rows * self.columns, 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:self.rows, 0:self.columns].T.reshape(-1, 2)
        tmp = objp[0, :, 0].copy()
        objp[0, :, 0] = objp[0, :, 1].copy()
        objp[0, :, 1] = tmp.copy()
        objp = self.square_size * objp
        self.object_points = objp.copy()

    def __getCorners(self):
        self.corners_left = []
        for image in self.images_left:
            ret, corners = cv2.findChessboardCornersSB(image, (self.rows, self.columns))
            self.corners_left.append(corners)

        self.corners_right = []
        for image in self.images_right:
            ret, corners = cv2.findChessboardCornersSB(image, (self.rows, self.columns))
            self.corners_right.append(corners)

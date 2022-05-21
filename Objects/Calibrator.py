import cv2

from Utils.FileUtils import getFramesImages, saveProjectionMatrix, saveFundamentalMatrix, saveDistorCoeffs, \
    saveCameraMatrix, saveRectifyMap
from Utils.StereoUtils import getProjectionMatrixCalibrated
from Utils.ImageUtils import splitMergedImage
import numpy as np

'''
This class implements default Stereo System calibration
The constructor receives next parameters:
---------------------------------------------
images - an array of images for calibrating
rows - the amount of rows on calibrating chessboard
columns - the amount of columns on calibrating chessboard
square_size - the size of square in real life metrics (for instance cm)
'''


class Calibrator():
    def __init__(self, images=None, rows=9, columns=14, square_size=6):
        if images is None:
            images = getFramesImages()
        self.images = images
        self.__splitImages()
        self.rows = rows
        self.columns = columns
        self.square_size = square_size

    '''
    This method executes full calibration process and saves the result of calibration
    You can also set use_rectify True to calibrate and save Rectify matrices
    '''

    def calibrate(self, use_rectify=False):
        self.__getCorners()
        self.__generateWorldCoordinates()
        self.__calibrateInternalParams()

        if use_rectify:
            self.calibrateRectify()

    '''
    This method calculates only internal camera parameters
    '''

    def __calibrateInternalParams(self):
        self.height, self.width = self.images_left[0].shape

        N = len(self.corners_left)
        obj_points = []
        for i in range(N):
            obj_points.append(self.object_points)

        retval, leftMatrix, leftCoeffs, leftR, leftT = cv2.calibrateCamera(obj_points, self.corners_left,
                                                                           (self.width, self.height), None, None)

        self.height, self.width = self.images_right[0].shape
        retval, rightMatrix, rightCoeffs, rightR, rightT = cv2.calibrateCamera(obj_points, self.corners_right,
                                                                               (self.width, self.height), None, None)

        flags = 0
        flags |= cv2.CALIB_SAME_FOCAL_LENGTH

        retval, leftMatrix, leftCoeffs, rightMatrix, rightCoeffs, R, T, E, F = \
            cv2.stereoCalibrate(obj_points, self.corners_left, self.corners_right,
                                leftMatrix, leftCoeffs, rightMatrix,
                                rightCoeffs, (self.width, self.height), flags=flags)

        print("Calibrate ret value is: " + str(retval))
        print("Used images count is: " + str(len(self.corners_left)))

        self.rotation = R
        self.translate = T
        self.dist_left = leftCoeffs
        self.dist_right = rightCoeffs

        self.internal_left = leftMatrix
        self.internal_right = rightMatrix
        self.F = F

        self.P_1, self.P_2 = getProjectionMatrixCalibrated(leftMatrix, leftMatrix, R, T)

    '''
    This method calculates only rectify camera parameters
    '''

    def __calibrateRectify(self):
        rectify_scale = 1
        leftR, rightR, self.P1, self.P2, Q, roiL, roiR = cv2.stereoRectify(self.internal_left, self.dist_left,
                                                                           self.internal_right, self.dist_right,
                                                                           (self.width, self.height), self.rotation,
                                                                           self.translate,
                                                                           rectify_scale, (0, 0))

        self.leftStereoMap = cv2.initUndistortRectifyMap(self.internal_left, self.dist_left, leftR, self.P1,
                                                         (self.width, self.height), cv2.CV_16SC2)

        self.rightStereoMap = cv2.initUndistortRectifyMap(self.internal_right, self.dist_right, rightR, self.P2,
                                                          (self.width, self.height), cv2.CV_16SC2)

    '''
    This method splits images onto 2 arrays of left and right images
    '''

    def __splitImages(self):
        self.images_left = []
        self.images_right = []

        for image in self.images:
            image_l, image_r = splitMergedImage(image)
            self.images_left.append(image_l)
            self.images_right.append(image_r)

    '''
    This method generates world coordinates of chessboard points
    '''

    def __generateWorldCoordinates(self):
        # Генерация мировых координат
        objp = np.zeros((1, self.rows * self.columns, 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:self.rows, 0:self.columns].T.reshape(-1, 2)
        tmp = objp[0, :, 0].copy()
        objp[0, :, 0] = objp[0, :, 1].copy()
        objp[0, :, 1] = tmp.copy()
        objp = self.square_size * objp
        self.object_points = objp.copy()

    '''
    This method finds chessboard points from images
    '''

    def __getCorners(self):
        self.corners_left = []
        self.corners_right = []

        left_arr = []
        right_arr = []
        left_ret_arr = []
        right_ret_arr = []

        for image in self.images_left:
            left_ret, left_corners = cv2.findChessboardCornersSB(image, (self.rows, self.columns))
            left_ret_arr.append(left_ret)
            left_arr.append(left_corners)
            # self.corners_left.append(corners)

        for image in self.images_right:
            right_ret, right_corners = cv2.findChessboardCornersSB(image, (self.rows, self.columns))
            right_ret_arr.append(right_ret)
            right_arr.append(right_corners)
            # self.corners_right.append(corners)

        for i in range(len(right_ret_arr)):
            if left_ret_arr[i] == True and right_ret_arr[i] == True:
                self.corners_left.append(left_arr[i])
                self.corners_right.append(right_arr[i])

    '''
    This method exports calibration result into settings files
    '''

    def exportCalibration(self):
        saveProjectionMatrix(self.P_1, self.P_2)
        saveFundamentalMatrix(self.F)
        saveDistorCoeffs(self.dist_left, self.dist_right)
        saveCameraMatrix(self.internal_left, self.internal_right)

        if self.leftStereoMap is not None:
            saveRectifyMap(self.leftStereoMap, self.rightStereoMap)

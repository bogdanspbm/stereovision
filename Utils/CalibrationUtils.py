import cv2


class Calibrator():
    def __init__(self, images, rows=9, columns=14):
        self.images = images
        self.rows = rows
        self.columns = columns

    def calibrate(self):
        self.__getCorners()

    def __getCorners(self):
        tmp = cv2.cvtColor(self.images, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(tmp, (self.rows, self.columns))
        self.corners = corners
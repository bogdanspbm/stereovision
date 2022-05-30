import cv2

'''
This is an implementation of KeyPoint for BRIEF detector
---------------------------------------------
coord - coorditanes of the key point
param - a param vector of the key point
'''


class KeyPoint():
    def __init__(self, coord, param):
        self.coord = coord
        self.param = param

    def getCoord(self):
        return (self.coord[0], self.coord[1])

    def getParam(self):
        return self.param


'''
This method calculates an Hausdorf distance between two key points
'''


def getHausdorfDistance(keypoin_a, keypoint_b):
    vec_a = keypoin_a.getParam()
    vec_b = keypoint_b.getParam()
    distance = 0
    for i in range(len(vec_a)):
        if vec_a[i] != vec_b[i]:
            dif = (int(vec_a[i]) - int(vec_b[i]))
            if dif > 0:
                distance += dif
            else:
                distance -= dif
    return distance


'''
This method converts an array of key points into an array of OpenCV key points
'''


def convertToCVKeypoint(array):
    result = []
    for point in array:
        try:
            result.append(cv2.KeyPoint((float)(point.getCoord()[0]), (float)(point.getCoord()[1]), 1))
        except:
            result.append(cv2.KeyPoint((float)(point[0]), (float)(point[1]), 1))
    return result

import cv2
import numpy as np

'''
This method calculates projection matrices for two cameras
in stereovision system using two property matrices and R|T matrices 
'''


def getProjectionMatrixCalibrated(A1, A2, R, T):
    RT = []

    for i in range(3):
        vec = [R[i][0], R[i][1], R[i][2], T[i][0]]
        RT.append(vec)
    RT_2 = np.array(RT)

    RT_1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    P_1 = np.matmul(A1, RT_1)
    P_2 = np.matmul(A2, RT_2)

    return P_1, P_2


'''
This method triangulates stereo pair into 3D world coordinates
using two projection matrices and stereo pair coordinates
'''

def getWorldCoordinates(P1, P2, CORD1, CORD2):
    pts4D = cv2.triangulatePoints(P1, P2, CORD1, CORD2)
    pts3D = pts4D[:3, :] / np.repeat(pts4D[3, :], 3).reshape(3, -1)
    return pts3D

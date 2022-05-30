from os import listdir
from cv2 import imread, FileStorage, FILE_STORAGE_WRITE, FILE_STORAGE_READ
import pandas as pd
import numpy as np

'''
This method reads images from Frames directory
'''


def getFramesImages():
    arr = []
    path = "../Frames"
    images_list = listdir(path)

    for im_name in images_list:
        img = imread(path + "/" + im_name, 0)
        arr.append(img)

    return arr


'''
This method saves input projection matrices into special files
'''


def saveProjectionMatrix(P1, P2):
    df = pd.DataFrame(data=P1.astype(float))
    df.to_csv('../config/projection_left.csv', sep=' ', header=False, float_format='%.6f', index=False)
    df = pd.DataFrame(data=P2.astype(float))
    df.to_csv('../config/projection_right.csv', sep=' ', header=False, float_format='%.6f', index=False)
    print("Projection matrix are saved...")


'''
This method loads projection matrices from special files
'''


def loadProjectionMatrix():
    P1 = np.genfromtxt('../config/projection_left.csv', delimiter=' ')
    P2 = np.genfromtxt('../config/projection_right.csv', delimiter=' ')
    return P1, P2


'''
This method saves input camera matrices into special files
'''


def saveCameraMatrix(P1, P2):
    df = pd.DataFrame(data=P1.astype(float))
    df.to_csv('../config/camera_left.csv', sep=' ', header=False, float_format='%.6f', index=False)
    df = pd.DataFrame(data=P2.astype(float))
    df.to_csv('../config/camera_right.csv', sep=' ', header=False, float_format='%.6f', index=False)
    print("Camera matrix are saved...")


'''
This method loads camera matrices from special files
'''


def loadCameraMatrix():
    P1 = np.genfromtxt('../config/camera_left.csv', delimiter=' ')
    P2 = np.genfromtxt('../config/camera_right.csv', delimiter=' ')
    return P1, P2


'''
This method saves input fundamental matrix into special file
'''


def saveFundamentalMatrix(F):
    df = pd.DataFrame(data=F.astype(float))
    df.to_csv('../config/fundamental_matrix.csv', sep=' ', header=False, float_format='%.6f', index=False)


'''
This method saves input distor coeffs into special files
'''


def saveDistorCoeffs(D1, D2):
    df = pd.DataFrame(data=D1.astype(float))
    df.to_csv('../config/distor_left.csv', sep=' ', header=False, float_format='%.6f', index=False)
    df = pd.DataFrame(data=D2.astype(float))
    df.to_csv('../config/distor_right.csv', sep=' ', header=False, float_format='%.6f', index=False)
    print("Dist matrix are saved...")


'''
This method loads distor coeffs from special files
'''


def loadDistortionMatrix():
    D1 = np.genfromtxt('../config/distor_left.csv', delimiter=' ')
    D2 = np.genfromtxt('../config/distor_right.csv', delimiter=' ')
    return D1, D2


'''
This method loads fundamental matrix from special file
'''


def loadFundamentalMatrix():
    F = np.genfromtxt('../config/fundamental_matrix.csv', delimiter=' ')
    return F


'''
This method loads distances vectors from special file
'''


def loadDistancesForTest():
    mat = np.genfromtxt('../config/distance_test.csv', delimiter=';')
    return mat


'''
This method saves rectify map into special file
'''


def saveRectifyMap(left_map, right_map):
    cv_file = FileStorage("../config/stereo_map.xml", FILE_STORAGE_WRITE)
    cv_file.write("Left_Stereo_Map_x", left_map[0])
    cv_file.write("Left_Stereo_Map_y", left_map[1])
    cv_file.write("Right_Stereo_Map_x", right_map[0])
    cv_file.write("Right_Stereo_Map_y", right_map[1])
    cv_file.release()


'''
This method loads rectify map from special file
'''


def loadRectifyMap():
    cv_file = FileStorage("../config/stereo_map.xml", FILE_STORAGE_READ)

    a = cv_file.getNode("Left_Stereo_Map_x").mat()
    b = cv_file.getNode("Left_Stereo_Map_y").mat()
    left_map = (a, b)

    a = cv_file.getNode("Right_Stereo_Map_x").mat()
    b = cv_file.getNode("Right_Stereo_Map_y").mat()
    right_map = (a, b)

    cv_file.release()

    return left_map, right_map

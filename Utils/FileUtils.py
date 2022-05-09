from os import listdir
from cv2 import imread
import pandas as pd
import numpy as np


def getFramesImages():
    arr = []
    path = "../Frames"
    images_list = listdir(path)

    for im_name in images_list:
        img = imread(path + "/" + im_name, 0)
        arr.append(img)

    return arr


def saveProjectionMatrix(P1, P2):
    df = pd.DataFrame(data=P1.astype(float))
    df.to_csv('../config/projection_left.csv', sep=' ', header=False, float_format='%.6f', index=False)
    df = pd.DataFrame(data=P2.astype(float))
    df.to_csv('../config/projection_right.csv', sep=' ', header=False, float_format='%.6f', index=False)
    print("Projection matrix are saved...")


def loadProjectionMatrix():
    P1 = np.genfromtxt('../config/projection_left.csv', delimiter=' ')
    P2 = np.genfromtxt('../config/projection_right.csv', delimiter=' ')
    return P1, P2

def saveCameraMatrix(P1, P2):
    df = pd.DataFrame(data=P1.astype(float))
    df.to_csv('../config/camera_left.csv', sep=' ', header=False, float_format='%.6f', index=False)
    df = pd.DataFrame(data=P2.astype(float))
    df.to_csv('../config/camera_right.csv', sep=' ', header=False, float_format='%.6f', index=False)
    print("Camera matrix are saved...")


def loadCameraMatrix():
    P1 = np.genfromtxt('../config/camera_left.csv', delimiter=' ')
    P2 = np.genfromtxt('../config/camera_right.csv', delimiter=' ')
    return P1, P2


def saveFundamentalMatrix(F):
    df = pd.DataFrame(data=F.astype(float))
    df.to_csv('../config/fundamental_matrix.csv', sep=' ', header=False, float_format='%.6f', index=False)


def saveDistorCoeffs(D1, D2):
    df = pd.DataFrame(data=D1.astype(float))
    df.to_csv('../config/distor_left.csv', sep=' ', header=False, float_format='%.6f', index=False)
    df = pd.DataFrame(data=D2.astype(float))
    df.to_csv('../config/distor_right.csv', sep=' ', header=False, float_format='%.6f', index=False)
    print("Dist matrix are saved...")


def loadDistortionMatrix():
    D1 = np.genfromtxt('../config/distor_left.csv', delimiter=' ')
    D2 = np.genfromtxt('../config/distor_right.csv', delimiter=' ')
    return D1, D2


def loadFundamentalMatrix():
    F = np.genfromtxt('../config/fundamental_matrix.csv', delimiter=' ')
    return F

def loadDistancesForTest():
    mat = np.genfromtxt('../config/distance_test.csv', delimiter=';')
    return mat
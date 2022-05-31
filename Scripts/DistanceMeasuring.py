from Utils.FileUtils import loadDistancesForTest, loadProjectionMatrix
from Utils.StereoUtils import getWorldCoordinates
from Utils.MathUtils import getVectorNorm
import numpy as np
import matplotlib.pyplot as plt

'''
This is a script for calculating a stereo vision distance error
It loads hand collected data from a file and draws a comparing plot
'''


mat = loadDistancesForTest()
P_1, P_2 = loadProjectionMatrix()

A = []
B = []

for vec in mat:
    real_distance = vec[0]
    point_left = np.array([vec[1], vec[2]])
    point_right = np.array([vec[3], vec[4]])
    coord = getWorldCoordinates(P_1, P_2, point_left, point_right)
    dist = getVectorNorm(coord) / 100
    A.append(real_distance)
    B.append(dist)

plt.plot(A, label="Реальная дистанция")
plt.plot(B, label="Стерео дистанция")
plt.legend()
plt.show()

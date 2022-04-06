from os import listdir
from cv2 import imread


def getFramesImages():
    arr = []
    path = "../Frames"
    images_list = listdir(path)

    for im_name in images_list:
        img = imread(path + "/" + im_name, 0)
        arr.append(img)

    return arr

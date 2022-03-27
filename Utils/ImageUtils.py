import cv2


def removeImageBorder(image, border = 0):
    if image is None:
        return None

    height = image.shape[0]
    width = image.shape[1]

    s1 = image[:, :width - border]
    s2 = s1[:, border:]


    return s2
